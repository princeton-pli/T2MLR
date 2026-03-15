#!/usr/bin/env python3
"""
Generate a shortest-path-finding dataset on DAGs with dead ends.

The task: given a graph as shuffled directed edges plus start/end nodes,
generate the unique shortest path.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


SYLLABLES = [
    "ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu",
    "da", "de", "di", "do", "du", "fa", "fe", "fi", "fo", "fu",
    "ga", "ge", "gi", "go", "gu", "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu",
    "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro", "ru",
    "sa", "se", "si", "so", "su", "ta", "te", "ti", "to", "tu",
    "va", "ve", "vi", "vo", "vu", "za", "ze", "zi", "zo", "zu",
]


def generate_unique_names(k: int, rng: random.Random) -> List[str]:
    out: Set[str] = set()
    suffixes = ["x", "n", "r", "s", "t", "l", "m", "v"]
    while len(out) < k:
        n_syl = 2 if rng.random() < 0.7 else 3
        parts = [rng.choice(SYLLABLES) for _ in range(n_syl)]
        out.add("".join(parts) + rng.choice(suffixes))
    return list(out)


@dataclass
class PathfindingConfig:
    num_nodes: int = 100
    path_length_min: int = 10
    path_length_max: int = 18
    num_parallel_lanes: int = 6
    lane_extra_length_min: int = 1
    lane_extra_length_max: int = 3
    dead_end_prob: float = 0.2
    dead_end_depth: int = 2
    add_wait_tokens: bool = True
    add_random_path_prefix: bool = False
    random_path_divider: str = "|"
    random_path_max_steps: int = 30


class PathfindingGeneratorV2:
    def __init__(self, config: PathfindingConfig, rng: random.Random):
        self.config = config
        self.rng = rng

    def _build_graph(self) -> Tuple[List[Tuple[int, int]], List[int], int, int, int, Set[int]]:
        cfg = self.config
        edges: List[Tuple[int, int]] = []
        edge_set: Set[Tuple[int, int]] = set()

        def add_edge(u: int, v: int) -> None:
            if (u, v) not in edge_set and u != v:
                edges.append((u, v))
                edge_set.add((u, v))

        path_length = self.rng.randint(cfg.path_length_min, cfg.path_length_max)
        backbone = list(range(path_length + 1))
        start, end = backbone[0], backbone[-1]

        for i in range(len(backbone) - 1):
            add_edge(backbone[i], backbone[i + 1])

        next_node = path_length + 1
        can_reach_end: Set[int] = set(backbone)

        for _ in range(cfg.num_parallel_lanes):
            branch_at = self.rng.randint(0, path_length // 2)
            merge_at = self.rng.randint(path_length // 2 + 1, path_length - 1)
            extra = self.rng.randint(cfg.lane_extra_length_min, cfg.lane_extra_length_max)
            lane_length = (merge_at - branch_at) + extra

            lane_nodes = []
            for _ in range(lane_length):
                lane_nodes.append(next_node)
                can_reach_end.add(next_node)
                next_node += 1

            add_edge(backbone[branch_at], lane_nodes[0])
            for i in range(len(lane_nodes) - 1):
                add_edge(lane_nodes[i], lane_nodes[i + 1])
            add_edge(lane_nodes[-1], backbone[merge_at])

        reachable_nodes = list(can_reach_end - {end})
        for src in reachable_nodes:
            if self.rng.random() < cfg.dead_end_prob:
                current = next_node
                next_node += 1
                add_edge(src, current)
                for _ in range(cfg.dead_end_depth - 1):
                    nxt = next_node
                    next_node += 1
                    add_edge(current, nxt)
                    current = nxt

        return edges, backbone, next_node, start, end, can_reach_end

    def _compute_shortest_path_bfs(
        self,
        edges: List[Tuple[int, int]],
        num_nodes: int,
        start: int,
        end: int,
    ) -> Optional[List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for src, dst in edges:
            adj[src].append(dst)

        visited = {start}
        parent: Dict[int, int] = {}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == end:
                break
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)

        if end not in visited:
            return None

        path = [end]
        current = end
        while current != start:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    def _count_shortest_paths(
        self,
        edges: List[Tuple[int, int]],
        num_nodes: int,
        start: int,
        end: int,
    ) -> Tuple[int, int]:
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for src, dst in edges:
            adj[src].append(dst)

        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        if end not in dist:
            return 0, 0

        shortest_dist = dist[end]
        num_paths = {start: 1}
        for d in range(shortest_dist + 1):
            nodes_at_d = [n for n, dd in dist.items() if dd == d]
            for node in nodes_at_d:
                if node not in num_paths:
                    num_paths[node] = 0
                for neighbor in adj[node]:
                    if neighbor in dist and dist[neighbor] == dist[node] + 1:
                        num_paths[neighbor] = num_paths.get(neighbor, 0) + num_paths[node]

        return shortest_dist, num_paths.get(end, 0)

    def _generate_random_walk(self, edges: List[Tuple[int, int]], num_nodes: int, start: int) -> List[int]:
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for src, dst in edges:
            adj[src].append(dst)

        path = [start]
        current = start
        for _ in range(self.config.random_path_max_steps):
            neighbors = adj[current]
            if not neighbors:
                break
            current = self.rng.choice(neighbors)
            path.append(current)
        return path

    def _make_example(self) -> Optional[Dict[str, object]]:
        edges, backbone, num_nodes, start, end, _ = self._build_graph()
        shortest = self._compute_shortest_path_bfs(edges, num_nodes, start, end)
        if shortest is None:
            return None

        shortest_dist, num_shortest = self._count_shortest_paths(edges, num_nodes, start, end)
        if num_shortest != 1:
            return None

        names = generate_unique_names(num_nodes, self.rng)
        named_edges = [(names[u], names[v]) for u, v in edges]
        self.rng.shuffle(named_edges)

        start_name = names[start]
        end_name = names[end]
        answer_path = [names[i] for i in shortest]

        prompt = (
            "Find the shortest path in this directed graph.\n"
            f"Start: {start_name}\n"
            f"End: {end_name}\n"
            "Edges:\n"
            + "\n".join(f"{src} -> {dst}" for src, dst in named_edges)
            + "\nAnswer: "
        )

        answer = " -> ".join(answer_path)
        if self.config.add_wait_tokens:
            answer = "Wait " + answer

        if self.config.add_random_path_prefix:
            random_walk = self._generate_random_walk(edges, num_nodes, start)
            random_prefix = " -> ".join(names[i] for i in random_walk)
            answer = f"{random_prefix} {self.config.random_path_divider} {answer}"

        return {
            "question": prompt,
            "response": answer,
            "answer": " -> ".join(answer_path),
            "edges": named_edges,
            "start": start_name,
            "end": end_name,
            "meta": {
                "num_nodes": num_nodes,
                "shortest_length": shortest_dist,
                "num_shortest_paths": num_shortest,
                "backbone_length": len(backbone),
            },
        }

    def make_dataset(self, size: int, max_attempts_multiplier: int = 50) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        attempts = 0
        max_attempts = max(size * max_attempts_multiplier, size)
        while len(rows) < size and attempts < max_attempts:
            attempts += 1
            ex = self._make_example()
            if ex is not None:
                rows.append(ex)
            if attempts % 1000 == 0:
                print(f"[progress] attempts={attempts} rows={len(rows)}", file=sys.stderr)
        if len(rows) < size:
            raise RuntimeError(f"Only generated {len(rows)}/{size} examples after {attempts} attempts.")
        return rows


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("data/pathfinding"))
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--valid", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_nodes", type=int, default=100)
    p.add_argument("--path_length_min", type=int, default=10)
    p.add_argument("--path_length_max", type=int, default=18)
    p.add_argument("--num_parallel_lanes", type=int, default=6)
    p.add_argument("--lane_extra_length_min", type=int, default=1)
    p.add_argument("--lane_extra_length_max", type=int, default=3)
    p.add_argument("--dead_end_prob", type=float, default=0.2)
    p.add_argument("--dead_end_depth", type=int, default=2)
    p.add_argument("--add_wait_tokens", action="store_true")
    p.add_argument("--add_random_path_prefix", action="store_true")
    p.add_argument("--random_path_divider", type=str, default="|")
    p.add_argument("--random_path_max_steps", type=int, default=30)
    args = p.parse_args()

    rng = random.Random(args.seed)
    config = PathfindingConfig(
        num_nodes=args.num_nodes,
        path_length_min=args.path_length_min,
        path_length_max=args.path_length_max,
        num_parallel_lanes=args.num_parallel_lanes,
        lane_extra_length_min=args.lane_extra_length_min,
        lane_extra_length_max=args.lane_extra_length_max,
        dead_end_prob=args.dead_end_prob,
        dead_end_depth=args.dead_end_depth,
        add_wait_tokens=args.add_wait_tokens,
        add_random_path_prefix=args.add_random_path_prefix,
        random_path_divider=args.random_path_divider,
        random_path_max_steps=args.random_path_max_steps,
    )
    generator = PathfindingGeneratorV2(config, rng)

    t0 = time.time()
    train_rows = generator.make_dataset(args.train)
    valid_rows = generator.make_dataset(args.valid)
    test_rows = generator.make_dataset(args.test)

    out_dir = args.out_dir
    _write_json(out_dir / "pathfinding_train.json", train_rows)
    _write_json(out_dir / "pathfinding_valid.json", valid_rows)
    _write_json(out_dir / "pathfinding_test.json", test_rows)

    elapsed = time.time() - t0
    print(f"Wrote dataset to {out_dir} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
