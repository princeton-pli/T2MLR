#!/usr/bin/env python3
"""
Generate the variable-assignment dataset used by the T2MLR variable-assignment task.

Each example contains:
- `input`: prompt text
- `target`: final answer text
- `task`: always `variable_assignment`
- `meta`: generation metadata
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
from typing import Any, Dict, List, Optional, Tuple


def _rng(seed: Optional[int] = None, rng: Optional[random.Random] = None) -> random.Random:
    return rng if rng is not None else random.Random(seed)


def _var_name(idx: int, alphabet: str = string.ascii_lowercase) -> str:
    base = len(alphabet)
    n = idx + 1
    out: List[str] = []
    while n > 0:
        n, rem = divmod(n - 1, base)
        out.append(alphabet[rem])
    return "".join(reversed(out))


def _sample_vars(r: random.Random, n: int, pool: str = string.ascii_lowercase) -> List[str]:
    if n <= len(pool):
        return r.sample(list(pool), n)
    expanded = [_var_name(i, pool) for i in range(n)]
    return r.sample(expanded, n)


def gen_variable_assignment(
    *,
    n_base: int = 8,
    depth: int = 2,
    n_distractor_aliases: int = 6,
    value_range: Tuple[int, int] = (0, 9),
    form: str = "basic",
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    r = _rng(seed, rng)
    if depth < 0:
        raise ValueError("depth must be >= 0")

    n_chain_vars = max(1, depth + 1)
    total_vars = n_base + n_chain_vars + n_distractor_aliases
    vars_ = _sample_vars(r, total_vars)

    base_vars = vars_[:n_base]
    chain_vars = vars_[n_base : n_base + n_chain_vars]
    distractor_vars = vars_[n_base + n_chain_vars :]

    lo, hi = value_range
    base_vals = {v: r.randint(lo, hi) for v in base_vars}
    sink = r.choice(base_vars)

    assignments: List[Tuple[str, str]] = []
    if depth == 0:
        query_var = sink
    else:
        query_var = chain_vars[0]
        for i in range(depth):
            left = chain_vars[i]
            right = chain_vars[i + 1] if (i + 1) < len(chain_vars) else sink
            assignments.append((left, right))
        if assignments and assignments[-1][1] != sink:
            assignments.append((assignments[-1][1], sink))

    pool = base_vars + chain_vars
    order = pool + distractor_vars
    r.shuffle(order)

    for dv in distractor_vars:
        rhs = r.choice([x for x in order if x != dv])
        assignments.append((dv, rhs))

    base_kvs = [(v, base_vals[v]) for v in base_vars]
    alias_kvs = assignments[:]

    def resolve(v: str) -> int:
        seen = set()
        amap = {a: b for a, b in alias_kvs}
        while v in amap:
            if v in seen:
                raise RuntimeError("cycle detected")
            seen.add(v)
            v = amap[v]
        if v not in base_vals:
            return base_vals[sink]
        return base_vals[v]

    target_val = resolve(query_var)

    all_lines: List[str] = []
    for v, val in base_kvs:
        all_lines.append(f"{v}={val}")
    for a, b in alias_kvs:
        all_lines.append(f"{a}={b}")
    r.shuffle(all_lines)

    if form == "basic":
        prompt = (
            "Variable assignment. Follow the assignments and fill in the blank.\n"
            + "\n".join(all_lines)
            + f"\n{query_var}=___\n"
            "Answer: | "
        )
        target = str(target_val)
    elif form == "math":
        prompt = (
            "A student writes down equations relating letter-variables.\n"
            "Each line means the left-hand side equals the right-hand side.\n"
            + "\n".join(all_lines)
            + f"\nQuestion: What is the numerical value of {query_var}?\n"
            "Answer: | "
        )
        target = str(target_val)
    elif form == "code":
        code_lines = []
        for line in all_lines:
            lhs, rhs = line.split("=")
            code_lines.append(f"{lhs.strip()} = {rhs.strip()}")
        prompt = (
            "Consider this short Python program:\n\n"
            + "\n".join(code_lines)
            + f"\n\nWhat is the value of {query_var}?\n"
            "Answer: | "
        )
        target = str(target_val)
    else:
        raise ValueError("form must be 'basic', 'math', or 'code'")

    return {
        "input": prompt,
        "target": target,
        "task": "variable_assignment",
        "meta": {
            "n_base": n_base,
            "depth": depth,
            "n_distractor_aliases": n_distractor_aliases,
            "value_range": value_range,
            "form": form,
            "query_var": query_var,
            "sink_var": sink,
        },
    }


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generate_split(n: int, rng: random.Random, args: argparse.Namespace, shuffle: bool) -> List[Dict[str, Any]]:
    rows = [
        gen_variable_assignment(
            rng=rng,
            n_base=args.n_base,
            depth=args.depth,
            n_distractor_aliases=args.n_distractor_aliases,
            value_range=(args.value_min, args.value_max),
            form=args.form,
        )
        for _ in range(n)
    ]
    if shuffle:
        rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory for train/validation/test JSONL files")
    ap.add_argument("--train", type=int, default=200000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_base", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n_distractor_aliases", type=int, default=6)
    ap.add_argument("--value_min", type=int, default=0)
    ap.add_argument("--value_max", type=int, default=9)
    ap.add_argument("--form", choices=["basic", "math", "code"], default="basic")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_rows = _generate_split(args.train, rng, args, shuffle=True)
    val_rows = _generate_split(args.val, rng, args, shuffle=False)
    test_rows = _generate_split(args.test, rng, args, shuffle=False)

    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_rows)
    _write_jsonl(os.path.join(args.out_dir, "validation.jsonl"), val_rows)
    _write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test_rows)

    print(f"Wrote dataset to {args.out_dir} (train/validation/test jsonl).")
    print("Fields: input, target, task, meta.")


if __name__ == "__main__":
    main()
