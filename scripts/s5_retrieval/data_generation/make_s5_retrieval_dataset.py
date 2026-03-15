"""Generate an S5 retrieval dataset: state tracking + in-context key->value retrieval.

We reuse the existing S5 permutation token family `<A_XXXXX>` as *keys*.
Each example has:
  - A dictionary prefix (non-recurrent, control flow 1) containing pairs:
        <A_key1>v1...vk<A_key2>v1...vk...  (pairs are in random order, no delimiters; values have k characters)
  - A delimiter token between the dictionary and the action/retrieval segment (a single '|').
  - Then an action+key sequence (recurrent, control flow 2) where each action is followed
    by explicit key tokens (also `<A_.....>` tokens), repeated k times to create k aligned retrieval slots:
        <A_act1><A_key1>...<A_key1><A_act2><A_key2>...<A_key2>...
    In the default construction here, we set key_t == act_t, so the key token is explicit
    instead of relying on a placeholder space.
  - Target repeats the dictionary prefix, then emits (per action) one token aligned
    to the action position and k characters aligned to the k placeholder key-token positions:
        <A_state1>v1...vk<A_state2>v1...vk...

Semantics after '|':
  - The *input* after '|' is an action+key sequence with k retrieval slots per step:
        <A_act1><A_key1>...<A_key1><A_act2><A_key2>...<A_key2>...
  - The *target* after '|' replaces each action token with the running composed state token
    (composition of all actions up to and including that step), and replaces each key token
    with the dictionary value characters keyed by the *input-side key token* (not the composed state).

Optional "easy" mode (retrieval disabled):
  - The dictionary prefix is still present in input/target, but the key-token positions
    after '|' always emit space characters (' ') instead of a retrieved value.

Semantics:
  - Start from identity permutation (12345).
  - Apply each action permutation to the current state.
  - After each prefix, look up the resulting state in the dictionary and output its value.

We directly store a per-token `control_flow` vector for the *prompt tokens*
under the S5 character tokenizer:
  - dict prefix pairs (key token + value characters) => control_flow = 1
  - action+key tokens => control_flow = 2

Optionally, we can force the dictionary to include the *entire* permutation class
(all 5! = 120 permutations) as keys. Keys not used by the action/state-tracking
sequence then act as distractors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import multiprocessing as mp
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

IDENTITY: Tuple[int, ...] = (1, 2, 3, 4, 5)
TOKEN_TEMPLATE = "<A_{perm}>"


def _render_perm(perm: Sequence[int]) -> str:
    return "".join(str(x) for x in perm)


def _apply_action(state: List[int], action: Sequence[int]) -> List[int]:
    if len(action) != 5:
        raise ValueError(f"Expected action of length 5, got {len(action)}")
    return [state[act - 1] for act in action]


def _sample_actions(num_actions: int, rng: random.Random, pool: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    if not pool:
        raise ValueError("Action pool is empty; cannot sample actions")
    return [rng.choice(pool) for _ in range(num_actions)]


def _rollout_states(actions: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    state = list(IDENTITY)
    states: List[Tuple[int, ...]] = []
    for action in actions:
        state = _apply_action(state, action)
        states.append(tuple(state))
    return states


def _render_token(perm: Sequence[int] | str) -> str:
    perm_str = _render_perm(perm) if not isinstance(perm, str) else perm
    return TOKEN_TEMPLATE.format(perm=perm_str)


def _sample_value_str(*, rng: random.Random, alphabet: str, value_len: int) -> str:
    if value_len <= 0:
        raise ValueError(f"value_len must be positive, got {value_len}")
    if not alphabet:
        raise ValueError("value_alphabet must be non-empty")
    return "".join(rng.choice(alphabet) for _ in range(int(value_len)))


def make_example(
    *,
    num_actions: int,
    rng: random.Random,
    pool: List[Tuple[int, ...]],
    dict_size: Optional[int],
    value_alphabet: str,
    value_len: int = 1,
    include_key_token: bool = False,
    disable_retrieval: bool = False,
    full_dict: bool = False,
) -> Dict[str, object]:
    actions = _sample_actions(num_actions, rng, pool)
    states = _rollout_states(actions)

    # Dictionary keys.
    # - If full_dict=True: include the full permutation class as distractors.
    # - Else: include all actions used (so retrieval is always defined), plus random distractors.
    if full_dict:
        # Use the entire pool (expected size: 120 for S5 permutations).
        dict_keys = list(dict.fromkeys(pool))  # preserve pool order if it contains duplicates
        rng.shuffle(dict_keys)
    else:
        # Keys we must include so every input-side action token is retrievable.
        required_keys = list(dict.fromkeys(actions))  # stable unique input-side action tokens

        target_dict_size = int(dict_size) if dict_size is not None else num_actions
        if target_dict_size < len(required_keys):
            target_dict_size = len(required_keys)

        # Fill dictionary with additional random keys (unique)
        all_keys = set(required_keys)
        while len(all_keys) < target_dict_size:
            all_keys.add(rng.choice(pool))
        dict_keys = list(all_keys)
        rng.shuffle(dict_keys)

    # Assign values (string of length value_len; character-level tokenizer downstream)
    mapping: Dict[Tuple[int, ...], str] = {
        k: _sample_value_str(rng=rng, alphabet=value_alphabet, value_len=value_len) for k in dict_keys
    }

    # Randomize pair order
    dict_pairs = [(k, mapping[k]) for k in dict_keys]
    rng.shuffle(dict_pairs)

    # Serialize dictionary: key token + value string, no delimiters
    dict_prefix = "".join(_render_token(k) + v for k, v in dict_pairs)
    # Delimiter between dict and action segment (keeps parsing unambiguous).
    delimiter = "|"
    # Prompt after '|': per-step tokens depend on include_key_token.
    #
    # - include_key_token=True (explicit key token):
    #     input : <A_act> <A_key> <space x value_len>
    # - include_key_token=False (single A-token per step; key_t == act_t):
    #     input : <A_act> <space x value_len>
    #
    # In all cases, the value_len spaces are retrieval slots replaced by target value characters.
    if include_key_token:
        action_prompt = "".join(_render_token(a) + _render_token(a) + (" " * int(value_len)) for a in actions)
    else:
        action_prompt = "".join(_render_token(a) + (" " * int(value_len)) for a in actions)

    # Target: repeat dictionary prefix, then per-step tokens depend on include_key_token:
    # - include_key_token=False: (1 + value_len) tokens per step: <A_state> + value_len chars
    # - include_key_token=True : (2 + value_len) tokens per step: <A_state> + <A_key> + value_len chars
    if disable_retrieval:
        # Keep supervision shape identical, but make retrieval trivially predictable.
        # (value_len characters per key slot under S5CharTokenizer.)
        retrieval_str = " " * int(value_len)
        if include_key_token:
            target_suffix = "".join(_render_token(s) + _render_token(a) + retrieval_str for a, s in zip(actions, states))
        else:
            target_suffix = "".join(_render_token(s) + retrieval_str for s in states)
        target_pairs = [(_render_perm(a), _render_perm(s), retrieval_str) for a, s in zip(actions, states)]
    else:
        if include_key_token:
            target_suffix = "".join(_render_token(s) + _render_token(a) + mapping[a] for a, s in zip(actions, states))
        else:
            target_suffix = "".join(_render_token(s) + mapping[a] for a, s in zip(actions, states))
        target_pairs = [(_render_perm(a), _render_perm(s), mapping[a]) for a, s in zip(actions, states)]

    # Under S5CharTokenizer:
    # - each dict entry is (1 + value_len) tokens (key token + value_len chars)
    # - each action step has equal tokens on input and target:
    #   - include_key_token=False:
    #       input : <A_act>   + value_len spaces
    #       target: <A_state> + value_len value chars
    #   - include_key_token=True:
    #       input : <A_act>   + <A_key> + value_len spaces
    #       target: <A_state> + <A_key> + value_len value chars
    # Control flow is per *prompt token* (train.py will use it when present).
    # Prompt tokens:
    #   - dict: (key token + value chars) per entry => (1+value_len)*dict_size
    #   - delimiter: 1 ('|')
    #   - action segment per action:
    #       - include_key_token=False: (1+value_len)
    #       - include_key_token=True : (2+value_len)
    dict_prompt_tokens = (1 + int(value_len)) * len(dict_pairs)
    per_step_prompt = (2 + int(value_len)) if include_key_token else (1 + int(value_len))
    action_prompt_tokens = per_step_prompt * int(num_actions)
    prompt_token_count = dict_prompt_tokens + 1 + action_prompt_tokens
    control_flow = ([1] * (dict_prompt_tokens + 1)) + ([2] * action_prompt_tokens)
    assert len(control_flow) == prompt_token_count

    return {
        "input": dict_prefix + delimiter + action_prompt,
        "target": dict_prefix + delimiter + target_suffix,
        "control_flow": control_flow,
        "length": (num_actions, len(dict_pairs), 5),
        "attributes": {
            "num_actions": num_actions,
            "dict_size": len(dict_pairs),
            "value_len": int(value_len),
            "include_key_token": bool(include_key_token),
            "disable_retrieval": bool(disable_retrieval),
            "actions": [_render_perm(a) for a in actions],
            "action_tokens": [_render_token(a) for a in actions],
            "states": [_render_perm(s) for s in states],
            "state_tokens": [_render_token(s) for s in states],
            "dict_pairs": [(_render_perm(k), v) for k, v in dict_pairs],
            # (action, composed_state, value_emitted_in_placeholder_slot)
            "target_pairs": target_pairs,
        },
    }


@dataclass
class ActionRange:
    min_actions: int
    max_actions: int

    def __post_init__(self):
        if self.min_actions <= 0:
            raise ValueError("Minimum actions must be positive")
        if self.min_actions > self.max_actions:
            raise ValueError(f"Invalid action range: {self.min_actions} > {self.max_actions}")


@dataclass
class GenConfig:
    out_dir: str
    train: int
    val: int
    test: int
    seed: int
    dict_size: Optional[int]
    value_alphabet: str
    value_len: int
    include_key_token: bool


def _enumerate_lengths(action_range: ActionRange) -> List[int]:
    return list(range(action_range.min_actions, action_range.max_actions + 1))


def _allocate_uniform_counts(lengths: List[int], total_examples: int) -> Dict[int, int]:
    if total_examples < 0:
        raise ValueError("Number of examples must be non-negative")
    if total_examples == 0:
        return {length: 0 for length in lengths}
    k = len(lengths)
    base = total_examples // k
    rem = total_examples % k
    counts = {length: base for length in lengths}
    for length in lengths[:rem]:
        counts[length] += 1
    return counts


def generate_split_stream(
    n: int,
    action_range: ActionRange,
    rng: random.Random,
    pool: List[Tuple[int, ...]],
    dict_size: Optional[int],
    value_alphabet: str,
    value_len: int = 1,
    include_key_token: bool = False,
    disable_retrieval: bool = False,
    full_dict: bool = False,
    counts_per_length: Optional[Dict[int, int]] = None,
):
    """Yield examples without materializing the full split in memory."""
    if n <= 0:
        return

    lengths = _enumerate_lengths(action_range)
    if not lengths:
        raise ValueError("No feasible action lengths given the config")

    counts = _allocate_uniform_counts(lengths, n) if counts_per_length is None else counts_per_length

    # To avoid giant memory usage for large splits, we iterate lengths in a randomized order
    # but do not attempt a full global shuffle.
    ordered_lengths = list(counts.keys())
    rng.shuffle(ordered_lengths)

    for length in ordered_lengths:
        quota = int(counts[length])
        for _ in range(quota):
            yield make_example(
                num_actions=int(length),
                rng=rng,
                pool=pool,
                dict_size=dict_size,
                value_alphabet=value_alphabet,
                value_len=value_len,
                include_key_token=include_key_token,
                disable_retrieval=disable_retrieval,
                full_dict=full_dict,
            )


def write_jsonl_stream(path: str, rows_iter, *, report_every: int = 0) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows_iter:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if report_every and written % report_every == 0:
                print(f"[write] {os.path.basename(path)}: wrote {written} examples...")
    return written


def _print_split_summary(name: str, counts: Counter, n: int):
    print(f"\n[{name}] examples: {n}")
    if n <= 0:
        return
    if not counts:
        return
    total = sum(counts.values())
    mean = (sum(k * v for k, v in counts.items()) / max(1, total))
    print(f"  actions: min={min(counts)} max={max(counts)} mean={mean:.2f}")
    items = sorted(counts.items())
    print(f"  length distribution (first 20): {', '.join([f'{k}:{v}' for k, v in items[:20]])}")


def _split_counts_across_workers(counts: Dict[int, int], num_proc: int) -> List[Dict[int, int]]:
    """Deterministically split per-length quotas across workers."""
    per_worker = [dict() for _ in range(num_proc)]
    for length, total in sorted(counts.items()):
        base = int(total) // num_proc
        rem = int(total) % num_proc
        for w in range(num_proc):
            per_worker[w][int(length)] = base + (1 if w < rem else 0)
    return per_worker


def _worker_write_shard(
    *,
    split_name: str,
    shard_path: str,
    seed: int,
    worker_id: int,
    counts_per_length: Dict[int, int],
    action_range: ActionRange,
    dict_size: Optional[int],
    value_alphabet: str,
    value_len: int,
    include_key_token: bool,
    disable_retrieval: bool,
    full_dict: bool,
) -> Dict[str, object]:
    rng = random.Random(int(seed) + 1000003 * int(worker_id) + 9176)
    pool = list(permutations((1, 2, 3, 4, 5)))
    local_counts = Counter()

    def _iter_rows():
        for length, quota in counts_per_length.items():
            quota_i = int(quota)
            if quota_i <= 0:
                continue
            for _ in range(quota_i):
                row = make_example(
                    num_actions=int(length),
                    rng=rng,
                    pool=pool,
                    dict_size=dict_size,
                    value_alphabet=value_alphabet,
                    value_len=value_len,
                    include_key_token=include_key_token,
                    disable_retrieval=disable_retrieval,
                    full_dict=full_dict,
                )
                try:
                    local_counts[int(row["length"][0])] += 1
                except Exception:
                    pass
                yield row

    written = write_jsonl_stream(shard_path, _iter_rows(), report_every=0)
    return {
        "split": split_name,
        "worker_id": int(worker_id),
        "written": int(written),
        "counts": dict(local_counts),
    }


def _worker_write_shard_from_dict(params: Dict[str, object]) -> Dict[str, object]:
    """multiprocessing-friendly wrapper."""
    return _worker_write_shard(**params)  # type: ignore[arg-type]


def _write_split_parallel(
    *,
    split_name: str,
    out_path: str,
    total: int,
    action_range: ActionRange,
    base_seed: int,
    dict_size: Optional[int],
    value_alphabet: str,
    value_len: int,
    include_key_token: bool,
    disable_retrieval: bool,
    full_dict: bool,
    num_proc: int,
    keep_shards: bool,
) -> Tuple[int, Counter]:
    if total <= 0:
        # Create empty file for consistency.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8"):
            pass
        return 0, Counter()

    lengths = _enumerate_lengths(action_range)
    counts = _allocate_uniform_counts(lengths, total)
    per_worker_counts = _split_counts_across_workers(counts, num_proc)

    shard_dir = os.path.join(os.path.dirname(out_path), f".shards_{os.path.basename(out_path)}")
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = [os.path.join(shard_dir, f"{split_name}.shard{w:04d}.jsonl") for w in range(num_proc)]

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_proc) as pool:
        results = pool.map(
            _worker_write_shard_from_dict,
            [
                {
                    "split_name": split_name,
                    "shard_path": shard_paths[w],
                    "seed": base_seed,
                    "worker_id": w,
                    "counts_per_length": per_worker_counts[w],
                    "action_range": action_range,
                    "dict_size": dict_size,
                    "value_alphabet": value_alphabet,
                    "value_len": value_len,
                    "include_key_token": include_key_token,
                    "disable_retrieval": disable_retrieval,
                    "full_dict": full_dict,
                }
                for w in range(num_proc)
            ],
        )

    # Merge shards
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged_counts = Counter()
    written_total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for w, shard_path in enumerate(shard_paths):
            if not os.path.exists(shard_path):
                continue
            with open(shard_path, "r", encoding="utf-8") as in_f:
                shutil.copyfileobj(in_f, out_f)

    # Aggregate stats from worker returns
    for res in results:
        written_total += int(res.get("written", 0))
        merged_counts.update({int(k): int(v) for k, v in (res.get("counts") or {}).items()})

    if not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)

    return written_total, merged_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--val", type=int, default=0)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_min_actions", type=int, required=True)
    parser.add_argument("--train_max_actions", type=int, required=True)
    parser.add_argument("--test_min_actions", type=int, required=True)
    parser.add_argument("--test_max_actions", type=int, required=True)

    parser.add_argument(
        "--dict_size",
        type=int,
        default=None,
        help="Dictionary size per example. None => dict_size = num_actions.",
    )
    parser.add_argument(
        "--full_dict",
        action="store_true",
        help="If set, include the full 5!=120 permutation class as dictionary keys (unused keys act as distractors).",
    )
    parser.add_argument(
        "--value_alphabet",
        type=str,
        default=(string.ascii_letters + string.digits),
        help="Alphabet to sample value characters from (default: alphanumeric).",
    )
    parser.add_argument(
        "--value_len",
        type=int,
        default=1,
        help="Number of characters per dictionary value / retrieval output (default: 1).",
    )
    parser.add_argument(
        "--include_key_token",
        action="store_true",
        help="If set, include an explicit key token after each action token in the prompt. "
        "Default: off (single action token per step; key_t == act_t).",
    )
    parser.add_argument(
        "--disable_retrieval",
        action="store_true",
        help="Easier variant: key-token positions after '|' always predict a single space character.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use for generation. When >1, writes shards then merges.",
    )
    parser.add_argument(
        "--keep_shards",
        action="store_true",
        help="Keep intermediate shard files when using --num_proc>1.",
    )

    args = parser.parse_args()

    cfg = GenConfig(
        out_dir=args.out_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        dict_size=args.dict_size,
        value_alphabet=args.value_alphabet,
        value_len=int(args.value_len),
        include_key_token=bool(args.include_key_token),
    )

    train_range = ActionRange(args.train_min_actions, args.train_max_actions)
    test_range = ActionRange(args.test_min_actions, args.test_max_actions)

    train_path = os.path.join(cfg.out_dir, "train.jsonl")
    test_path = os.path.join(cfg.out_dir, "test.jsonl")
    val_path = os.path.join(cfg.out_dir, "validation.jsonl")

    # Stream write to avoid holding millions of examples in memory.
    num_proc = int(args.num_proc or 1)
    keep_shards = bool(args.keep_shards)
    disable_retrieval = bool(args.disable_retrieval)
    full_dict = bool(args.full_dict)
    if num_proc > 1:
        print(f"\n[train] parallel write ({num_proc} processes)...")
        written_train, train_counts = _write_split_parallel(
            split_name="train",
            out_path=train_path,
            total=cfg.train,
            action_range=train_range,
            base_seed=cfg.seed,
            dict_size=cfg.dict_size,
            value_alphabet=cfg.value_alphabet,
            value_len=cfg.value_len,
            include_key_token=cfg.include_key_token,
            disable_retrieval=disable_retrieval,
            full_dict=full_dict,
            num_proc=num_proc,
            keep_shards=keep_shards,
        )
        _print_split_summary("train", train_counts, written_train)

        print("\n[val] parallel write...")
        written_val, val_counts = _write_split_parallel(
            split_name="val",
            out_path=val_path,
            total=cfg.val,
            action_range=train_range,
            base_seed=cfg.seed + 13,
            dict_size=cfg.dict_size,
            value_alphabet=cfg.value_alphabet,
            value_len=cfg.value_len,
            include_key_token=cfg.include_key_token,
            disable_retrieval=disable_retrieval,
            full_dict=full_dict,
            num_proc=num_proc,
            keep_shards=keep_shards,
        )
        if cfg.val:
            _print_split_summary("val", val_counts, written_val)

        print("\n[test] parallel write...")
        written_test, test_counts = _write_split_parallel(
            split_name="test",
            out_path=test_path,
            total=cfg.test,
            action_range=test_range,
            base_seed=cfg.seed + 37,
            dict_size=cfg.dict_size,
            value_alphabet=cfg.value_alphabet,
            value_len=cfg.value_len,
            include_key_token=cfg.include_key_token,
            disable_retrieval=disable_retrieval,
            full_dict=full_dict,
            num_proc=num_proc,
            keep_shards=keep_shards,
        )
        _print_split_summary("test", test_counts, written_test)
    else:
        rng = random.Random(cfg.seed)
        pool = list(permutations((1, 2, 3, 4, 5)))

        print("\n[train] streaming write...")
        train_counts = Counter()
        train_iter = generate_split_stream(
            cfg.train,
            train_range,
            rng,
            pool,
            cfg.dict_size,
            cfg.value_alphabet,
            value_len=cfg.value_len,
            include_key_token=cfg.include_key_token,
            disable_retrieval=disable_retrieval,
            full_dict=full_dict,
        )

        def _train_iter_with_stats():
            for row in train_iter:
                try:
                    train_counts[int(row["length"][0])] += 1
                except Exception:
                    pass
                yield row

        written_train = write_jsonl_stream(train_path, _train_iter_with_stats(), report_every=100000 if cfg.train >= 100000 else 0)
        _print_split_summary("train", train_counts, written_train)

        print("\n[val] streaming write...")
        written_val = 0
        if cfg.val:
            val_counts = Counter()
            val_iter = generate_split_stream(
                cfg.val,
                train_range,
                rng,
                pool,
                cfg.dict_size,
                cfg.value_alphabet,
                value_len=cfg.value_len,
                include_key_token=cfg.include_key_token,
                disable_retrieval=disable_retrieval,
                full_dict=full_dict,
            )

            def _val_iter_with_stats():
                for row in val_iter:
                    try:
                        val_counts[int(row["length"][0])] += 1
                    except Exception:
                        pass
                    yield row

            written_val = write_jsonl_stream(val_path, _val_iter_with_stats(), report_every=100000 if cfg.val >= 100000 else 0)
            _print_split_summary("val", val_counts, written_val)
        else:
            # Create empty file for consistency with the parallel path.
            os.makedirs(os.path.dirname(val_path), exist_ok=True)
            with open(val_path, "w", encoding="utf-8"):
                pass

        print("\n[test] streaming write...")
        test_counts = Counter()
        test_iter = generate_split_stream(
            cfg.test,
            test_range,
            rng,
            pool,
            cfg.dict_size,
            cfg.value_alphabet,
            value_len=cfg.value_len,
            include_key_token=cfg.include_key_token,
            disable_retrieval=disable_retrieval,
            full_dict=full_dict,
        )

        def _test_iter_with_stats():
            for row in test_iter:
                try:
                    test_counts[int(row["length"][0])] += 1
                except Exception:
                    pass
                yield row

        written_test = write_jsonl_stream(test_path, _test_iter_with_stats(), report_every=100000 if cfg.test >= 100000 else 0)
        _print_split_summary("test", test_counts, written_test)

    print(f"\n[OK] Wrote dataset to: {cfg.out_dir}")
    print(f"  - train: {written_train}")
    print(f"  - val:   {written_val}")
    print(f"  - test:  {written_test}")



if __name__ == "__main__":
    main()


