from __future__ import annotations

import json
import os
import string
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class S5CharTokenizer(PreTrainedTokenizer):
    """
    Minimal character-level tokenizer that also treats S5 permutation tokens like
    `<A_12345>` as atomic tokens.

    This avoids any dependency on an external subword tokenizer (e.g. SmolLM),
    while keeping the existing S5 token format intact.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        *,
        extra_chars: Optional[str] = None,
        include_s5_permutation_tokens: bool = True,
        **kwargs,
    ):
        # Standard special tokens (store locally because PreTrainedTokenizer.__init__
        # calls get_vocab() before our subclass __init__ finishes).
        pad_token = kwargs.pop("pad_token", "<pad>")
        unk_token = kwargs.pop("unk_token", "<unk>")
        bos_token = kwargs.pop("bos_token", "<bos>")
        eos_token = kwargs.pop("eos_token", "<eos>")

        # Base character set: alphanumeric plus space (used as a lightweight delimiter/placeholder
        # in S5 retrieval to align per-position supervision). Optionally add extras.
        # Include '|' as a compact delimiter between dict and action segments in S5 retrieval.
        charset = string.ascii_letters + string.digits + " " + "|"
        if extra_chars:
            charset += "".join(ch for ch in extra_chars if ch not in charset)

        # Build initial vocab
        tokens: List[str] = [
            pad_token,
            unk_token,
            bos_token,
            eos_token,
        ]
        tokens += list(charset)

        # Add S5 permutation tokens as *single* tokens
        if include_s5_permutation_tokens:
            s5_tokens = [f"<A_{''.join(map(str, p))}>" for p in permutations((1, 2, 3, 4, 5))]
            tokens += s5_tokens

        # De-duplicate while preserving order
        seen = set()
        self._tokens: List[str] = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            self._tokens.append(tok)

        self._token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self._tokens)}
        self._id_to_token: Dict[int, str] = {i: t for t, i in self._token_to_id.items()}

        # Fast lookup for special multi-character tokens
        self._s5_prefix = "<A_"

        # Now that vocab is ready, let HF initialize special-token bookkeeping.
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self._tokens)

    def get_vocab(self) -> Dict[str, int]:  # type: ignore[override]
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> List[str]:  # type: ignore[override]
        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "<" and text.startswith(self._s5_prefix, i):
                j = text.find(">", i)
                if j != -1:
                    candidate = text[i : j + 1]
                    if candidate in self._token_to_id:
                        tokens.append(candidate)
                        i = j + 1
                        continue
            ch = text[i]
            tokens.append(ch if ch in self._token_to_id else self.unk_token)
            i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:  # type: ignore[override]
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:  # type: ignore[override]
        return self._id_to_token.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:  # type: ignore[override]
        return "".join(tokens)

    def build_inputs_with_special_tokens(  # type: ignore[override]
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:  # type: ignore[override]
        os.makedirs(save_directory, exist_ok=True)
        name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._token_to_id, f, ensure_ascii=False, indent=2)
        return (path,)


