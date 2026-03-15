import torch
import os
from typing import List, Any, Optional
import logging

LOGGING_FORMAT = (
    "%(asctime)s | %(levelname)-6s | [%(name)s] | "
    "%(funcName)s:%(lineno)d - %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S' 
)

logger = logging.getLogger(__name__)

def split_batch_by_recurrent_flow(
        total_recurrence,
        input_embeds,
        control_flows,
        attention_mask
    ):
    """
    Split the input embeddings into chunks for parallel processing
    """
    assert len(total_recurrence) == input_embeds.shape[1], "The total recurrence and input embeddings must have the same length, {} != {}".format(len(total_recurrence), input_embeds.shape[1])

    mats = [input_embeds, control_flows, attention_mask]
    processing_chunks = []
    chunk_start = 0

    for i, ctrl in enumerate(total_recurrence):
        if ctrl:
            if chunk_start != i:
                processing_chunks.append([x[:, chunk_start:i] for x in mats])
            # add the budget to the processing chunk
            processing_chunks.append([x[:, i:i+1] for x in mats])
            # update the chunk start
            chunk_start = i + 1
    
    if chunk_start < len(control_flows):
        processing_chunks.append([x[:, chunk_start:] for x in mats])
    return processing_chunks

def visualize_sample_control_flow(
    tokenizer: Any,
    dataset: Any,
    *,
    idx: int = 0,
    max_tokens: int = 32768,
    input_ids_key: str = "input_ids",
    control_flow_key: str = "control_flow",
    only_rank0: bool = True,
) -> None:
    if only_rank0:
        try:
            if int(os.environ.get("RANK", "0")) != 0:
                return
        except Exception:
            pass

    if dataset is None:
        print("[control_flow] dataset=None", flush=True)
        return

    try:
        n = len(dataset)
    except Exception:
        n = None

    if n is not None and (idx < 0 or idx >= n):
        raise IndexError(f"idx={idx} out of range for dataset of size {n}")

    sample = dataset[idx] if n is not None else dataset
    input_ids = sample.get(input_ids_key)
    control_flow = sample.get(control_flow_key)

    if input_ids is None or control_flow is None:
        keys = list(sample.keys()) if hasattr(sample, "keys") else None
        print(f"[control_flow] missing '{input_ids_key}' or '{control_flow_key}'. keys={keys}", flush=True)
        return

    input_ids = list(input_ids)[:max_tokens]
    control_flow = list(control_flow)[: len(input_ids)]

    red = "\033[31m"
    reset = "\033[0m"

    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    def _token_to_text(tok: str) -> str:
        try:
            return tokenizer.convert_tokens_to_string([tok])
        except Exception:
            pass
        if tok.startswith("Ġ"):
            return " " + tok[1:]
        if tok.startswith("▁"):
            return " " + tok[1:]
        if tok == "Ċ":
            return "\n"
        return tok

    colored_text_parts: List[str] = []
    for tok, cf in zip(tokens, control_flow):
        piece = _token_to_text(str(tok))
        if int(cf) > 1:
            colored_text_parts.append(f"{red}{piece}{reset}")
        else:
            colored_text_parts.append(piece)

    logger.info("[control_flow] decoded (cf>1 in red):")
    logger.info("".join(colored_text_parts))