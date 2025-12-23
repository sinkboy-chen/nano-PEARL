from nano_pearl.pearl_engine.sequence import Sequence
from typing import Optional

def ngram_next_token(token_ids: list[int], max_n: int) -> Optional[int]:
    """
    Find the next token based on n-gram matching.

    Args:
        token_ids: List of token IDs representing the current sequence.
        max_n: Maximum n-gram length to consider.

    Returns:
        The next token ID if a match is found, otherwise None.
    """
    for match_len in range(min(max_n, len(token_ids)), 0, -1):
        pattern = token_ids[-match_len:]
        for start_idx in range(len(token_ids) - match_len - 1, -1, -1):
            if token_ids[start_idx: start_idx + match_len] == pattern:
                return token_ids[start_idx + match_len]
    return None


def draft_by_ngram(seq: Sequence, ngram_n: int, max_draft_len: int) -> int:
    """
    Generate draft tokens for a sequence using n-gram repetition.

    Args:
        seq: Sequence object to append tokens to.
        ngram_n: Maximum n-gram length to consider.
        max_draft_len: Maximum number of tokens to draft.
    """
    draft_token_count = 0
    for _ in range(max_draft_len):
        next_token = ngram_next_token(seq.token_ids, ngram_n)
        if next_token is None:
            break
        seq.append_token(next_token)
        draft_token_count += 1
    return draft_token_count
