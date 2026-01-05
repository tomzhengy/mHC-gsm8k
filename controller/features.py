"""
Feature extraction for RL controller.

Minimal features: prompt_length and question_length (2 features total).

Design decisions:
- sequence_len should come from model.config.sequence_len (default 1024)
- remaining_budget removed (constant 1.0 in one-step env = no signal)
- question_length clamped to 1000 to avoid outlier skew
"""

import numpy as np


def extract_features(info: dict, sequence_len: int = 1024) -> np.ndarray:
    """
    Extract minimal features from environment info dict.

    Args:
        info: Info dict from env.reset() containing:
            - prompt_length: number of tokens in the prompt
            - question (or question_length): the question text or its char count
        sequence_len: Model context window size for normalization
            (should come from model.config.sequence_len)

    Returns:
        2-element float32 array with values in [0, 1]:
            - prompt_length / sequence_len
            - min(question_length, 1000) / 1000
    """
    # prompt_length: normalized by model's context window
    prompt_length = info.get("prompt_length", 0)

    # question_length: character count, clamped to 1000
    if "question_length" in info:
        question_length = info["question_length"]
    elif "question" in info:
        question_length = len(info["question"])
    else:
        question_length = 0

    return np.array([
        prompt_length / float(sequence_len),
        min(question_length, 1000) / 1000.0,
    ], dtype=np.float32)
