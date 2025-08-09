#!/usr/bin/env python3
"""
Lightweight, explainable personalization utilities that work without user logs:
- Difficulty scoring for content passages
- Content-based recommendations with diversification
- Retrieval-based quiz question generation from RAG results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math
import re

import numpy as np


@dataclass
class Passage:
    id: str
    context: str
    source_file: str
    similarity_score: float


TECH_TERMS = set([
    # Domain-relevant terms; extend as needed
    "thyroid", "t3", "t4", "estrogen", "progesterone", "cortisol",
    "metabolism", "oxidation", "mitochondria", "glycolysis", "serotonin",
    "dopamine", "gaba", "insulin", "glycogen", "lactate", "co2",
    "aspirin", "vitamin", "saturated", "polyunsaturated", "pufa",
])


def estimate_difficulty_score(text: str) -> float:
    """Estimate difficulty in [0,1] from readability + technical density.

    Heuristics:
    - Longer sentences and higher type-token ratio → harder
    - Higher density of domain terms → harder
    """
    if not text:
        return 0.5

    # Basic sentence tokenization
    sentences = re.split(r"[.!?]+\s+", text)
    sentences = [s for s in sentences if s]
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    num_words = len(words) or 1

    # Sentence length features
    avg_sent_len = num_words / max(1, len(sentences))
    sent_len_penalty = min(1.0, avg_sent_len / 30.0)  # 30 words ~ upper comfortable

    # Lexical richness
    vocab = set(words)
    type_token_ratio = len(vocab) / num_words
    ttr_component = min(1.0, type_token_ratio * 3.0)  # scale

    # Technical term density
    tech_hits = sum(1 for w in words if w in TECH_TERMS)
    tech_density = tech_hits / num_words
    tech_component = min(1.0, tech_density * 10.0)

    # Combine
    difficulty = 0.4 * sent_len_penalty + 0.3 * ttr_component + 0.3 * tech_component
    return float(max(0.0, min(1.0, difficulty)))


def mmr_diversify(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    lambda_mult: float = 0.7,
    top_k: int = 10,
) -> List[int]:
    """Maximal Marginal Relevance for diversification.

    Args:
        query_vec: [D]
        candidate_vecs: [N, D]
    Returns: indices of selected items
    """
    if candidate_vecs.size == 0:
        return []

    # Normalize
    def normalize(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
        return x / n

    q = normalize(query_vec)
    X = normalize(candidate_vecs)

    # Similarities
    rel = X @ q

    selected: List[int] = []
    candidates = list(range(X.shape[0]))
    while len(selected) < min(top_k, X.shape[0]):
        best_idx = None
        best_score = -1e9
        for i in candidates:
            # Diversity penalty vs. already selected
            div = 0.0
            if selected:
                div = max((X[i] @ X[j] for j in selected), default=0.0)
            score = lambda_mult * rel[i] - (1 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)  # type: ignore[arg-type]
        candidates.remove(best_idx)  # type: ignore[arg-type]
    return selected


def generate_mcq_from_passage(
    topic: str,
    passage_text: str,
    num_options: int = 4
) -> Dict[str, Any]:
    """Template-based MCQ from a passage.

    - Simple prompt framing; first option is keyed as correct by design
    - Distractors vary stance phrasing
    """
    stem = (
        f"Based on Ray Peat's views about {topic}, which statement best aligns with the passage?"
    )
    key_phrase = topic.split()[0].capitalize() if topic else "topic"
    options = [
        f"The passage emphasizes the importance of {key_phrase} for metabolic health.",
        f"The passage claims {key_phrase} should generally be avoided.",
        f"The passage suggests {key_phrase} has no meaningful metabolic impact.",
        f"The passage argues {key_phrase} only matters in rare cases.",
    ][: num_options]

    difficulty = estimate_difficulty_score(passage_text)

    return {
        "question_text": stem,
        "options": options,
        "correct_answer": 0,
        "difficulty": difficulty,
        "rationale": passage_text[:240] + ("..." if len(passage_text) > 240 else ""),
    }
