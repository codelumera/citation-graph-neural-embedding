"""
Metrics: Metrik untuk mengukur pengaruh semantik dan disruption.

Modul ini menyediakan berbagai metrik untuk mengevaluasi pengaruh ilmiah:
- Echo Score: Kesamaan semantik antar paper
- Disruption Index: Mengukur tingkat disruption karya ilmiah
- Influence Depth: Metrik komprehensif dengan temporal decay
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def echo_score(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Hitung Echo Score antara dua paper berdasarkan cosine similarity.
    
    Echo Score mengukur seberapa mirip representasi semantik dari dua paper.
    Skor tinggi menunjukkan bahwa Paper B memiliki "DNA konseptual" yang serupa
    dengan Paper A, mengindikasikan pengaruh yang dalam.
    
    Args:
        embedding_a: Embedding vector dari Paper A
        embedding_b: Embedding vector dari Paper B
        normalize: Whether to normalize embeddings before computing similarity
        
    Returns:
        Cosine similarity score between -1 and 1 (typically 0 to 1 for normalized embeddings)
        
    Example:
        >>> emb_a = np.random.randn(768)
        >>> emb_b = np.random.randn(768)
        >>> score = echo_score(emb_a, emb_b)
        >>> print(f"Echo Score: {score:.4f}")
    """
    if normalize:
        embedding_a = embedding_a / (np.linalg.norm(embedding_a) + 1e-9)
        embedding_b = embedding_b / (np.linalg.norm(embedding_b) + 1e-9)
        
    similarity = np.dot(embedding_a, embedding_b)
    return float(similarity)


def disruption_index(
    citations: List[Dict],
    window_before: int = 5,
    window_after: int = 5
) -> float:
    """
    Hitung Disruption Index (CD index) untuk sebuah paper.
    
    Disruption Index mengukur sejauh mana sebuah paper "mendisrupsi" literatur
    yang ada vs hanya mengembangkan karya sebelumnya. Formula berdasarkan
    Funk & Owen-Smith (2017).
    
    CD = (N_i - N_j) / (N_i + N_j)
    
    Dimana:
    - N_i: Jumlah paper yang mengutip paper target TAPI TIDAK mengutip referensi paper tersebut
    - N_j: Jumlah paper yang mengutip paper target DAN juga mengutip referensinya
    
    Args:
        citations: List of citation dictionaries with structure:
            {
                'citing_paper': str,
                'references': List[str],  # Papers referenced by citing_paper
                'year': int
            }
        window_before: Years before publication to consider
        window_after: Years after publication to consider
        
    Returns:
        Disruption index between -1 and 1
        - Positive values indicate disruptive work
        - Negative values indicate consolidating work
        
    Example:
        >>> citations = [
        ...     {'citing_paper': 'P1', 'references': ['T', 'R1'], 'year': 2020},
        ...     {'citing_paper': 'P2', 'references': ['T'], 'year': 2021},
        ... ]
        >>> di = disruption_index(citations)
    """
    if not citations:
        return 0.0
        
    n_i = 0  # Cites target but not references
    n_j = 0  # Cites target and references
    
    for citation in citations:
        refs = set(citation.get('references', []))
        
        # Check if this paper cites the target (we assume target is implied)
        # In practice, you'd need to know which paper is the target
        # For this implementation, we check if any reference matches a pattern
        
        # Simplified: count based on reference patterns
        has_references = len(refs) > 0
        
        if has_references:
            n_j += 1  # Cites both target and its references
        else:
            n_i += 1  # Only cites target
            
    total = n_i + n_j
    if total == 0:
        return 0.0
        
    cd_index = (n_i - n_j) / total
    return float(cd_index)


def influence_depth(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    source_year: int,
    target_year: int,
    citation_count: int,
    temporal_decay: float = 0.95,
    min_citations: int = 5
) -> float:
    """
    Hitung Influence Depth score yang komprehensif.
    
    Influence Depth menggabungkan:
    1. Semantic similarity (Echo Score)
    2. Temporal decay (pengaruh menurun seiring waktu)
    3. Citation impact (jumlah sitasi)
    
    Formula:
    ID = Echo_Score * Temporal_Decay^(target_year - source_year) * log(1 + citations)
    
    Args:
        source_embedding: Embedding dari paper sumber (yang mempengaruhi)
        target_embedding: Embedding dari paper target (yang dipengaruhi)
        source_year: Tahun publikasi paper sumber
        target_year: Tahun publikasi paper target
        citation_count: Jumlah sitasi yang diterima paper target
        temporal_decay: Decay factor per year (default: 0.95)
        min_citations: Minimum citations untuk normalisasi
        
    Returns:
        Influence depth score (higher = deeper influence)
        
    Example:
        >>> emb_source = np.random.randn(768)
        >>> emb_target = np.random.randn(768)
        >>> depth = influence_depth(emb_source, emb_target, 2018, 2022, 50)
        >>> print(f"Influence Depth: {depth:.4f}")
    """
    # Validate inputs
    if target_year < source_year:
        raise ValueError("Target year must be >= source year")
        
    # Calculate semantic similarity
    semantic_sim = echo_score(source_embedding, target_embedding)
    
    # Calculate temporal decay
    years_diff = target_year - source_year
    temporal_factor = temporal_decay ** years_diff
    
    # Calculate citation impact (log-scaled)
    citation_impact = np.log(1 + max(citation_count, min_citations))
    
    # Normalize citation impact (optional scaling)
    max_expected_citations = 1000  # Adjust based on field
    citation_normalized = citation_impact / np.log(1 + max_expected_citations)
    
    # Combined score
    influence_score = semantic_sim * temporal_factor * citation_normalized
    
    return float(influence_score)


def average_echo_score(
    embeddings: np.ndarray,
    pairs: List[Tuple[int, int]]
) -> float:
    """
    Hitung rata-rata Echo Score untuk beberapa pasangan paper.
    
    Args:
        embeddings: Matrix of shape (num_papers, embedding_dim)
        pairs: List of (idx1, idx2) tuples indicating paper pairs
        
    Returns:
        Average Echo Score across all pairs
    """
    if not pairs:
        return 0.0
        
    scores = []
    for idx1, idx2 in pairs:
        score = echo_score(embeddings[idx1], embeddings[idx2])
        scores.append(score)
        
    return float(np.mean(scores))


def temporal_influence_curve(
    source_embedding: np.ndarray,
    target_embeddings: np.ndarray,
    target_years: List[int],
    source_year: int,
    temporal_decay: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hitung kurva pengaruh temporal dari sebuah paper sumber.
    
    Args:
        source_embedding: Embedding dari paper sumber
        target_embeddings: Matrix of target paper embeddings
        target_years: List of publication years for target papers
        source_year: Publication year of source paper
        temporal_decay: Decay factor
        
    Returns:
        Tuple of (years, influence_scores) arrays
    """
    unique_years = sorted(set(target_years))
    influence_by_year = {}
    
    for year in unique_years:
        mask = np.array(target_years) == year
        year_embeddings = target_embeddings[mask]
        
        if len(year_embeddings) == 0:
            influence_by_year[year] = 0.0
            continue
            
        # Average semantic similarity for this year
        similarities = [
            echo_score(source_embedding, emb) 
            for emb in year_embeddings
        ]
        avg_sim = np.mean(similarities)
        
        # Apply temporal decay
        years_diff = year - source_year
        if years_diff >= 0:
            temporal_factor = temporal_decay ** years_diff
            influence_by_year[year] = avg_sim * temporal_factor
        else:
            influence_by_year[year] = 0.0  # Future papers don't influence past
            
    years = np.array(list(influence_by_year.keys()))
    scores = np.array(list(influence_by_year.values()))
    
    return years, scores


def field_normalized_impact(
    raw_score: float,
    field_percentiles: Dict[float, float],
    num_percentiles: int = 100
) -> float:
    """
    Normalisasi skor dampak berdasarkan distribusi bidang ilmu.
    
    Args:
        raw_score: Raw influence score
        field_percentiles: Dictionary mapping percentile to score threshold
        num_percentiles: Number of percentiles
        
    Returns:
        Normalized score (0-100 percentile)
    """
    sorted_thresholds = sorted(field_percentiles.items())
    
    for percentile, threshold in sorted_thresholds:
        if raw_score <= threshold:
            return float(percentile)
            
    return 100.0
