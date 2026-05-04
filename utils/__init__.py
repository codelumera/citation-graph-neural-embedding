"""
Utils: Fungsi-fungsi helper untuk Semantic Echo.
"""


def format_score(score: float, decimals: int = 4) -> str:
    """
    Format score untuk ditampilkan.
    
    Args:
        score: Score value
        decimals: Number of decimal places
        
    Returns:
        Formatted score string
    """
    return f"{score:.{decimals}f}"


def validate_doi(doi: str) -> bool:
    """
    Validate DOI format.
    
    Args:
        doi: DOI string to validate
        
    Returns:
        True if valid DOI format
    """
    import re
    pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
    return bool(re.match(pattern, doi, re.IGNORECASE))


def validate_arxiv_id(arxiv_id: str) -> bool:
    """
    Validate arXiv ID format.
    
    Args:
        arxiv_id: arXiv ID string to validate
        
    Returns:
        True if valid arXiv ID format
    """
    import re
    # New format: YYMM.NNNNN
    pattern_new = r'^\d{4}\.\d{4,5}$'
    # Old format: arch-ive/YYMMNNN
    pattern_old = r'^[a-z\-]+/\d{7}$'
    return bool(re.match(pattern_new, arxiv_id) or re.match(pattern_old, arxiv_id))
