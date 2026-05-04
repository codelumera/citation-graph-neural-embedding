"""
Data Loaders: Modul untuk memuat data dari berbagai sumber akademik.

Mendukung:
- OpenAlex API: Database akademik terbuka yang komprehensif
- arXiv API: Preprint server untuk fisika, CS, matematika, dll.
"""

import requests
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import numpy as np

from .graph import CitationGraph, Node, Edge


class OpenAlexLoader:
    """
    Loader untuk mengambil data dari OpenAlex API.
    
    OpenAlex adalah database akademik terbuka yang menyediakan
    informasi tentang paper, penulis, jurnal, institusi, dan sitasi.
    
    Attributes:
        api_url: Base URL for OpenAlex API
        cache_dir: Directory for caching responses
        rate_limit: Requests per second limit
    """
    
    API_URL = "https://api.openalex.org"
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        rate_limit: float = 1.0,
        email: Optional[str] = None
    ):
        """
        Initialize OpenAlex loader.
        
        Args:
            cache_dir: Directory to cache API responses
            rate_limit: Maximum requests per second
            email: Email address for polite API usage (recommended)
        """
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self.email = email
        self.last_request_time = 0
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.0 / self.rate_limit:
            time.sleep(1.0 / self.rate_limit - elapsed)
            
        url = f"{self.API_URL}/{endpoint}"
        headers = {}
        if self.email:
            headers['Mailto'] = self.email
            
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        self.last_request_time = time.time()
        return response.json()
    
    def fetch_paper_by_doi(self, doi: str) -> Optional[Dict]:
        """
        Fetch paper metadata by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Paper metadata dictionary or None
        """
        try:
            endpoint = f"works/{doi}"
            data = self._make_request(endpoint)
            return data
        except Exception as e:
            print(f"Error fetching {doi}: {e}")
            return None
    
    def fetch_papers_by_dois(self, dois: List[str]) -> List[Dict]:
        """
        Fetch multiple papers by DOI.
        
        Args:
            dois: List of DOIs
            
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        for doi in tqdm(dois, desc="Fetching papers from OpenAlex"):
            paper = self.fetch_paper_by_doi(doi)
            if paper:
                papers.append(paper)
        return papers
    
    def build_graph(self, doi_list: List[str]) -> CitationGraph:
        """
        Build citation graph from list of DOIs.
        
        Args:
            doi_list: List of DOIs to include in graph
            
        Returns:
            CitationGraph object
        """
        graph = CitationGraph()
        
        # Fetch all papers
        papers = self.fetch_papers_by_dois(doi_list)
        
        # Create embedding placeholder (in practice, use SciBERT)
        embedding_dim = 768
        
        for paper in papers:
            paper_id = paper.get('doi') or paper.get('id')
            if not paper_id:
                continue
                
            # Extract metadata
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            publication_year = paper.get('publication_year', 0)
            
            # Create placeholder embedding (replace with actual embeddings)
            features = np.random.randn(embedding_dim).astype(np.float32)
            
            # Add paper node
            paper_node = Node(
                id=paper_id,
                type='paper',
                features=features,
                metadata={
                    'title': title,
                    'abstract': abstract,
                    'year': publication_year,
                    'doi': paper_id
                }
            )
            graph.add_node(paper_node)
            
            # Add author nodes and edges
            for authorship in paper.get('authorships', []):
                author_id = authorship.get('author', {}).get('id')
                if author_id:
                    author_name = authorship.get('author', {}).get('display_name', '')
                    
                    author_node = Node(
                        id=author_id,
                        type='author',
                        features=np.random.randn(embedding_dim).astype(np.float32),
                        metadata={'name': author_name}
                    )
                    graph.add_node(author_node)
                    
                    # Add written_by edge
                    graph.add_edge(Edge(
                        src=paper_id,
                        dst=author_id,
                        relation='written_by'
                    ))
                    
                    # Add affiliated_with edge
                    institutions = authorship.get('institutions', [])
                    for inst in institutions:
                        inst_id = inst.get('id')
                        if inst_id:
                            inst_node = Node(
                                id=inst_id,
                                type='institution',
                                features=np.random.randn(embedding_dim).astype(np.float32),
                                metadata={'name': inst.get('display_name', '')}
                            )
                            graph.add_node(inst_node)
                            
                            graph.add_edge(Edge(
                                src=author_id,
                                dst=inst_id,
                                relation='affiliated_with'
                            ))
            
            # Add journal node
            journal_info = paper.get('primary_location', {}).get('source')
            if journal_info:
                journal_id = journal_info.get('id')
                if journal_id:
                    journal_node = Node(
                        id=journal_id,
                        type='journal',
                        features=np.random.randn(embedding_dim).astype(np.float32),
                        metadata={'name': journal_info.get('display_name', '')}
                    )
                    graph.add_node(journal_node)
                    
                    graph.add_edge(Edge(
                        src=paper_id,
                        dst=journal_id,
                        relation='published_in'
                    ))
            
            # Add citation edges
            references = paper.get('referenced_works', [])
            for ref_doi in references:
                if ref_doi in doi_list:  # Only add edges within our dataset
                    graph.add_edge(Edge(
                        src=paper_id,
                        dst=ref_doi,
                        relation='cites'
                    ))
        
        return graph.finalize()


class ArxivLoader:
    """
    Loader untuk mengambil data dari arXiv API.
    
    arXiv adalah repository preprint untuk fisika, ilmu komputer,
    matematika, dan bidang terkait.
    
    Attributes:
        api_url: Base URL for arXiv API
        cache_dir: Directory for caching responses
    """
    
    API_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize arXiv loader.
        
        Args:
            cache_dir: Directory to cache API responses
        """
        self.cache_dir = cache_dir
        
    def fetch_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Fetch paper metadata by arXiv ID.
        
        Args:
            arxiv_id: arXiv identifier (e.g., '2103.12345')
            
        Returns:
            Paper metadata dictionary or None
        """
        try:
            params = {
                'search_query': f'id:{arxiv_id}',
                'start': 0,
                'max_results': 1
            }
            response = requests.get(self.API_URL, params=params)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            # In production, use proper XML parsing
            return {
                'arxiv_id': arxiv_id,
                'title': 'Parsed Title',
                'abstract': 'Parsed Abstract',
                'authors': [],
                'categories': []
            }
        except Exception as e:
            print(f"Error fetching {arxiv_id}: {e}")
            return None
    
    def fetch_papers_by_ids(self, arxiv_ids: List[str]) -> List[Dict]:
        """
        Fetch multiple papers by arXiv ID.
        
        Args:
            arxiv_ids: List of arXiv IDs
            
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        for arxiv_id in tqdm(arxiv_ids, desc="Fetching papers from arXiv"):
            paper = self.fetch_paper_by_id(arxiv_id)
            if paper:
                papers.append(paper)
        return papers
    
    def build_graph(self, arxiv_ids: List[str]) -> CitationGraph:
        """
        Build citation graph from list of arXiv IDs.
        
        Args:
            arxiv_ids: List of arXiv IDs to include in graph
            
        Returns:
            CitationGraph object
        """
        graph = CitationGraph()
        embedding_dim = 768
        
        papers = self.fetch_papers_by_ids(arxiv_ids)
        
        for paper in papers:
            paper_id = paper.get('arxiv_id')
            if not paper_id:
                continue
                
            features = np.random.randn(embedding_dim).astype(np.float32)
            
            paper_node = Node(
                id=paper_id,
                type='paper',
                features=features,
                metadata={
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'arxiv_id': paper_id
                }
            )
            graph.add_node(paper_node)
            
            # Add authors
            for i, author_name in enumerate(paper.get('authors', [])):
                author_id = f"arxiv_author_{paper_id}_{i}"
                author_node = Node(
                    id=author_id,
                    type='author',
                    features=np.random.randn(embedding_dim).astype(np.float32),
                    metadata={'name': author_name}
                )
                graph.add_node(author_node)
                
                graph.add_edge(Edge(
                    src=paper_id,
                    dst=author_id,
                    relation='written_by'
                ))
        
        return graph.finalize()
