"""
Embeddings: Representasi teks ilmiah menggunakan SciBERT.

Modul ini menangani ekstraksi fitur teks dari abstrak dan judul paper
menggunakan transformer models yang telah pre-trained pada domain ilmiah.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class SciBERTEmbedding:
    """
    SciBERT embedding extractor untuk representasi teks ilmiah.
    
    Menggunakan model SciBERT (Scientific BERT) yang telah pre-trained
    pada korpus ilmiah untuk menghasilkan embedding yang kaya semantik.
    
    Attributes:
        model_name: Nama model HuggingFace
        tokenizer: Tokenizer untuk model
        model: Transformer model
        device: Device untuk komputasi (cuda/cpu)
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512,
        pooling: str = "cls",
        device: Optional[str] = None
    ):
        """
        Initialize SciBERT embedding extractor.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            pooling: Pooling strategy ('cls', 'mean', 'max')
            device: Device for computation (default: auto-detect)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode teks menjadi embedding vectors.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings with shape (num_texts, hidden_size)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")
            
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get model outputs
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state
            
            # Apply pooling
            if self.pooling == "cls":
                embeddings = last_hidden_state[:, 0, :]
            elif self.pooling == "mean":
                mask = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                embeddings = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            elif self.pooling == "max":
                mask = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                embeddings = torch.max(last_hidden_state * mask, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
    
    def encode_paper(
        self,
        title: str,
        abstract: Optional[str] = None,
        combine: str = "concat"
    ) -> np.ndarray:
        """
        Encode paper berdasarkan title dan abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract (optional)
            combine: How to combine title and abstract embeddings ('concat', 'add', 'title_only')
            
        Returns:
            Embedding vector for the paper
        """
        if combine == "title_only" or abstract is None:
            return self.encode(title)[0]
            
        title_emb = self.encode(title)[0]
        abstract_emb = self.encode(abstract)[0]
        
        if combine == "concat":
            return np.concatenate([title_emb, abstract_emb])
        elif combine == "add":
            return title_emb + abstract_emb
        else:
            raise ValueError(f"Unknown combine strategy: {combine}")
    
    def similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Hitung cosine similarity antara dua set teks.
        
        Args:
            text1: First text or list of texts
            text2: Second text or list of texts
            normalize: Whether to normalize embeddings
            
        Returns:
            Cosine similarity scores
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        if normalize:
            emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-9)
            emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-9)
            
        return np.dot(emb1, emb2.T)
