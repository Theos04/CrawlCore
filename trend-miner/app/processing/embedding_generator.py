from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embedding generation disabled.")

from app.utils.logger import setup_logger
from app.storage.vector_store import VectorStore
from app.config import EMBEDDING_MODEL, BATCH_SIZE, MODELS_DIR

logger = setup_logger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for text content"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.vector_store = VectorStore()
        
        if EMBEDDINGS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Embedding generation unavailable - install sentence-transformers")
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            model_path = os.path.join(MODELS_DIR, f"embedding_model_{self.model_name.replace('/', '_')}.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded embedding model from {model_path}")
            else:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Save for future use
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Saved embedding model to {model_path}")
                
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None
    
    def generate(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.model or not text:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        if not self.model or not texts:
            return [None] * len(texts)
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def generate_and_store(self, posts: List[Dict[str, Any]]):
        """Generate embeddings for posts and store them"""
        if not self.model:
            logger.warning("Embedding model not available")
            return
        
        # Extract texts for embedding
        texts = []
        post_ids = []
        
        for post in posts:
            # Use cleaned text if available, otherwise raw text
            text = post.get('cleaned_text') or post.get('text', '')
            if text:
                texts.append(text)
                post_ids.append(post.get('post_id'))
        
        if not texts:
            logger.warning("No texts to generate embeddings for")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} posts")
        embeddings = self.generate_batch(texts)
        
        # Store embeddings
        stored_count = 0
        for post_id, embedding in zip(post_ids, embeddings):
            if embedding:
                self.vector_store.store_embedding(post_id, embedding)
                stored_count += 1
        
        logger.info(f"Stored {stored_count} embeddings")
    
    def find_similar(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar posts based on text"""
        embedding = self.generate(text)
        if not embedding:
            return []
        
        return self.vector_store.find_similar(embedding, top_k)
    
    def find_similar_by_id(self, post_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar posts based on a post ID"""
        return self.vector_store.find_similar_by_id(post_id, top_k)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def cluster_embeddings(self, embeddings: List[List[float]], n_clusters: int = 10) -> List[int]:
        """Cluster embeddings into groups"""
        try:
            from sklearn.cluster import KMeans
            
            X = np.array(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            return labels.tolist()
        except Exception as e:
            logger.error(f"Error clustering embeddings: {e}")
            return [0] * len(embeddings)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        return self.vector_store.get_stats()