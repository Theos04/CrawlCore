from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from app.storage.mongodb import MongoDBClient
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """Vector storage and similarity search"""
    
    def __init__(self):
        self.db = MongoDBClient()
        
    def store_embedding(self, post_id: str, embedding: List[float]):
        """Store embedding in database"""
        self.db.store_embedding(post_id, embedding)
    
    def get_embedding(self, post_id: str) -> Optional[List[float]]:
        """Get embedding by post ID"""
        return self.db.get_embedding(post_id)
    
    def find_similar(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar vectors using cosine similarity"""
        # Get all embeddings (in production, use specialized vector DB)
        all_embeddings = self.db.get_all_embeddings(limit=1000)
        
        if not all_embeddings:
            return []
        
        # Convert to numpy for computation
        query = np.array(query_vector)
        
        similarities = []
        for emb in all_embeddings:
            vector = np.array(emb['vector'])
            
            # Compute cosine similarity
            dot_product = np.dot(query, vector)
            norm_query = np.linalg.norm(query)
            norm_vector = np.linalg.norm(vector)
            
            if norm_query == 0 or norm_vector == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_query * norm_vector)
            
            # Get post details
            post = self.db.get_processed_post(emb['post_id'])
            
            similarities.append({
                'post_id': emb['post_id'],
                'similarity': float(similarity),
                'post': post
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def find_similar_by_id(self, post_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar posts by post ID"""
        embedding = self.get_embedding(post_id)
        if not embedding:
            logger.warning(f"No embedding found for post {post_id}")
            return []
        
        return self.find_similar(embedding, top_k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        embeddings = self.db.get_all_embeddings(limit=10000)
        
        if not embeddings:
            return {
                'total_embeddings': 0,
                'avg_dimension': 0,
                'oldest': None,
                'newest': None
            }
        
        # Calculate statistics
        dimensions = [len(e['vector']) for e in embeddings if e.get('vector')]
        
        # Get timestamps
        timestamps = [e.get('created_at') for e in embeddings if e.get('created_at')]
        
        return {
            'total_embeddings': len(embeddings),
            'avg_dimension': sum(dimensions) / len(dimensions) if dimensions else 0,
            'min_dimension': min(dimensions) if dimensions else 0,
            'max_dimension': max(dimensions) if dimensions else 0,
            'oldest': min(timestamps) if timestamps else None,
            'newest': max(timestamps) if timestamps else None
        }
    
    def create_faiss_index(self):
        """Create FAISS index for faster similarity search"""
        try:
            import faiss
            
            # Get all embeddings
            embeddings = self.db.get_all_embeddings()
            
            if not embeddings:
                logger.warning("No embeddings to index")
                return None
            
            # Prepare vectors
            vectors = np.array([e['vector'] for e in embeddings]).astype('float32')
            dimension = vectors.shape[1]
            
            # Create index
            index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Add to index
            index.add(vectors)
            
            # Store post IDs mapping
            post_ids = [e['post_id'] for e in embeddings]
            
            logger.info(f"Created FAISS index with {len(vectors)} vectors")
            
            return {
                'index': index,
                'post_ids': post_ids
            }
            
        except ImportError:
            logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
            return None
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return None
    
    def find_similar_faiss(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar vectors using FAISS (faster for large datasets)"""
        try:
            import faiss
            
            # Create index if not exists (in production, persist this)
            index_data = self.create_faiss_index()
            if not index_data:
                return self.find_similar(query_vector, top_k)  # Fallback to brute force
            
            index = index_data['index']
            post_ids = index_data['post_ids']
            
            # Prepare query
            query = np.array([query_vector]).astype('float32')
            faiss.normalize_L2(query)
            
            # Search
            similarities, indices = index.search(query, min(top_k, len(post_ids)))
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(post_ids):
                    post_id = post_ids[idx]
                    post = self.db.get_processed_post(post_id)
                    results.append({
                        'post_id': post_id,
                        'similarity': float(similarities[0][i]),
                        'post': post
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self.find_similar(query_vector, top_k)  # Fallback