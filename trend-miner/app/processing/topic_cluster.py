from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    import hdbscan
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("Warning: scikit-learn or hdbscan not installed. Clustering disabled.")

from app.utils.logger import setup_logger
from app.processing.embedding_generator import EmbeddingGenerator
from app.config import MIN_CLUSTER_SIZE

logger = setup_logger(__name__)

class TopicCluster:
    """Clusters posts into topics"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        ) if CLUSTERING_AVAILABLE else None
        
    def cluster_by_embeddings(self, posts: List[Dict[str, Any]], 
                              method: str = 'hdbscan',
                              min_cluster_size: int = MIN_CLUSTER_SIZE) -> List[Dict[str, Any]]:
        """Cluster posts using embeddings"""
        if not CLUSTERING_AVAILABLE:
            logger.warning("Clustering libraries not available")
            return self._assign_default_clusters(posts)
        
        # Extract texts and generate embeddings
        texts = []
        valid_posts = []
        
        for post in posts:
            text = post.get('cleaned_text') or post.get('text', '')
            if text:
                texts.append(text)
                valid_posts.append(post)
        
        if len(texts) < min_cluster_size:
            logger.warning(f"Not enough posts for clustering: {len(texts)}")
            return self._assign_default_clusters(posts)
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_batch(texts)
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if len(valid_embeddings) < min_cluster_size:
            logger.warning("Not enough valid embeddings for clustering")
            return self._assign_default_clusters(posts)
        
        # Perform clustering
        if method == 'hdbscan':
            labels = self._hdbscan_cluster(valid_embeddings, min_cluster_size)
        elif method == 'kmeans':
            labels = self._kmeans_cluster(valid_embeddings)
        elif method == 'dbscan':
            labels = self._dbscan_cluster(valid_embeddings)
        else:
            labels = self._hdbscan_cluster(valid_embeddings, min_cluster_size)
        
        # Assign cluster IDs to posts
        for i, post in enumerate(valid_posts):
            if i < len(labels):
                post['cluster_id'] = f"cluster_{labels[i]}"
                post['cluster_method'] = method
            else:
                post['cluster_id'] = 'cluster_-1'
                post['cluster_method'] = method
        
        # Add cluster info for invalid posts
        for post in posts:
            if post not in valid_posts:
                post['cluster_id'] = 'cluster_-1'
                post['cluster_method'] = method
        
        return posts
    
    def _hdbscan_cluster(self, embeddings: List[List[float]], 
                         min_cluster_size: int) -> List[int]:
        """Cluster using HDBSCAN"""
        try:
            X = np.array(embeddings)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(X)
            return labels.tolist()
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}")
            return [-1] * len(embeddings)
    
    def _kmeans_cluster(self, embeddings: List[List[float]], 
                        n_clusters: int = 10) -> List[int]:
        """Cluster using K-Means"""
        try:
            X = np.array(embeddings)
            kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42)
            labels = kmeans.fit_predict(X)
            return labels.tolist()
        except Exception as e:
            logger.error(f"K-Means clustering failed: {e}")
            return [-1] * len(embeddings)
    
    def _dbscan_cluster(self, embeddings: List[List[float]], 
                        eps: float = 0.5, min_samples: int = 3) -> List[int]:
        """Cluster using DBSCAN"""
        try:
            X = np.array(embeddings)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            return labels.tolist()
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return [-1] * len(embeddings)
    
    def cluster_by_keywords(self, posts: List[Dict[str, Any]], 
                           n_clusters: int = 10) -> List[Dict[str, Any]]:
        """Cluster posts using TF-IDF and keyword analysis"""
        if not CLUSTERING_AVAILABLE:
            return self._assign_default_clusters(posts)
        
        # Extract texts
        texts = []
        valid_posts = []
        
        for post in posts:
            text = post.get('cleaned_text') or post.get('text', '')
            if text:
                texts.append(text)
                valid_posts.append(post)
        
        if len(texts) < n_clusters:
            return self._assign_default_clusters(posts)
        
        try:
            # Create TF-IDF matrix
            X = self.vectorizer.fit_transform(texts)
            
            # Cluster
            kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Get top keywords per cluster
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_keywords = {}
            
            for i in range(n_clusters):
                if i in labels:
                    # Get centroids for this cluster
                    center = kmeans.cluster_centers_[i]
                    top_indices = center.argsort()[-10:][::-1]
                    keywords = [feature_names[idx] for idx in top_indices if center[idx] > 0]
                    cluster_keywords[f"cluster_{i}"] = keywords
            
            # Assign cluster IDs and keywords
            for i, post in enumerate(valid_posts):
                cluster_id = f"cluster_{labels[i]}"
                post['cluster_id'] = cluster_id
                post['cluster_keywords'] = cluster_keywords.get(cluster_id, [])
                post['cluster_method'] = 'keywords'
            
        except Exception as e:
            logger.error(f"Keyword clustering failed: {e}")
            return self._assign_default_clusters(posts)
        
        # Add default for invalid posts
        for post in posts:
            if post not in valid_posts:
                post['cluster_id'] = 'cluster_-1'
                post['cluster_method'] = 'keywords'
                post['cluster_keywords'] = []
        
        return posts
    
    def _assign_default_clusters(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign default cluster values"""
        for post in posts:
            post['cluster_id'] = 'cluster_-1'
            post['cluster_method'] = 'none'
        return posts
    
    def extract_topic_summary(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract summary information for each topic cluster"""
        # Group posts by cluster
        clusters = defaultdict(list)
        for post in posts:
            cluster_id = post.get('cluster_id', 'cluster_-1')
            clusters[cluster_id].append(post)
        
        summaries = []
        
        for cluster_id, cluster_posts in clusters.items():
            if cluster_id == 'cluster_-1':
                continue  # Skip noise/outliers
            
            # Get all texts
            texts = [p.get('cleaned_text') or p.get('text', '') for p in cluster_posts if p.get('text')]
            
            if not texts:
                continue
            
            # Find common words
            all_words = ' '.join(texts).lower().split()
            word_counts = Counter(all_words)
            common_words = [w for w, c in word_counts.most_common(20) if len(w) > 3]
            
            # Get representative post (longest with good engagement)
            representative = max(cluster_posts, 
                               key=lambda p: (len(p.get('text', '')), p.get('engagement_score', 0)))
            
            # Calculate statistics
            total_engagement = sum(p.get('engagement_score', 0) for p in cluster_posts)
            avg_engagement = total_engagement / len(cluster_posts) if cluster_posts else 0
            
            # Get timestamps
            timestamps = []
            for p in cluster_posts:
                ts = p.get('timestamp')
                if ts:
                    try:
                        timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    except:
                        pass
            
            summaries.append({
                'cluster_id': cluster_id,
                'size': len(cluster_posts),
                'common_words': common_words[:10],
                'representative_text': representative.get('text', '')[:200] + '...',
                'representative_post_id': representative.get('post_id'),
                'total_engagement': total_engagement,
                'avg_engagement': avg_engagement,
                'earliest_post': min(timestamps).isoformat() if timestamps else None,
                'latest_post': max(timestamps).isoformat() if timestamps else None,
                'posts': [p.get('post_id') for p in cluster_posts]
            })
        
        # Sort by size (largest first)
        summaries.sort(key=lambda x: x['size'], reverse=True)
        
        return summaries