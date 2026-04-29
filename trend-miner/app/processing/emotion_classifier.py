from typing import Dict, Any, List, Tuple
import re
from collections import defaultdict


class EmotionClassifier:
    """Classifies emotional content in text"""
    
    def __init__(self):
        # Basic emotion lexicons (simplified - in production use NLTK or trained model)
        self.emotion_keywords = {
            'joy': [
                'happy', 'love', 'great', 'amazing', 'excellent', 'wonderful',
                'fantastic', 'awesome', 'glad', 'delighted', 'pleased', 'joy',
                'excited', 'thrilled', 'blessed', 'grateful', '😊', '😄', '😍', '❤️'
            ],
            'sadness': [
                'sad', 'sorry', 'unfortunate', 'unhappy', 'disappointed',
                'heartbroken', 'depressed', 'grief', 'mourn', 'regret',
                'miss', 'lonely', '😢', '😭', '💔'
            ],
            'anger': [
                'angry', 'mad', 'furious', 'outrage', 'hate', 'terrible',
                'awful', 'horrible', 'annoyed', 'frustrated', '😠', '👿', '🤬'
            ],
            'fear': [
                'scared', 'afraid', 'terrified', 'fear', 'anxious', 'worried',
                'nervous', 'panic', 'dread', '😨', '😱', '😰'
            ],
            'surprise': [
                'wow', 'surprised', 'shocked', 'amazed', 'astonished',
                'unbelievable', 'incredible', '😮', '😲', '🤯'
            ],
            'inspiration': [
                'inspire', 'motivation', 'dream', 'achieve', 'success',
                'goal', 'believe', 'hope', 'future', 'together', '✨', '🌟', '💪'
            ],
            'curiosity': [
                'wonder', 'curious', 'interesting', 'fascinating', 'question',
                'why', 'how', 'what if', '🤔', '🧐'
            ],
            'gratitude': [
                'thank', 'thanks', 'grateful', 'appreciate', 'blessed',
                'thankful', '🙏', '💝'
            ]
        }
        
        # Compile regex patterns for each emotion
        self.emotion_patterns = {}
        for emotion, keywords in self.emotion_keywords.items():
            pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            self.emotion_patterns[emotion] = re.compile(pattern, re.IGNORECASE)
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify emotions in text"""
        if not text:
            return {
                'primary_emotion': 'neutral',
                'emotion_scores': {'neutral': 1.0},
                'emotion_score': 0.0,
                'emotion_confidence': 0.0
            }
        
        # Count emotion keyword occurrences
        emotion_counts = defaultdict(int)
        total_matches = 0
        
        for emotion, pattern in self.emotion_patterns.items():
            matches = pattern.findall(text)
            count = len(matches)
            if count > 0:
                emotion_counts[emotion] = count
                total_matches += count
        
        if total_matches == 0:
            return {
                'primary_emotion': 'neutral',
                'emotion_scores': {'neutral': 1.0},
                'emotion_score': 0.0,
                'emotion_confidence': 0.0
            }
        
        # Calculate scores
        emotion_scores = {}
        for emotion, count in emotion_counts.items():
            emotion_scores[emotion] = count / total_matches
        
        # Determine primary emotion
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Calculate overall emotion score (0-1)
        emotion_score = total_matches / len(text.split()) if text.split() else 0
        emotion_score = min(emotion_score, 1.0)
        
        # Calculate confidence (based on distribution)
        if len(emotion_scores) == 1:
            confidence = 1.0
        else:
            # Entropy-based confidence
            import math
            entropy = -sum(p * math.log(p) for p in emotion_scores.values())
            max_entropy = math.log(len(emotion_scores))
            confidence = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
        
        return {
            'primary_emotion': primary_emotion[0],
            'emotion_scores': dict(emotion_scores),
            'emotion_score': round(emotion_score, 3),
            'emotion_confidence': round(confidence, 3)
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify emotions for multiple texts"""
        return [self.classify(text) for text in texts]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis (positive/negative/neutral)"""
        if not text:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        # Simple lexicon-based sentiment
        positive_words = set([
            'good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful',
            'fantastic', 'amazing', 'positive', 'nice', 'best', 'perfect', 'glad'
        ])
        
        negative_words = set([
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor',
            'negative', 'sad', 'angry', 'upset', 'disappointing', 'wrong'
        ])
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        score = (pos_count - neg_count) / total
        
        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'positive_count': pos_count,
            'negative_count': neg_count
        }
    
    def extract_emotion_features(self, text: str) -> Dict[str, Any]:
        """Extract detailed emotion features for ML"""
        emotion_result = self.classify(text)
        sentiment_result = self.analyze_sentiment(text)
        
        # Extract emoji-based emotions
        import emoji
        emojis = [c for c in text if c in emoji.EMOJI_DATA]
        
        # Count exclamation marks and question marks (indicators of emotion intensity)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Check for all caps (shouting)
        words = text.split()
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        return {
            'primary_emotion': emotion_result['primary_emotion'],
            'emotion_scores': emotion_result['emotion_scores'],
            'sentiment': sentiment_result['sentiment'],
            'sentiment_score': sentiment_result['score'],
            'emotion_intensity': emotion_result['emotion_score'],
            'emotion_confidence': emotion_result['emotion_confidence'],
            'emoji_count': len(emojis),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'all_caps_words': all_caps_words,
            'has_emotion': emotion_result['primary_emotion'] != 'neutral'
        }
    
    def get_emotion_stats(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get emotion statistics across multiple posts"""
        emotions = []
        sentiments = []
        
        for post in posts:
            text = post.get('text', '')
            emotion = self.classify(text)
            sentiment = self.analyze_sentiment(text)
            
            emotions.append(emotion['primary_emotion'])
            sentiments.append(sentiment['sentiment'])
        
        from collections import Counter
        
        return {
            'emotion_distribution': dict(Counter(emotions)),
            'sentiment_distribution': dict(Counter(sentiments)),
            'total_posts': len(posts),
            'primary_emotion': Counter(emotions).most_common(1)[0][0] if emotions else 'neutral'
        }


from app.utils.logger import setup_logger
logger = setup_logger(__name__)