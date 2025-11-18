#!/usr/bin/env python3
"""
Taglish Spam Detection - Enhanced Flask Web UI
==============================================
A web interface to compare three different spam detection models with:
- Gibberish detection
- Sample message library
- Improved security and error handling
- Better configuration management

Author: Claude
Date: October 2025
"""

from flask import Flask, render_template_string, request, jsonify
import joblib
import os
import re
import numpy as np
import json
from typing import Dict, Tuple, Any, List
import warnings
import logging
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict
import string
import h5py
import pickle

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    """Application configuration"""
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    MAX_MESSAGE_LENGTH = 5000
    MAX_LSTM_LENGTH = 100
    MAX_TRANSFORMER_LENGTH = 512
    RATE_LIMIT_REQUESTS = 30  # requests per window
    RATE_LIMIT_WINDOW = 60  # seconds
    # Default decision thresholds if none are provided by evaluate_models.py
    DEFAULT_THRESHOLDS = {
        'logistic_regression': 0.5,
        'lstm': 0.5,
        'xlm_roberta': 0.5,
    }

# Sample messages library
SAMPLE_MESSAGES = {
    'spam': [
        "CONGRATULATIONS! You've won $1,000,000! Click here now to claim your prize!",
        "FREE IPHONE 15 PRO MAX! Limited offer! Text CLAIM to 12345 now!",
        "Kumita ng P50,000 daily! Work from home! PM me now for details!",
        "URGENT: Your bank account has been suspended. Click here to verify immediately!",
        "Libreng load! Txt your number to 09123456789 to get FREE 100 pesos load!",
        "Make money fast! Invest now and earn 500% returns in 30 days! Limited slots!",
        "YOU ARE A WINNER! Claim your brand new car today! Call 123-4567 now!",
        "Get rich quick! Join our pyramid- I mean network marketing business today!",
        "CONGRATZ KA NANALO! Mag claim ng 100K sa www.scamsite.com",
        "Discount 90% OFF luxury watches! Rolex PHP 500 only! Order now before stocks run out!"
    ],
    'ham': [
        "Hi! Are you free this weekend? Let's grab coffee and catch up!",
        "Good morning! Just reminding you about our meeting at 2 PM today.",
        "Kumusta na? Long time no see! How's the family?",
        "Thanks for your help yesterday. I really appreciate it!",
        "See you later! Don't forget to bring the documents we discussed.",
        "Happy birthday! Wishing you all the best on your special day!",
        "Can you send me the report when you have time? No rush.",
        "Congrats on your promotion! You deserve it. Let's celebrate!",
        "Papunta na ako. Traffic lang dito sa EDSA. Sorry for the delay.",
        "Thank you for your purchase! Your order will arrive in 3-5 business days."
    ]
}

# Rate limiting
rate_limit_storage = defaultdict(list)

def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        now = datetime.now()
        
        # Clean old requests
        rate_limit_storage[ip] = [
            req_time for req_time in rate_limit_storage[ip]
            if now - req_time < timedelta(seconds=Config.RATE_LIMIT_WINDOW)
        ]
        
        # Check rate limit
        if len(rate_limit_storage[ip]) >= Config.RATE_LIMIT_REQUESTS:
            return jsonify({
                'error': f'Rate limit exceeded. Maximum {Config.RATE_LIMIT_REQUESTS} requests per {Config.RATE_LIMIT_WINDOW} seconds.'
            }), 429
        
        # Add current request
        rate_limit_storage[ip].append(now)
        
        return f(*args, **kwargs)
    return decorated_function


class GibberishDetector:
    """Detects if input text is gibberish/nonsensical"""
    
    def __init__(self):
        # Common words in English and Filipino
        self.common_words = {
            # English
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'free', 'win', 'claim', 'click', 'now', 'call', 'text', 'urgent', 'congratulations',
            # Filipino/Taglish
            'ang', 'ng', 'sa', 'na', 'ay', 'mga', 'ko', 'mo', 'ka', 'po',
            'ako', 'ikaw', 'siya', 'kami', 'tayo', 'kayo', 'sila', 'yan', 'yun', 'ito',
            'kung', 'para', 'dahil', 'kasi', 'pero', 'at', 'o', 'din', 'rin', 'lang',
            'ba', 'naman', 'kaya', 'nga', 'talaga', 'pa', 'na', 'kumusta', 'salamat',
            'pls', 'plz', 'txt', 'msg', 'pm', 'asap', 'btw', 'omg', 'lol'
        }
        
        # Common bigrams (word pairs)
        self.common_bigrams = {
            ('you', 'are'), ('i', 'am'), ('thank', 'you'), ('for', 'free'),
            ('click', 'here'), ('call', 'now'), ('text', 'to'), ('limited', 'offer'),
            ('ang', 'mga'), ('sa', 'akin'), ('para', 'sa'), ('kung', 'gusto')
        }
    
    def calculate_vowel_ratio(self, text: str) -> float:
        """Calculate ratio of vowels in text"""
        if not text:
            return 0.0
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for c in text if c in vowels)
        letter_count = sum(1 for c in text if c.isalpha())
        return vowel_count / letter_count if letter_count > 0 else 0.0
    
    def calculate_word_score(self, text: str) -> float:
        """Calculate percentage of recognized words"""
        # Split by word boundaries and also by common punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        # Also try splitting by punctuation if no words found
        if not words:
            # Split by common delimiters
            words = re.split(r'[^\w]+', text.lower())
            words = [w for w in words if len(w) > 0]
        
        if not words:
            return 0.0
        
        # Only count words that are in the common words list (not just length >= 3)
        recognized = sum(1 for word in words if word in self.common_words)
        return recognized / len(words) if len(words) > 0 else 0.0
    
    def has_repeated_chars(self, text: str, threshold: int = 5) -> bool:
        """Check for excessive character repetition"""
        pattern = r'(.)\1{' + str(threshold) + r',}'
        return bool(re.search(pattern, text))
    
    def calculate_consonant_clusters(self, text: str) -> float:
        """Detect unusual consonant clusters"""
        if not text:
            return 0.0
        
        vowels = 'aeiouAEIOU'
        consonant_cluster_pattern = r'[^aeiouAEIOU\s\d]{4,}'
        clusters = re.findall(consonant_cluster_pattern, text)
        
        # Filter out common patterns like "http", "www", etc.
        clusters = [c for c in clusters if c.lower() not in ['http', 'https', 'wwww']]
        
        return len(clusters)
    
    def is_gibberish(self, text: str) -> Tuple[bool, str, float]:
        """
        Determine if text is gibberish
        Returns: (is_gibberish, reason, confidence)
        """
        if not text or len(text.strip()) < 3:
            return True, "Text too short or empty", 1.0
        
        # Remove URLs and numbers for analysis
        clean_text = re.sub(r'http\S+|www\.\S+', '', text)
        clean_text = re.sub(r'\d+', '', clean_text)
        
        # Check for excessive character repetition
        if self.has_repeated_chars(text):
            return True, "Excessive character repetition detected", 0.95
        
        # Calculate metrics
        vowel_ratio = self.calculate_vowel_ratio(clean_text)
        word_score = self.calculate_word_score(clean_text)
        consonant_clusters = self.calculate_consonant_clusters(clean_text)
        text_length = len(clean_text.strip())
        
        # Check vowel ratio (normal English/Filipino: 0.35-0.45)
        # For shorter texts, use a more lenient threshold
        vowel_threshold_min = 0.1 if text_length <= 10 else 0.15
        if vowel_ratio < vowel_threshold_min or vowel_ratio > 0.7:
            if text_length >= 5:  # Lower threshold for shorter texts
                return True, f"Unusual vowel ratio: {vowel_ratio:.2f}", 0.85
        
        # Check word recognition - improved for short texts
        # Use same word extraction as calculate_word_score
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            words = re.split(r'[^\w]+', text.lower())
            words = [w for w in words if len(w) > 0]
        
        if len(words) > 0:
            # For texts with words, check if any are recognized
            if word_score == 0.0 and len(words) >= 1:
                # No recognized words at all
                if text_length >= 5:
                    return True, "No recognizable words found", 0.92
            elif word_score < 0.3:
                # For longer texts with multiple words
                if len(words) > 3:
                    return True, f"Too many unrecognized words: {word_score:.2%} recognized", 0.90
                # For shorter texts, be more strict
                elif len(words) >= 1 and text_length >= 5:
                    return True, "No recognizable words found", 0.90
        else:
            # No words found (just characters/punctuation)
            if text_length >= 5:
                return True, "No recognizable words found", 0.88
        
        # Check consonant clusters
        if consonant_clusters > 2:  # Lowered threshold
            return True, f"Excessive consonant clusters detected: {consonant_clusters}", 0.80
        
        # Check if text is just random characters
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars > 0:
            special_ratio = sum(1 for c in text if c in string.punctuation) / alpha_chars
            if special_ratio > 0.3:  # Lowered threshold
                return True, "Too many special characters", 0.85
        
        # Additional check: if text has very low vowel ratio and no recognized words
        if vowel_ratio < 0.2 and word_score == 0.0 and text_length >= 5:
            return True, "Text contains no recognizable words and unusual character patterns", 0.88
        
        # Check for random character sequences (no spaces, no recognized words, short length)
        if text_length >= 5 and text_length <= 20:
            # Check if text has no spaces or very few spaces
            space_count = text.count(' ')
            if space_count <= 1:  # Allow for one space in short texts
                # Check if it's mostly consonants with very few vowels
                if vowel_ratio < 0.25:
                    # If no recognized words or very low word score
                    if word_score == 0.0 or (word_score < 0.2 and len(words) <= 2):
                        return True, "Random character sequence detected", 0.90
                # Also check: if no recognized words and low vowel ratio, it's likely gibberish
                elif word_score == 0.0 and vowel_ratio < 0.3:
                    return True, "No recognizable words found", 0.88
        
        return False, "Text appears legitimate", 0.0


# Global variables
models = {
    'logistic_regression': None,
    'lstm': None,
    'xlm_roberta': None
}

model_metadata = {
    'logistic_regression': {
        'name': 'Logistic Regression + TF-IDF',
        'accuracy': 0.9725,
        'precision': 0.9936,
        'recall': 0.9012,
        'f1': 0.9451,
        'training_time': '~5 seconds',
        'description': 'Traditional ML using TF-IDF features'
    },
    'lstm': {
        'name': 'LSTM (Long Short-Term Memory)',
        'accuracy': 0.8342,
        'precision': 0.9922,
        'recall': 0.3721,
        'f1': 0.5412,
        'training_time': '~2 minutes',
        'description': 'Deep Learning RNN architecture'
    },
    'xlm_roberta': {
        'name': 'XLM-RoBERTa Base',
        'accuracy': 0.9824,
        'precision': 0.9573,
        'recall': 0.9767,
        'f1': 0.9669,
        'training_time': '~10 minutes',
        'description': 'Transformer-based multilingual model'
    }
}

gibberish_detector = GibberishDetector()


def load_thresholds() -> Dict[str, float]:
    """Load per-model decision thresholds if available."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        thresholds_path = os.path.join(project_root, 'thresholds.json')
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Merge with defaults to avoid missing keys
                    merged = Config.DEFAULT_THRESHOLDS.copy()
                    for k, v in data.items():
                        try:
                            merged[k] = float(v)
                        except (TypeError, ValueError):
                            continue
                    return merged
    except Exception as e:
        logger.error(f"Error loading thresholds: {e}")
    return Config.DEFAULT_THRESHOLDS.copy()


MODEL_THRESHOLDS = load_thresholds()


class StandaloneTokenizer:
    """
    Minimal tokenizer compatible with pickled tensorflow.keras Tokenizer objects.
    Supports the subset of features needed for inference (texts_to_sequences).
    """
    
    def __init__(self):
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.word_counts = {}
        self.word_docs = {}
        self.document_count = 0
        self.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.split = ' '
        self.lower = True
        self.char_level = False
        self.oov_token = None
        self.num_words = None
        self._filter_map = None
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            seq = []
            for token in tokens:
                index = self.word_index.get(token)
                if index is None:
                    if self.oov_token is not None and self.oov_token in self.word_index:
                        index = self.word_index[self.oov_token]
                    else:
                        continue
                if self.num_words and index >= self.num_words:
                    continue
                seq.append(index)
            sequences.append(seq)
        return sequences
    
    def _tokenize(self, text: Any) -> List[str]:
        if text is None:
            return []
        if not isinstance(text, str):
            text = str(text)
        if getattr(self, 'lower', True):
            text = text.lower()
        if getattr(self, 'char_level', False):
            return list(text)
        
        filters = getattr(self, 'filters', self.filters)
        split_char = getattr(self, 'split', self.split) or ' '
        translate_map = getattr(self, '_filter_map', None)
        if translate_map is None:
            translate_map = str.maketrans({c: split_char for c in filters})
            setattr(self, '_filter_map', translate_map)
        text = text.translate(translate_map)
        return [tok for tok in text.split(split_char) if tok]


class LogisticRegressionModel:
    """Wrapper for Logistic Regression model."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.loaded = False
    
    def load(self):
        """Load the trained model and vectorizer."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, 'models', 'logistic_regression', 'model_files', 'logistic_regression_taglish_spam_model.pkl')
            vectorizer_path = os.path.join(project_root, 'models', 'logistic_regression', 'model_files', 'tfidf_vectorizer_taglish_spam_model.pkl')
            
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.loaded = True
            logger.info("✓ Logistic Regression model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Error loading Logistic Regression model: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, message: str) -> Dict[str, Any]:
        """Predict spam probability for a message."""
        if not self.loaded:
            return {'error': 'Model not loaded'}
        
        try:
            processed_message = self.preprocess_text(message)
            message_tfidf = self.vectorizer.transform([processed_message])
            
            prediction_proba = self.model.predict_proba(message_tfidf)[0]
            spam_prob = float(prediction_proba[1])
            ham_prob = float(prediction_proba[0])

            threshold = MODEL_THRESHOLDS.get('logistic_regression', Config.DEFAULT_THRESHOLDS['logistic_regression'])
            prediction = 1 if spam_prob >= threshold else 0
            return {
                'prediction': int(prediction),
                'label': 'SPAM' if prediction == 1 else 'HAM',
                'confidence': float(max(spam_prob, ham_prob)),
                'spam_probability': spam_prob,
                'ham_probability': ham_prob
            }
        except Exception as e:
            logger.error(f"Prediction error in Logistic Regression: {e}")
            return {'error': str(e)}


class LSTMModel:
    """Wrapper for the legacy LSTM model without requiring TensorFlow at runtime."""
    
    def __init__(self):
        self.model = None  # retained for compatibility, not used
        self.tokenizer = None
        self.label_encoder = None
        self.config = {}
        self.loaded = False
        
        # Numpy runtime weights
        self.embedding_matrix = None
        self.lstm_kernel = None
        self.lstm_recurrent_kernel = None
        self.lstm_bias = None
        self.dense1_kernel = None
        self.dense1_bias = None
        self.dense2_kernel = None
        self.dense2_bias = None
        self.output_kernel = None
        self.output_bias = None
        
        self.vocab_size = 0
        self.lstm_units = 0
        self.max_sequence_length = Config.MAX_LSTM_LENGTH
    
    def load(self):
        """Load serialized artifacts and hydrate numpy weights."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, 'models', 'lstm', 'model_files', 'lstm_spam_model.h5')
            tokenizer_path = os.path.join(project_root, 'models', 'lstm', 'model_files', 'tokenizer.pkl')
            label_encoder_path = os.path.join(project_root, 'models', 'lstm', 'model_files', 'label_encoder.pkl')
            config_path = os.path.join(project_root, 'models', 'lstm', 'model_files', 'model_config.pkl')
            
            self.tokenizer = self._load_tokenizer(tokenizer_path)
            self.label_encoder = joblib.load(label_encoder_path)
            
            if os.path.exists(config_path):
                loaded_config = joblib.load(config_path)
                if isinstance(loaded_config, dict):
                    self.config = loaded_config
            else:
                self.config = {}
            
            max_len = self.config.get('max_length') or self.config.get('max_sequence_length') or Config.MAX_LSTM_LENGTH
            try:
                self.max_sequence_length = max(1, int(max_len))
            except (TypeError, ValueError):
                self.max_sequence_length = Config.MAX_LSTM_LENGTH
            
            self._load_weights_without_tensorflow(model_path)
            
            self.loaded = True
            logger.info("✓ LSTM model loaded successfully (TensorFlow-free runtime)")
            
            if 'metrics' in self.config:
                self._update_metadata_from_config()
            else:
                metrics_path = os.path.join(project_root, 'models', 'lstm', 'model_files', 'metrics.json')
                if os.path.exists(metrics_path):
                    self._update_metadata_from_file(metrics_path)
                else:
                    logger.info(f"  Using default metrics - Accuracy: {model_metadata['lstm']['accuracy']:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Error loading LSTM model: {e}")
            return False
    
    def _load_weights_without_tensorflow(self, model_path: str):
        """Extract layer weights from the saved Keras H5 file using h5py."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM weight file not found: {model_path}")
        
        with h5py.File(model_path, 'r') as f:
            weights = f['model_weights']
            
            def read(dataset_path: List[str]) -> np.ndarray:
                node = weights
                for key in dataset_path:
                    node = node[key]
                return node[()].astype(np.float32)
            
            self.embedding_matrix = read(['embedding', 'sequential', 'embedding', 'embeddings'])
            self.lstm_kernel = read(['lstm', 'sequential', 'lstm', 'lstm_cell', 'kernel'])
            self.lstm_recurrent_kernel = read(['lstm', 'sequential', 'lstm', 'lstm_cell', 'recurrent_kernel'])
            self.lstm_bias = read(['lstm', 'sequential', 'lstm', 'lstm_cell', 'bias'])
            self.dense1_kernel = read(['dense_1', 'sequential', 'dense_1', 'kernel'])
            self.dense1_bias = read(['dense_1', 'sequential', 'dense_1', 'bias'])
            self.dense2_kernel = read(['dense_2', 'sequential', 'dense_2', 'kernel'])
            self.dense2_bias = read(['dense_2', 'sequential', 'dense_2', 'bias'])
            self.output_kernel = read(['output', 'sequential', 'output', 'kernel'])
            self.output_bias = read(['output', 'sequential', 'output', 'bias'])
        
        self.vocab_size = self.embedding_matrix.shape[0]
        self.lstm_units = self.lstm_recurrent_kernel.shape[0]
    
    def _load_tokenizer(self, tokenizer_path: str):
        """Custom unpickler that maps tensorflow.keras classes to standalone Keras."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        class LegacyTokenizerUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in (
                    'tensorflow.keras.preprocessing.text',
                    'keras.preprocessing.text',
                    'keras.src.legacy.preprocessing.text'
                ) and name == 'Tokenizer':
                    return StandaloneTokenizer
                return super().find_class(module, name)
        
        with open(tokenizer_path, 'rb') as handle:
            return LegacyTokenizerUnpickler(handle).load()
    
    def _update_metadata_from_config(self):
        """Update metadata from config"""
        metrics = self.config.get('metrics', {})
        model_metadata['lstm'].update({
            'accuracy': metrics.get('accuracy', metrics.get('test_accuracy', model_metadata['lstm']['accuracy'])),
            'precision': metrics.get('precision', metrics.get('test_precision', model_metadata['lstm']['precision'])),
            'recall': metrics.get('recall', metrics.get('test_recall', model_metadata['lstm']['recall'])),
            'f1': metrics.get('f1', metrics.get('f1_score', metrics.get('test_f1', model_metadata['lstm']['f1']))),
            'training_time': self.config.get('training_time', metrics.get('training_time', model_metadata['lstm']['training_time']))
        })
        logger.info(f"  Loaded metrics - Accuracy: {model_metadata['lstm']['accuracy']:.4f}")
    
    def _update_metadata_from_file(self, path: str):
        """Update metadata from JSON file"""
        with open(path, 'r') as f:
            metrics = json.load(f)
            model_metadata['lstm'].update({
                'accuracy': metrics.get('accuracy', metrics.get('test_accuracy', model_metadata['lstm']['accuracy'])),
                'precision': metrics.get('precision', metrics.get('test_precision', model_metadata['lstm']['precision'])),
                'recall': metrics.get('recall', metrics.get('test_recall', model_metadata['lstm']['recall'])),
                'f1': metrics.get('f1', metrics.get('f1_score', metrics.get('test_f1', model_metadata['lstm']['f1']))),
                'training_time': metrics.get('training_time', model_metadata['lstm']['training_time'])
            })
            logger.info(f"  Loaded metrics from file - Accuracy: {model_metadata['lstm']['accuracy']:.4f}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, message: str) -> Dict[str, Any]:
        """Predict spam probability for a message using numpy runtime."""
        if not self.loaded:
            return {'error': 'Model not loaded'}
        
        try:
            processed_message = self.preprocess_text(message)
            if not self.tokenizer:
                raise RuntimeError("Tokenizer not loaded")
            
            sequences = self.tokenizer.texts_to_sequences([processed_message])
            sequence = sequences[0] if sequences else []
            padded_sequence, valid_length = self._prepare_sequence(sequence)
            spam_prob = float(self._run_inference(padded_sequence, valid_length))

            threshold = MODEL_THRESHOLDS.get('lstm', Config.DEFAULT_THRESHOLDS['lstm'])
            ham_prob = float(1.0 - spam_prob)
            prediction = int(spam_prob >= threshold)
            confidence = max(spam_prob, ham_prob)
            
            if self.label_encoder:
                try:
                    label = self.label_encoder.inverse_transform([prediction])[0]
                except Exception:
                    label = 'spam' if prediction == 1 else 'ham'
            else:
                label = 'spam' if prediction == 1 else 'ham'
            
            return {
                'prediction': prediction,
                'label': label.upper(),
                'confidence': confidence,
                'spam_probability': spam_prob,
                'ham_probability': ham_prob
            }
        except Exception as e:
            logger.error(f"Prediction error in LSTM: {e}")
            return {'error': str(e)}
    
    def _prepare_sequence(self, sequence: List[int]) -> Tuple[np.ndarray, int]:
        """Pad/truncate sequences to the configured max length."""
        max_len = self.max_sequence_length or Config.MAX_LSTM_LENGTH
        padded = np.zeros(max_len, dtype=np.int32)
        if not sequence:
            return padded, 0
        
        truncated = sequence[:max_len]
        valid_length = len(truncated)
        arr = np.array(truncated, dtype=np.int32)
        if self.vocab_size:
            arr = np.where((arr >= 0) & (arr < self.vocab_size), arr, 0)
        padded[:valid_length] = arr
        return padded, valid_length
    
    def _run_inference(self, padded_sequence: np.ndarray, valid_length: int) -> float:
        """Execute the forward pass using numpy."""
        if self.embedding_matrix is None or self.lstm_kernel is None:
            raise RuntimeError("LSTM weights have not been loaded")
        
        embeddings = self.embedding_matrix[padded_sequence]
        hidden_state = self._lstm_forward(embeddings, valid_length)
        
        x = np.dot(hidden_state, self.dense1_kernel) + self.dense1_bias
        x = self._relu(x)
        x = np.dot(x, self.dense2_kernel) + self.dense2_bias
        x = self._relu(x)
        logits = float(np.dot(x, self.output_kernel).squeeze() + self.output_bias.squeeze())
        return float(self._sigmoid(logits))
    
    def _lstm_forward(self, embeddings: np.ndarray, valid_length: int) -> np.ndarray:
        """Manual LSTM cell forward pass (single layer)."""
        units = self.lstm_units
        h = np.zeros(units, dtype=np.float32)
        c = np.zeros(units, dtype=np.float32)
        steps = valid_length if valid_length > 0 else embeddings.shape[0]
        
        for t in range(steps):
            x_t = embeddings[t]
            gates = (
                np.dot(x_t, self.lstm_kernel) +
                np.dot(h, self.lstm_recurrent_kernel) +
                self.lstm_bias
            )
            i, f, g, o = np.split(gates, 4)
            i = self._sigmoid(i)
            f = self._sigmoid(f)
            g = np.tanh(g)
            o = self._sigmoid(o)
            c = f * c + i * g
            h = o * np.tanh(c)
        
        return h
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)


class XLMRobertaModel:
    """Wrapper for XLM-RoBERTa model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        self.loaded = False
    
    def load(self):
        """Load the trained XLM-RoBERTa model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_dir = os.path.join(project_root, 'models', 'xlm-roberta', 'saved_model')
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            
            # Load label mapping
            label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r') as f:
                    loaded_mapping = json.load(f)
                    if isinstance(loaded_mapping, dict):
                        if 'label2id' in loaded_mapping:
                            self.label_mapping = loaded_mapping['label2id']
                        elif 'id2label' in loaded_mapping:
                            id2label = loaded_mapping['id2label']
                            self.label_mapping = {v: int(k) for k, v in id2label.items()}
                        else:
                            self.label_mapping = loaded_mapping
                    else:
                        self.label_mapping = loaded_mapping
            else:
                self.label_mapping = {"ham": 0, "spam": 1}
            
            # Load metrics
            self._load_metrics(model_dir, project_root)
            
            self.model.eval()
            self.loaded = True
            logger.info("✓ XLM-RoBERTa model loaded successfully")
            logger.info(f"  Label mapping: {self.label_mapping}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Error loading XLM-RoBERTa model: {e}")
            return False
    
    def _load_metrics(self, model_dir: str, project_root: str):
        """Load metrics from various sources"""
        metrics_loaded = False
        
        # Try config.json
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'metrics' in config:
                    self._update_metadata(config['metrics'])
                    metrics_loaded = True
                    logger.info(f"  Loaded metrics from config - Accuracy: {model_metadata['xlm_roberta']['accuracy']:.4f}")
        
        # Try metrics.json
        if not metrics_loaded:
            metrics_path = os.path.join(model_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self._update_metadata(metrics)
                    metrics_loaded = True
                    logger.info(f"  Loaded metrics from file - Accuracy: {model_metadata['xlm_roberta']['accuracy']:.4f}")
        
        if not metrics_loaded:
            logger.info(f"  Using default metrics - Accuracy: {model_metadata['xlm_roberta']['accuracy']:.4f}")
    
    def _update_metadata(self, metrics: dict):
        """Update model metadata"""
        model_metadata['xlm_roberta'].update({
            'accuracy': metrics.get('accuracy', metrics.get('test_accuracy', model_metadata['xlm_roberta']['accuracy'])),
            'precision': metrics.get('precision', metrics.get('test_precision', model_metadata['xlm_roberta']['precision'])),
            'recall': metrics.get('recall', metrics.get('test_recall', model_metadata['xlm_roberta']['recall'])),
            'f1': metrics.get('f1', metrics.get('f1_score', metrics.get('test_f1', model_metadata['xlm_roberta']['f1']))),
            'training_time': metrics.get('training_time', model_metadata['xlm_roberta']['training_time'])
        })
    
    def predict(self, message: str) -> Dict[str, Any]:
        """Predict spam probability for a message."""
        if not self.loaded:
            return {'error': 'Model not loaded'}
        
        try:
            import torch
            import torch.nn.functional as F
            
            inputs = self.tokenizer(
                message,
                return_tensors="pt",
                truncation=True,
                max_length=Config.MAX_TRANSFORMER_LENGTH,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)[0]
            
            prediction_idx = torch.argmax(probabilities).item()
            confidence = float(probabilities[prediction_idx])
            
            spam_idx = self.label_mapping.get('spam', 1)
            ham_idx = self.label_mapping.get('ham', 0)
            
            spam_prob = float(probabilities[spam_idx]) if spam_idx < len(probabilities) else 0.0
            ham_prob = float(probabilities[ham_idx]) if ham_idx < len(probabilities) else 0.0

            threshold = MODEL_THRESHOLDS.get('xlm_roberta', Config.DEFAULT_THRESHOLDS['xlm_roberta'])
            prediction = int(spam_prob >= threshold)
            label_text = 'SPAM' if prediction == 1 else 'HAM'
            
            return {
                'prediction': prediction,
                'label': label_text,
                'confidence': confidence,
                'spam_probability': spam_prob,
                'ham_probability': ham_prob
            }
        except Exception as e:
            logger.error(f"Prediction error in XLM-RoBERTa: {e}")
            return {'error': f'Prediction error: {str(e)}'}


def initialize_models():
    """Initialize all three models."""
    logger.info("🚀 Initializing models...")
    
    models['logistic_regression'] = LogisticRegressionModel()
    models['logistic_regression'].load()
    
    models['lstm'] = LSTMModel()
    models['lstm'].load()
    
    models['xlm_roberta'] = XLMRobertaModel()
    models['xlm_roberta'].load()
    
    logger.info("✓ All models initialized!")


def get_model(model_name: str):
    """Lazy load model if not already loaded"""
    if not models[model_name] or not models[model_name].loaded:
        logger.info(f"Loading {model_name} on demand...")
        if model_name == 'logistic_regression':
            models[model_name] = LogisticRegressionModel()
        elif model_name == 'lstm':
            models[model_name] = LSTMModel()
        elif model_name == 'xlm_roberta':
            models[model_name] = XLMRobertaModel()
        models[model_name].load()
    return models[model_name]


# HTML Template (continued in next part due to length)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Taglish Spam Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #000000;
            color: #f5f5f7;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 80px;
            animation: fadeInDown 0.8s ease-out;
        }
        
        .header h1 {
            font-size: 64px;
            font-weight: 700;
            letter-spacing: -2px;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 21px;
            color: #a1a1a6;
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .input-section {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 48px;
            margin-bottom: 60px;
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }
        
        .input-label {
            font-size: 17px;
            font-weight: 500;
            color: #f5f5f7;
            margin-bottom: 16px;
            display: block;
        }
        
        .char-counter {
            float: right;
            font-size: 13px;
            color: #86868b;
        }
        
        .char-counter.warning {
            color: #ff9f0a;
        }
        
        .char-counter.error {
            color: #ff453a;
        }
        
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            color: #f5f5f7;
            font-size: 16px;
            font-family: 'Inter', sans-serif;
            resize: vertical;
            transition: all 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            background: rgba(255, 255, 255, 0.08);
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        textarea.error-input {
            border-color: #ff453a;
            box-shadow: 0 0 0 4px rgba(255, 69, 58, 0.1);
        }
        
        textarea::placeholder {
            color: #86868b;
        }
        
        .sample-messages {
            margin-top: 24px;
        }
        
        .sample-header {
            font-size: 15px;
            font-weight: 500;
            color: #a1a1a6;
            margin-bottom: 12px;
        }
        
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
        }
        
        .sample-btn {
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #f5f5f7;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .sample-btn:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: #667eea;
            transform: translateY(-2px);
        }
        
        .sample-badge {
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-spam-sample {
            background: rgba(255, 69, 58, 0.2);
            color: #ff453a;
        }
        
        .badge-ham-sample {
            background: rgba(48, 209, 88, 0.2);
            color: #30d158;
        }
        
        .sample-text {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .btn-primary {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 17px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 24px;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:active {
            transform: translateY(0);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            margin: 0 auto 20px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 20px;
            border-radius: 16px;
            margin-bottom: 32px;
            display: none;
            animation: slideIn 0.3s ease-out;
        }
        
        .alert-error {
            background: rgba(255, 69, 58, 0.15);
            border: 1px solid rgba(255, 69, 58, 0.3);
            color: #ff453a;
        }
        
        .alert-warning {
            background: rgba(255, 159, 10, 0.15);
            border: 1px solid rgba(255, 159, 10, 0.3);
            color: #ff9f0a;
        }
        
        .alert-gibberish {
            background: rgba(255, 214, 10, 0.15);
            border: 1px solid rgba(255, 214, 10, 0.3);
            color: #ffd60a;
        }
        
        .alert strong {
            display: block;
            margin-bottom: 8px;
            font-size: 15px;
        }
        
        .results-section {
            display: none;
            animation: fadeInUp 0.6s ease-out;
        }
        
        .results-header {
            text-align: center;
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 48px;
            color: #f5f5f7;
        }
        
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin-bottom: 60px;
        }
        
        .model-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 32px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .model-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transform: scaleX(0);
            transition: transform 0.4s ease;
        }
        
        .model-card:hover::before {
            transform: scaleX(1);
        }
        
        .model-card:hover {
            transform: translateY(-8px);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }
        
        .model-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .model-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            background: rgba(255, 255, 255, 0.05);
        }
        
        .model-name {
            font-size: 19px;
            font-weight: 600;
            color: #f5f5f7;
        }
        
        .prediction-badge {
            display: inline-block;
            padding: 12px 24px;
            border-radius: 100px;
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .badge-spam {
            background: rgba(255, 69, 58, 0.2);
            color: #ff453a;
            border: 1px solid rgba(255, 69, 58, 0.3);
        }
        
        .badge-ham {
            background: rgba(48, 209, 88, 0.2);
            color: #30d158;
            border: 1px solid rgba(48, 209, 88, 0.3);
        }
        
        .confidence-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 100px;
            overflow: hidden;
            margin: 16px 0;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 100px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .fill-spam {
            background: linear-gradient(90deg, #ff453a, #ff6961);
        }
        
        .fill-ham {
            background: linear-gradient(90deg, #30d158, #32d74b);
        }
        
        .probabilities {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: #86868b;
            margin-top: 12px;
        }
        
        .prob-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .prob-value {
            font-size: 17px;
            font-weight: 600;
            color: #f5f5f7;
        }
        
        .comparison-section {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 48px;
            animation: fadeInUp 0.8s ease-out 0.4s both;
        }
        
        .comparison-header {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 32px;
            color: #f5f5f7;
        }
        
        .table-responsive {
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
        }
        
        thead th {
            text-align: left;
            padding: 12px 16px;
            font-size: 13px;
            font-weight: 500;
            color: #86868b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        tbody tr {
            background: rgba(255, 255, 255, 0.03);
            transition: all 0.3s ease;
        }
        
        tbody tr:hover {
            background: rgba(255, 255, 255, 0.06);
        }
        
        tbody td {
            padding: 20px 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #f5f5f7;
            font-size: 15px;
        }
        
        tbody td:first-child {
            border-left: 1px solid rgba(255, 255, 255, 0.05);
            border-top-left-radius: 12px;
            border-bottom-left-radius: 12px;
        }
        
        tbody td:last-child {
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            border-top-right-radius: 12px;
            border-bottom-right-radius: 12px;
        }
        
        .model-description {
            font-size: 13px;
            color: #86868b;
            margin-top: 4px;
        }
        
        .metric-badge {
            display: inline-block;
            padding: 4px 12px;
            background: rgba(102, 126, 234, 0.15);
            color: #667eea;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Modal Dialog Styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(8px);
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
            align-items: center;
            justify-content: center;
        }
        
        .modal-overlay.show {
            display: flex;
        }
        
        .modal-dialog {
            background: rgba(28, 28, 30, 0.95);
            backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 24px;
            padding: 40px;
            max-width: 500px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.3s ease-out;
            position: relative;
        }
        
        .modal-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .modal-icon {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            background: rgba(255, 214, 10, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
        }
        
        .modal-title {
            font-size: 24px;
            font-weight: 600;
            color: #f5f5f7;
            margin: 0;
        }
        
        .modal-body {
            color: #a1a1a6;
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 32px;
        }
        
        .modal-body strong {
            color: #ffd60a;
            display: block;
            margin-bottom: 12px;
            font-size: 17px;
        }
        
        .modal-footer {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
        }
        
        .btn-modal {
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        }
        
        .btn-modal-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-modal-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        }
        
        .btn-modal-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #f5f5f7;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-modal-secondary:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 40px;
            }
            
            .input-section, .comparison-section {
                padding: 32px 24px;
            }
            
            .models-grid {
                grid-template-columns: 1fr;
            }
            
            .sample-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Spam Detection</h1>
            <p>Advanced AI-powered spam detection for English and Filipino messages using three different models</p>
        </div>

        <div class="input-section">
            <label class="input-label">
                Enter your message
                <span class="char-counter" id="charCounter">0 / 5000</span>
            </label>
            <form id="spamForm">
                <textarea 
                    id="messageInput" 
                    placeholder="Type or paste your message here..."
                    maxlength="5000"
                    required
                ></textarea>
                
                <div class="sample-messages">
                    <div class="sample-header">📝 Try sample messages:</div>
                    <div class="sample-grid" id="sampleGrid"></div>
                </div>
                
                <button type="submit" class="btn-primary" id="submitBtn">
                    🔍 Check for Spam
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #86868b;">Analyzing with all models...</p>
            </div>
        </div>

        <div class="alert alert-gibberish" id="gibberishAlert">
            <strong>⚠️ Gibberish Detected!</strong>
            <span id="gibberishMessage"></span>
        </div>
        
        <div class="alert alert-error" id="errorAlert">
            <strong>❌ Error:</strong>
            <span id="errorMessage"></span>
        </div>

        <!-- Gibberish Detection Modal Dialog -->
        <div class="modal-overlay" id="gibberishModal">
            <div class="modal-dialog">
                <div class="modal-header">
                    <div class="modal-icon">⚠️</div>
                    <h2 class="modal-title">Invalid Input Detected</h2>
                </div>
                <div class="modal-body">
                    <strong>What you entered is invalid.</strong>
                    <p id="gibberishModalMessage"></p>
                    <p style="margin-top: 16px; color: #f5f5f7;">Please type a valid spam or ham sentence with recognizable words.</p>
                </div>
                <div class="modal-footer">
                    <button class="btn-modal btn-modal-primary" id="gibberishModalOk">OK, I Understand</button>
                </div>
            </div>
        </div>

        <div class="results-section" id="resultsSection">
            <h2 class="results-header">📊 Detection Results</h2>
            
            <div class="models-grid" id="modelsGrid"></div>

            <div class="comparison-section">
                <h3 class="comparison-header">📈 Model Performance Comparison</h3>
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>Training Time</th>
                            </tr>
                        </thead>
                        <tbody id="metricsTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const MAX_LENGTH = 5000;
        const form = document.getElementById('spamForm');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');
        const modelsGrid = document.getElementById('modelsGrid');
        const errorAlert = document.getElementById('errorAlert');
        const errorMessage = document.getElementById('errorMessage');
        const gibberishAlert = document.getElementById('gibberishAlert');
        const gibberishMessage = document.getElementById('gibberishMessage');
        const messageInput = document.getElementById('messageInput');
        const charCounter = document.getElementById('charCounter');
        const sampleGrid = document.getElementById('sampleGrid');
        const gibberishModal = document.getElementById('gibberishModal');
        const gibberishModalMessage = document.getElementById('gibberishModalMessage');
        const gibberishModalOk = document.getElementById('gibberishModalOk');

        const modelConfigs = {
            logistic_regression: {
                name: 'Logistic Regression',
                icon: '📊',
                color: '#667eea'
            },
            lstm: {
                name: 'LSTM',
                icon: '🧠',
                color: '#ff6b6b'
            },
            xlm_roberta: {
                name: 'XLM-RoBERTa',
                icon: '🤖',
                color: '#51cf66'
            }
        };

        // Character counter
        messageInput.addEventListener('input', () => {
            const length = messageInput.value.length;
            charCounter.textContent = `${length} / ${MAX_LENGTH}`;
            
            if (length > MAX_LENGTH * 0.9) {
                charCounter.classList.add('error');
                charCounter.classList.remove('warning');
            } else if (length > MAX_LENGTH * 0.7) {
                charCounter.classList.add('warning');
                charCounter.classList.remove('error');
            } else {
                charCounter.classList.remove('warning', 'error');
            }
            
            // Clear errors when typing
            messageInput.classList.remove('error-input');
            hideAlert(gibberishAlert);
            hideAlert(errorAlert);
        });

        // Load sample messages
        async function loadSampleMessages() {
            try {
                const response = await fetch('/samples');
                const data = await response.json();
                
                // Mix spam and ham samples
                const allSamples = [
                    ...data.spam.slice(0, 3).map(text => ({ text, type: 'spam' })),
                    ...data.ham.slice(0, 3).map(text => ({ text, type: 'ham' }))
                ];
                
                // Shuffle
                allSamples.sort(() => Math.random() - 0.5);
                
                allSamples.forEach(sample => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'sample-btn';
                    btn.innerHTML = `
                        <span class="sample-badge badge-${sample.type}-sample">${sample.type}</span>
                        <span class="sample-text">${sample.text}</span>
                    `;
                    btn.onclick = () => {
                        messageInput.value = sample.text;
                        messageInput.dispatchEvent(new Event('input'));
                        messageInput.focus();
                    };
                    sampleGrid.appendChild(btn);
                });
            } catch (error) {
                console.error('Failed to load samples:', error);
            }
        }

        loadSampleMessages();

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const message = messageInput.value.trim();
            
            if (!message) {
                showAlert(errorAlert, 'Please enter a message to analyze.');
                return;
            }
            
            if (message.length > MAX_LENGTH) {
                showAlert(errorAlert, `Message is too long. Maximum ${MAX_LENGTH} characters allowed.`);
                return;
            }
            
            submitBtn.disabled = true;
            loading.style.display = 'block';
            resultsSection.style.display = 'none';
            hideAlert(errorAlert);
            hideAlert(gibberishAlert);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                if (response.status === 429) {
                    throw new Error('Rate limit exceeded. Please wait a moment before trying again.');
                }

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }
                
                if (data.gibberish_detected) {
                    showGibberishModal(data.gibberish_reason, data.gibberish_confidence);
                    loading.style.display = 'none';
                    submitBtn.disabled = false;
                    return;
                }

                displayResults(data);
                resultsSection.style.display = 'block';
                
                // Scroll to results
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

            } catch (error) {
                showAlert(errorAlert, error.message);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function showAlert(alertElement, message) {
            const messageElement = alertElement.querySelector('span:last-child');
            messageElement.textContent = message;
            alertElement.style.display = 'block';
        }

        function hideAlert(alertElement) {
            alertElement.style.display = 'none';
        }

        function showGibberishModal(reason, confidence) {
            const reasonText = reason || 'The input contains unrecognizable or nonsensical text.';
            const confidenceText = confidence ? ` (Confidence: ${(confidence * 100).toFixed(1)}%)` : '';
            gibberishModalMessage.textContent = reasonText + confidenceText;
            gibberishModal.classList.add('show');
            messageInput.classList.add('error-input');
        }

        function hideGibberishModal() {
            gibberishModal.classList.remove('show');
            messageInput.classList.remove('error-input');
            messageInput.focus();
        }

        // Close modal handlers
        gibberishModalOk.addEventListener('click', hideGibberishModal);
        
        // Close modal when clicking outside
        gibberishModal.addEventListener('click', (e) => {
            if (e.target === gibberishModal) {
                hideGibberishModal();
            }
        });

        // Close modal with ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && gibberishModal.classList.contains('show')) {
                hideGibberishModal();
            }
        });

        function displayResults(data) {
            modelsGrid.innerHTML = '';

            ['logistic_regression', 'lstm', 'xlm_roberta'].forEach(modelKey => {
                const result = data[modelKey];
                const config = modelConfigs[modelKey];
                
                const card = createModelCard(config, result);
                modelsGrid.appendChild(card);
            });

            displayMetricsTable(data.metadata);
        }

        function createModelCard(config, result) {
            const card = document.createElement('div');
            card.className = 'model-card';

            if (result.error) {
                card.innerHTML = `
                    <div class="model-header">
                        <div class="model-icon">${config.icon}</div>
                        <div class="model-name">${config.name}</div>
                    </div>
                    <div style="padding: 20px; background: rgba(255, 69, 58, 0.1); border-radius: 12px; color: #ff453a;">
                        ⚠️ ${result.error}
                    </div>
                `;
                return card;
            }

            const isSpam = result.label === 'SPAM';
            const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
            const fillClass = isSpam ? 'fill-spam' : 'fill-ham';

            card.innerHTML = `
                <div class="model-header">
                    <div class="model-icon">${config.icon}</div>
                    <div class="model-name">${config.name}</div>
                </div>
                <div>
                    <span class="prediction-badge ${badgeClass}">
                        ${isSpam ? '🚫' : '✅'} ${result.label}
                    </span>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${fillClass}" style="width: ${result.confidence * 100}%"></div>
                    </div>
                    <div style="text-align: center; margin: 12px 0;">
                        <span style="font-size: 24px; font-weight: 600; color: #f5f5f7;">
                            ${(result.confidence * 100).toFixed(1)}%
                        </span>
                        <span style="font-size: 13px; color: #86868b; margin-left: 4px;">confident</span>
                    </div>
                    <div class="probabilities">
                        <div class="prob-item">
                            <span>🚫 Spam</span>
                            <span class="prob-value">${(result.spam_probability * 100).toFixed(2)}%</span>
                        </div>
                        <div class="prob-item" style="text-align: right;">
                            <span>✅ Ham</span>
                            <span class="prob-value">${(result.ham_probability * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
            `;

            return card;
        }

        function displayMetricsTable(metadata) {
            const tbody = document.getElementById('metricsTableBody');
            tbody.innerHTML = '';

            const models = [
                { key: 'logistic_regression', icon: '📊' },
                { key: 'lstm', icon: '🧠' },
                { key: 'xlm_roberta', icon: '🤖' }
            ];

            models.forEach(model => {
                const meta = metadata[model.key];
                const row = document.createElement('tr');
                
                row.innerHTML = `
                    <td>
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 24px;">${model.icon}</span>
                            <div>
                                <div style="font-weight: 600;">${meta.name}</div>
                                <div class="model-description">${meta.description}</div>
                            </div>
                        </div>
                    </td>
                    <td><strong>${(meta.accuracy * 100).toFixed(2)}%</strong></td>
                    <td>${(meta.precision * 100).toFixed(2)}%</td>
                    <td>${(meta.recall * 100).toFixed(2)}%</td>
                    <td>${(meta.f1 * 100).toFixed(2)}%</td>
                    <td><span class="metric-badge">${meta.training_time}</span></td>
                `;
                
                tbody.appendChild(row);
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/samples')
def get_samples():
    """Return sample messages."""
    return jsonify(SAMPLE_MESSAGES)


@app.route('/predict', methods=['POST'])
@rate_limit
def predict():
    """Handle prediction requests with gibberish detection."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if len(message) > Config.MAX_MESSAGE_LENGTH:
            return jsonify({'error': f'Message too long. Maximum {Config.MAX_MESSAGE_LENGTH} characters.'}), 400
        
        # Check for gibberish
        is_gibberish, reason, confidence = gibberish_detector.is_gibberish(message)
        
        if is_gibberish:
            return jsonify({
                'gibberish_detected': True,
                'gibberish_reason': reason,
                'gibberish_confidence': confidence
            })
        
        # Get predictions from all models
        results = {
            'gibberish_detected': False,
            'logistic_regression': get_model('logistic_regression').predict(message),
            'lstm': get_model('lstm').predict(message),
            'xlm_roberta': get_model('xlm_roberta').predict(message),
            'metadata': model_metadata
        }
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    model_status = {
        'logistic_regression': models['logistic_regression'].loaded if models['logistic_regression'] else False,
        'lstm': models['lstm'].loaded if models['lstm'] else False,
        'xlm_roberta': models['xlm_roberta'].loaded if models['xlm_roberta'] else False
    }
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': model_status,
        'metadata': model_metadata
    })


if __name__ == '__main__':
    initialize_models()
    
    logger.info("\n🚀 Starting Flask Web UI...")
    logger.info(f"🌐 Access the web interface at: http://{Config.HOST}:{Config.PORT}")
    logger.info("⏹️  Press Ctrl+C to stop the server\n")
    
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)