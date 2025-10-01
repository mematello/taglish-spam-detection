#!/usr/bin/env python3
"""
Taglish Spam Detection - Flask Web UI
======================================
A web interface to compare three different spam detection models:
- Logistic Regression + TF-IDF
- LSTM (Deep Learning)
- XLM-RoBERTa (Transformer)

Author: Claude
Date: September 2025
"""

from flask import Flask, render_template_string, request, jsonify
import joblib
import os
import re
import numpy as np
import json
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store loaded models
models = {
    'logistic_regression': None,
    'lstm': None,
    'xlm_roberta': None
}

model_metadata = {
    'logistic_regression': {
        'name': 'Logistic Regression + TF-IDF',
        'accuracy': 0.9740,
        'precision': 0.9968,
        'recall': 0.9041,
        'f1': 0.9482,
        'training_time': '~5 seconds',
        'description': 'Traditional ML using TF-IDF features'
    },
    'lstm': {
        'name': 'LSTM (Long Short-Term Memory)',
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'training_time': 'TBD',
        'description': 'Deep Learning RNN architecture'
    },
    'xlm_roberta': {
        'name': 'XLM-RoBERTa Base',
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'training_time': 'TBD',
        'description': 'Transformer-based multilingual model'
    }
}


class LogisticRegressionModel:
    """Wrapper for Logistic Regression model."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.loaded = False
    
    def load(self):
        """Load the trained model and vectorizer."""
        try:
            model_path = '../models/logistic_regression/model_files/logistic_regression_taglish_spam_model.pkl'
            vectorizer_path = '../models/logistic_regression/model_files/tfidf_vectorizer_taglish_spam_model.pkl'
            
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.loaded = True
            print("✅ Logistic Regression model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading Logistic Regression model: {e}")
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
            
            prediction = self.model.predict(message_tfidf)[0]
            probabilities = self.model.predict_proba(message_tfidf)[0]
            confidence = probabilities.max()
            
            return {
                'prediction': int(prediction),
                'label': 'SPAM' if prediction == 1 else 'HAM',
                'confidence': float(confidence),
                'spam_probability': float(probabilities[1]),
                'ham_probability': float(probabilities[0])
            }
        except Exception as e:
            return {'error': str(e)}


class LSTMModel:
    """Wrapper for LSTM model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.loaded = False
    
    def load(self):
        """Load the trained LSTM model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model_path = '../models/lstm/model_files/lstm_spam_model.h5'
            tokenizer_path = '../models/lstm/model_files/tokenizer.pkl'
            label_encoder_path = '../models/lstm/model_files/label_encoder.pkl'
            config_path = '../models/lstm/model_files/model_config.pkl'
            
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load tokenizer and label encoder
            self.tokenizer = joblib.load(tokenizer_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.config = joblib.load(config_path)
            
            self.loaded = True
            print("✅ LSTM model loaded successfully")
            
            # Update metadata if available
            if 'metrics' in self.config:
                metrics = self.config['metrics']
                model_metadata['lstm'].update({
                    'accuracy': metrics.get('accuracy', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'f1': metrics.get('f1', 0.0),
                    'training_time': self.config.get('training_time', 'TBD')
                })
            
            return True
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            import traceback
            traceback.print_exc()
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
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            processed_message = self.preprocess_text(message)
            
            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([processed_message])
            max_length = self.config.get('max_length', 100)
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
            
            # Predict
            prediction_proba = self.model.predict(padded_sequence, verbose=0)[0][0]
            
            # FIXED: Proper threshold and label mapping
            # If model outputs probability of spam class
            spam_prob = float(prediction_proba)
            ham_prob = float(1 - prediction_proba)
            
            # Determine prediction based on threshold
            prediction = 1 if spam_prob > 0.5 else 0
            confidence = max(spam_prob, ham_prob)
            
            return {
                'prediction': prediction,
                'label': 'SPAM' if prediction == 1 else 'HAM',
                'confidence': confidence,
                'spam_probability': spam_prob,
                'ham_probability': ham_prob
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


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
            
            model_dir = '../models/xlm-roberta/saved_model'
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            
            # Load label mapping
            label_mapping_path = os.path.join(model_dir, 'label_mapping.json')
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r') as f:
                    loaded_mapping = json.load(f)
                    
                # FIXED: Handle nested structure
                if isinstance(loaded_mapping, dict):
                    if 'label2id' in loaded_mapping:
                        self.label_mapping = loaded_mapping['label2id']
                    elif 'id2label' in loaded_mapping:
                        # Convert id2label to label2id
                        id2label = loaded_mapping['id2label']
                        self.label_mapping = {v: int(k) for k, v in id2label.items()}
                    else:
                        self.label_mapping = loaded_mapping
                else:
                    self.label_mapping = loaded_mapping
            else:
                # Default mapping if file doesn't exist
                self.label_mapping = {"ham": 0, "spam": 1}
            
            # Load config to get metrics
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'metrics' in config:
                        metrics = config['metrics']
                        model_metadata['xlm_roberta'].update({
                            'accuracy': metrics.get('accuracy', 0.0),
                            'precision': metrics.get('precision', 0.0),
                            'recall': metrics.get('recall', 0.0),
                            'f1': metrics.get('f1', 0.0),
                            'training_time': metrics.get('training_time', 'TBD')
                        })
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.loaded = True
            print("✅ XLM-RoBERTa model loaded successfully")
            print(f"   Label mapping: {self.label_mapping}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading XLM-RoBERTa model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, message: str) -> Dict[str, Any]:
        """Predict spam probability for a message."""
        if not self.loaded:
            return {'error': 'Model not loaded'}
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Tokenize input
            inputs = self.tokenizer(
                message,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)[0]
            
            # Get prediction
            prediction = torch.argmax(probabilities).item()
            confidence = float(probabilities[prediction])
            
            # FIXED: Proper label mapping
            # Create reverse mapping from id to label
            id_to_label = {v: k for k, v in self.label_mapping.items()}
            label_text = id_to_label.get(prediction, 'UNKNOWN')
            
            # Get spam and ham probabilities
            spam_idx = self.label_mapping.get('spam', 1)
            ham_idx = self.label_mapping.get('ham', 0)
            
            spam_prob = float(probabilities[spam_idx]) if spam_idx < len(probabilities) else 0.0
            ham_prob = float(probabilities[ham_idx]) if ham_idx < len(probabilities) else 0.0
            
            return {
                'prediction': prediction,
                'label': label_text.upper(),
                'confidence': confidence,
                'spam_probability': spam_prob,
                'ham_probability': ham_prob
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'Prediction error: {str(e)}'}


def initialize_models():
    """Initialize all three models."""
    print("🚀 Initializing models...")
    
    # Load Logistic Regression
    models['logistic_regression'] = LogisticRegressionModel()
    models['logistic_regression'].load()
    
    # Load LSTM
    models['lstm'] = LSTMModel()
    models['lstm'].load()
    
    # Load XLM-RoBERTa
    models['xlm_roberta'] = XLMRobertaModel()
    models['xlm_roberta'].load()
    
    print("✅ All models initialized!")


# Modern Apple-inspired HTML Template
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
        
        /* Header Section */
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
        
        /* Input Section */
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
        
        textarea::placeholder {
            color: #86868b;
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
        
        /* Loading State */
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
        
        /* Results Section */
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
        
        /* Comparison Table */
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
        
        /* Error Alert */
        .error-alert {
            display: none;
            background: rgba(255, 69, 58, 0.15);
            border: 1px solid rgba(255, 69, 58, 0.3);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 32px;
            color: #ff453a;
        }
        
        /* Animations */
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
        
        /* Responsive */
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
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Spam Detection</h1>
            <p>Advanced AI-powered spam detection for English and Filipino messages using three different models</p>
        </div>

        <!-- Input Section -->
        <div class="input-section">
            <label class="input-label">Enter your message</label>
            <form id="spamForm">
                <textarea 
                    id="messageInput" 
                    placeholder="Type or paste your message here..."
                    required
                ></textarea>
                <button type="submit" class="btn-primary" id="submitBtn">
                    Check for Spam
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #86868b;">Analyzing with all models...</p>
            </div>
        </div>

        <!-- Error Alert -->
        <div class="error-alert" id="errorAlert">
            <strong>Error:</strong> <span id="errorMessage"></span>
        </div>

        <!-- Results Section -->
        <div class="results-section" id="resultsSection">
            <h2 class="results-header">Detection Results</h2>
            
            <div class="models-grid" id="modelsGrid">
                <!-- Model cards will be inserted here -->
            </div>

            <!-- Performance Comparison -->
            <div class="comparison-section">
                <h3 class="comparison-header">Model Performance Comparison</h3>
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
                        <tbody id="metricsTableBody">
                            <!-- Metrics will be inserted here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('spamForm');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');
        const modelsGrid = document.getElementById('modelsGrid');
        const errorAlert = document.getElementById('errorAlert');
        const errorMessage = document.getElementById('errorMessage');

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

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const message = document.getElementById('messageInput').value;
            
            // Show loading
            submitBtn.disabled = true;
            loading.style.display = 'block';
            resultsSection.style.display = 'none';
            errorAlert.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                displayResults(data);
                resultsSection.style.display = 'block';

            } catch (error) {
                errorMessage.textContent = error.message;
                errorAlert.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function displayResults(data) {
            // Clear previous results
            modelsGrid.innerHTML = '';

            // Display each model result
            ['logistic_regression', 'lstm', 'xlm_roberta'].forEach(modelKey => {
                const result = data[modelKey];
                const config = modelConfigs[modelKey];
                
                const card = createModelCard(config, result);
                modelsGrid.appendChild(card);
            });

            // Display metrics comparison
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
                        ${result.label}
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
                            <span>Spam</span>
                            <span class="prob-value">${(result.spam_probability * 100).toFixed(2)}%</span>
                        </div>
                        <div class="prob-item" style="text-align: right;">
                            <span>Ham</span>
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


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get predictions from all models
        results = {
            'logistic_regression': models['logistic_regression'].predict(message) if models['logistic_regression'] and models['logistic_regression'].loaded else {'error': 'Model not loaded'},
            'lstm': models['lstm'].predict(message) if models['lstm'] and models['lstm'].loaded else {'error': 'Model not loaded'},
            'xlm_roberta': models['xlm_roberta'].predict(message) if models['xlm_roberta'] and models['xlm_roberta'].loaded else {'error': 'Model not loaded'},
            'metadata': model_metadata
        }
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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
    # Initialize models on startup
    initialize_models()
    
    # Run Flask app
    print("\n🌐 Starting Flask Web UI...")
    print("🔗 Access the web interface at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)