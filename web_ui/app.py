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
        'description': 'Traditional ML approach using TF-IDF features'
    },
    'lstm': {
        'name': 'LSTM (Long Short-Term Memory)',
        'accuracy': 0.0,  # Will be updated when loading
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'training_time': 'TBD',
        'description': 'Deep Learning RNN architecture'
    },
    'xlm_roberta': {
        'name': 'XLM-RoBERTa Base',
        'accuracy': 0.0,  # Will be updated when loading
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
                    'f1': metrics.get('f1', 0.0)
                })
            
            return True
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
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
            prediction_proba = self.model.predict(padded_sequence, verbose=0)[0]
            prediction = int(prediction_proba[0] > 0.5)
            confidence = float(prediction_proba[0]) if prediction == 1 else float(1 - prediction_proba[0])
            
            return {
                'prediction': prediction,
                'label': 'SPAM' if prediction == 1 else 'HAM',
                'confidence': confidence,
                'spam_probability': float(prediction_proba[0]),
                'ham_probability': float(1 - prediction_proba[0])
            }
        except Exception as e:
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
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.loaded = True
            print("✅ XLM-RoBERTa model loaded successfully")
            
            return True
        except Exception as e:
            print(f"❌ Error loading XLM-RoBERTa model: {e}")
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
            
            # Map prediction to label
            reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            label = reverse_mapping.get(prediction, 'UNKNOWN')
            
            return {
                'prediction': prediction,
                'label': label.upper(),
                'confidence': confidence,
                'spam_probability': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                'ham_probability': float(probabilities[0])
            }
        except Exception as e:
            return {'error': str(e)}


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


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Taglish Spam Detection - Model Comparison</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header-card h1 {
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .header-card p {
            color: #666;
            font-size: 1.1rem;
        }
        .input-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 30px;
            margin-bottom: 30px;
        }
        .model-card {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 25px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .model-card:hover {
            transform: translateY(-5px);
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        .model-name {
            font-size: 1.3rem;
            font-weight: bold;
            color: #333;
        }
        .prediction-badge {
            font-size: 1.2rem;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
        }
        .badge-spam {
            background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            color: white;
        }
        .badge-ham {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
            color: white;
        }
        .confidence-bar {
            background: #f0f0f0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 15px 0;
        }
        .confidence-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        .confidence-spam {
            background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        }
        .confidence-ham {
            background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        }
        .metrics-table {
            margin-top: 20px;
        }
        .metrics-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        .btn-check-spam {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 40px;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 25px;
            color: white;
            transition: transform 0.2s ease;
        }
        .btn-check-spam:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        .loading-spinner {
            display: none;
        }
        .error-alert {
            border-radius: 10px;
        }
        textarea.form-control {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: border-color 0.3s ease;
        }
        textarea.form-control:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }
        .comparison-table {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 30px;
            margin-top: 30px;
        }
        .model-icon {
            font-size: 2rem;
            margin-right: 10px;
        }
        .icon-lr { color: #667eea; }
        .icon-lstm { color: #ff6b6b; }
        .icon-xlm { color: #51cf66; }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Header -->
        <div class="header-card">
            <h1><i class="fas fa-shield-alt"></i> Taglish Spam Detection</h1>
            <p class="mb-0">Compare three different machine learning models for spam detection</p>
            <p class="text-muted">🇵🇭 English & Filipino | 🤖 Logistic Regression • LSTM • XLM-RoBERTa</p>
        </div>

        <!-- Input Form -->
        <div class="input-card">
            <h4 class="mb-3"><i class="fas fa-envelope"></i> Enter Message to Check</h4>
            <form id="spamForm">
                <div class="mb-3">
                    <textarea 
                        class="form-control" 
                        id="messageInput" 
                        rows="4" 
                        placeholder="Type or paste your message here... (English or Filipino)"
                        required
                    ></textarea>
                </div>
                <div class="text-center">
                    <button type="submit" class="btn btn-check-spam">
                        <i class="fas fa-search"></i> Check for Spam
                    </button>
                    <div class="loading-spinner mt-3">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2">Analyzing with all models...</p>
                    </div>
                </div>
            </form>
        </div>

        <!-- Error Alert -->
        <div id="errorAlert" class="alert alert-danger error-alert" style="display: none;" role="alert">
            <i class="fas fa-exclamation-triangle"></i> <span id="errorMessage"></span>
        </div>

        <!-- Results Container -->
        <div id="resultsContainer" style="display: none;">
            <h3 class="text-white text-center mb-4"><i class="fas fa-chart-bar"></i> Detection Results</h3>
            
            <!-- Model Results -->
            <div class="row">
                <!-- Logistic Regression -->
                <div class="col-md-4 mb-3">
                    <div class="model-card" id="lr-card">
                        <div class="model-header">
                            <div>
                                <i class="fas fa-calculator model-icon icon-lr"></i>
                                <span class="model-name">Logistic Regression</span>
                            </div>
                        </div>
                        <div id="lr-result"></div>
                    </div>
                </div>

                <!-- LSTM -->
                <div class="col-md-4 mb-3">
                    <div class="model-card" id="lstm-card">
                        <div class="model-header">
                            <div>
                                <i class="fas fa-brain model-icon icon-lstm"></i>
                                <span class="model-name">LSTM</span>
                            </div>
                        </div>
                        <div id="lstm-result"></div>
                    </div>
                </div>

                <!-- XLM-RoBERTa -->
                <div class="col-md-4 mb-3">
                    <div class="model-card" id="xlm-card">
                        <div class="model-header">
                            <div>
                                <i class="fas fa-robot model-icon icon-xlm"></i>
                                <span class="model-name">XLM-RoBERTa</span>
                            </div>
                        </div>
                        <div id="xlm-result"></div>
                    </div>
                </div>
            </div>

            <!-- Performance Metrics Comparison -->
            <div class="comparison-table">
                <h4 class="mb-4"><i class="fas fa-trophy"></i> Model Performance Comparison</h4>
                <div class="table-responsive">
                    <table class="table table-hover metrics-table">
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
                            <!-- Will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const form = document.getElementById('spamForm');
        const resultsContainer = document.getElementById('resultsContainer');
        const loadingSpinner = document.querySelector('.loading-spinner');
        const errorAlert = document.getElementById('errorAlert');
        const errorMessage = document.getElementById('errorMessage');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const message = document.getElementById('messageInput').value;
            
            // Show loading
            loadingSpinner.style.display = 'block';
            resultsContainer.style.display = 'none';
            errorAlert.style.display = 'none';
            form.querySelector('button[type="submit"]').disabled = true;

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

                // Display results
                displayResults(data);
                resultsContainer.style.display = 'block';

            } catch (error) {
                errorMessage.textContent = error.message;
                errorAlert.style.display = 'block';
            } finally {
                loadingSpinner.style.display = 'none';
                form.querySelector('button[type="submit"]').disabled = false;
            }
        });

        function displayResults(data) {
            // Display Logistic Regression result
            displayModelResult('lr', data.logistic_regression);
            
            // Display LSTM result
            displayModelResult('lstm', data.lstm);
            
            // Display XLM-RoBERTa result
            displayModelResult('xlm', data.xlm_roberta);

            // Display metrics comparison
            displayMetricsComparison(data.metadata);
        }

        function displayModelResult(modelKey, result) {
            const resultDiv = document.getElementById(`${modelKey}-result`);
            
            if (result.error) {
                resultDiv.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-circle"></i> ${result.error}
                    </div>
                `;
                return;
            }

            const isSpam = result.label === 'SPAM';
            const badgeClass = isSpam ? 'badge-spam' : 'badge-ham';
            const confidenceClass = isSpam ? 'confidence-spam' : 'confidence-ham';
            const icon = isSpam ? 'fa-ban' : 'fa-check-circle';

            resultDiv.innerHTML = `
                <div class="text-center mb-3">
                    <span class="prediction-badge ${badgeClass}">
                        <i class="fas ${icon}"></i> ${result.label}
                    </span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill ${confidenceClass}" style="width: ${result.confidence * 100}%">
                        ${(result.confidence * 100).toFixed(1)}% confident
                    </div>
                </div>
                <div class="mt-3">
                    <small class="text-muted">
                        <strong>Spam probability:</strong> ${(result.spam_probability * 100).toFixed(2)}%<br>
                        <strong>Ham probability:</strong> ${(result.ham_probability * 100).toFixed(2)}%
                    </small>
                </div>
            `;
        }

        function displayMetricsComparison(metadata) {
            const tbody = document.getElementById('metricsTableBody');
            
            const models = [
                { key: 'logistic_regression', icon: 'fa-calculator', color: '#667eea' },
                { key: 'lstm', icon: 'fa-brain', color: '#ff6b6b' },
                { key: 'xlm_roberta', icon: 'fa-robot', color: '#51cf66' }
            ];

            tbody.innerHTML = models.map(model => {
                const meta = metadata[model.key];
                return `
                    <tr>
                        <td>
                            <i class="fas ${model.icon}" style="color: ${model.color}"></i>
                            <strong>${meta.name}</strong>
                            <br><small class="text-muted">${meta.description}</small>
                        </td>
                        <td><strong>${(meta.accuracy * 100).toFixed(2)}%</strong></td>
                        <td>${(meta.precision * 100).toFixed(2)}%</td>
                        <td>${(meta.recall * 100).toFixed(2)}%</td>
                        <td>${(meta.f1 * 100).toFixed(2)}%</td>
                        <td><span class="badge bg-info">${meta.training_time}</span></td>
                    </tr>
                `;
            }).join('');
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
        'models_loaded': model_status
    })


if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    
    # Run Flask app
    print("\n🌐 Starting Flask Web UI...")
    print("🔗 Access the web interface at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)