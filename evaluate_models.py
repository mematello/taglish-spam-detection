#!/usr/bin/env python3
"""
evaluate_models.py - Unified Model Evaluation Script

This script loads all trained models (XLM-RoBERTa, LSTM, TF-IDF + Logistic Regression)
and evaluates them on the same test set for consistent comparison.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add model directories to path
sys.path.append('models/logistic_regression')
sys.path.append('models/lstm')
sys.path.append('models/xlm-roberta')

def load_and_split_data(dataset_path='dataset/final_spam_ham_dataset.csv', test_size=0.2, random_state=42):
    """
    Load dataset and create consistent train-test split
    
    Args:
        dataset_path: Path to the dataset CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df)
    """
    print(" Loading dataset...")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f" Dataset loaded: {df.shape[0]} samples")
        
        # Check required columns
        if 'label' not in df.columns or 'text' not in df.columns:
            raise ValueError("Dataset must contain 'label' and 'text' columns")
        
        # Clean data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        
        # Encode labels consistently (spam=1, ham=0)
        label_mapping = {'spam': 1, 'ham': 0}
        df['encoded_label'] = df['label'].map(label_mapping)
        
        # Handle case variations
        if df['encoded_label'].isnull().any():
            df['label'] = df['label'].str.lower().str.strip()
            df['encoded_label'] = df['label'].map(label_mapping)
        
        if df['encoded_label'].isnull().any():
            raise ValueError("Invalid labels found. Expected 'spam' or 'ham' only.")
        
        print(f" Label distribution:")
        print(f"   Ham: {sum(df['encoded_label'] == 0)} ({sum(df['encoded_label'] == 0)/len(df)*100:.1f}%)")
        print(f"   Spam: {sum(df['encoded_label'] == 1)} ({sum(df['encoded_label'] == 1)/len(df)*100:.1f}%)")
        
        # Create stratified split
        X = df['text'].tolist()
        y = df['encoded_label'].tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f" Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, df
        
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return None, None, None, None, None

def evaluate_logistic_regression(X_test, y_test):
    """
    Evaluate Logistic Regression model
    
    Args:
        X_test: Test texts
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n Evaluating Logistic Regression Model...")
    
    try:
        import joblib
        
        # Load model and vectorizer
        model_path = 'models/logistic_regression/model_files/logistic_regression_taglish_spam_model.pkl'
        vectorizer_path = 'models/logistic_regression/model_files/tfidf_vectorizer_taglish_spam_model.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(" Logistic Regression model files not found")
            return None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess and vectorize test data
        X_test_processed = [str(text).lower().strip() for text in X_test]
        X_test_tfidf = vectorizer.transform(X_test_processed)
        
        # Make predictions
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        metrics = {
            'model_name': 'Logistic Regression',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        print(f" Accuracy: {metrics['accuracy']:.4f}")
        print(f" F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f" Error evaluating Logistic Regression: {e}")
        return None

def evaluate_lstm(X_test, y_test):
    """
    Evaluate LSTM model
    
    Args:
        X_test: Test texts
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n Evaluating LSTM Model...")
    
    try:
        import tensorflow as tf
        import pickle
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Load model artifacts
        model_path = 'models/lstm/model_files/lstm_spam_model.h5'
        tokenizer_path = 'models/lstm/model_files/tokenizer.pkl'
        config_path = 'models/lstm/model_files/model_config.pkl'
        
        if not all(os.path.exists(p) for p in [model_path, tokenizer_path, config_path]):
            print(" LSTM model files not found")
            return None
        
        # Load artifacts
        model = tf.keras.models.load_model(model_path)
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        max_length = config['max_length']
        
        # Preprocess test data
        def clean_text(text):
            import re
            import string
            if pd.isna(text) or not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        X_test_cleaned = [clean_text(text) for text in X_test]
        sequences = tokenizer.texts_to_sequences(X_test_cleaned)
        X_test_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        # Make predictions
        y_pred_prob = model.predict(X_test_padded, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        metrics = {
            'model_name': 'LSTM',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_prob.flatten().tolist()
        }
        
        print(f" Accuracy: {metrics['accuracy']:.4f}")
        print(f" F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f" Error evaluating LSTM: {e}")
        return None

def evaluate_xlm_roberta(X_test, y_test):
    """
    Evaluate XLM-RoBERTa model
    
    Args:
        X_test: Test texts
        y_test: Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n Evaluating XLM-RoBERTa Model...")
    
    try:
        import torch
        from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        
        model_path = 'models/xlm-roberta/saved_model'
        
        if not os.path.exists(model_path):
            print(" XLM-RoBERTa model files not found")
            return None
        
        # Load model and tokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Prepare test data
        predictions = []
        probabilities = []
        
        print("Making predictions...")
        for text in tqdm(X_test, desc="Processing texts"):
            inputs = tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                
                predictions.append(pred.cpu().item())
                probabilities.append(probs.cpu().numpy().flatten().tolist())
        
        # Calculate metrics
        metrics = {
            'model_name': 'XLM-RoBERTa',
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f" Accuracy: {metrics['accuracy']:.4f}")
        print(f" F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f" Error evaluating XLM-RoBERTa: {e}")
        return None

def create_comparison_visualization(all_metrics):
    """
    Create comparison visualization of all models
    
    Args:
        all_metrics: List of metrics dictionaries
    """
    print("\n Creating comparison visualization...")
    
    # Extract metrics for comparison
    model_names = [m['model_name'] for m in all_metrics if m is not None]
    accuracies = [m['accuracy'] for m in all_metrics if m is not None]
    precisions = [m['precision'] for m in all_metrics if m is not None]
    recalls = [m['recall'] for m in all_metrics if m is not None]
    f1_scores = [m['f1_score'] for m in all_metrics if m is not None]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics_data = {
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)])
        ax.set_title(f'{metric_name}', fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print(" Comparison visualization saved as 'model_comparison.png'")
    plt.show()

def create_confusion_matrices(all_metrics):
    """
    Create confusion matrices for all models
    
    Args:
        all_metrics: List of metrics dictionaries
    """
    print("\n Creating confusion matrices...")
    
    n_models = len([m for m in all_metrics if m is not None])
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    
    for idx, metrics in enumerate(all_metrics):
        if metrics is None:
            continue
            
        cm = np.array(metrics['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                   ax=axes[idx])
        axes[idx].set_title(f"{metrics['model_name']}\nAccuracy: {metrics['accuracy']:.3f}")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print(" Confusion matrices saved as 'confusion_matrices_comparison.png'")
    plt.show()

def save_metrics_to_json(all_metrics, output_path='metrics.json'):
    """
    Save all metrics to JSON file
    
    Args:
        all_metrics: List of metrics dictionaries
        output_path: Output file path
    """
    print(f"\n Saving metrics to {output_path}...")
    
    # Prepare data for JSON serialization
    metrics_data = {
        'evaluation_info': {
            'total_models': len([m for m in all_metrics if m is not None]),
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'test_size': 0.2,
                'random_state': 42
            }
        },
        'model_results': []
    }
    
    for metrics in all_metrics:
        if metrics is not None:
            # Remove predictions and probabilities for JSON (too large)
            clean_metrics = {k: v for k, v in metrics.items() 
                           if k not in ['predictions', 'probabilities']}
            metrics_data['model_results'].append(clean_metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f" Metrics saved to {output_path}")

def save_metrics_to_csv(all_metrics, output_path='metrics_summary.csv'):
    """
    Save metrics summary to CSV file
    
    Args:
        all_metrics: List of metrics dictionaries
        output_path: Output file path
    """
    print(f"\n Saving metrics summary to {output_path}...")
    
    # Create summary DataFrame
    summary_data = []
    for metrics in all_metrics:
        if metrics is not None:
            summary_data.append({
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score']
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
    
    print(f" Metrics summary saved to {output_path}")
    print("\n Summary Table:")
    print(df_summary.to_string(index=False, float_format='%.4f'))

def main():
    """
    Main evaluation function
    """
    print("="*80)
    print("UNIFIED MODEL EVALUATION SYSTEM")
    print("="*80)
    print("Evaluating all spam detection models on the same test set")
    print("="*80)
    
    # Load and split data
    X_train, X_test, y_train, y_test, df = load_and_split_data()
    
    if X_test is None:
        print(" Failed to load dataset. Exiting.")
        return
    
    # Evaluate all models
    all_metrics = []
    
    # Evaluate Logistic Regression
    lr_metrics = evaluate_logistic_regression(X_test, y_test)
    all_metrics.append(lr_metrics)
    
    # Evaluate LSTM
    lstm_metrics = evaluate_lstm(X_test, y_test)
    all_metrics.append(lstm_metrics)
    
    # Evaluate XLM-RoBERTa
    xlm_metrics = evaluate_xlm_roberta(X_test, y_test)
    all_metrics.append(xlm_metrics)
    
    # Filter out None results
    valid_metrics = [m for m in all_metrics if m is not None]
    
    if not valid_metrics:
        print(" No models could be evaluated. Please ensure all models are trained.")
        return
    
    # Create visualizations
    create_comparison_visualization(valid_metrics)
    create_confusion_matrices(valid_metrics)
    
    # Save metrics
    save_metrics_to_json(valid_metrics)
    save_metrics_to_csv(valid_metrics)
    
    # Print final summary
    print("\n" + "="*80)
    print(" FINAL EVALUATION SUMMARY")
    print("="*80)
    
    for metrics in valid_metrics:
        print(f"\n{metrics['model_name']}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Find best model
    best_model = max(valid_metrics, key=lambda x: x['f1_score'])
    print(f"\n Best Model (by F1-Score): {best_model['model_name']}")
    print(f"   F1-Score: {best_model['f1_score']:.4f}")
    
    print("\n Evaluation completed successfully!")
    print(" Files created:")
    print("    metrics.json (detailed metrics)")
    print("    metrics_summary.csv (summary table)")
    print("    model_comparison.png (performance comparison)")
    print("    confusion_matrices_comparison.png (confusion matrices)")

if __name__ == "__main__":
    main()
