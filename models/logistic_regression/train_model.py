#!/usr/bin/env python3
"""
Taglish Spam Detection System - Training Module
===============================================
Train a spam detection model for English and Filipino (Taglish) SMS messages.

Author: Claude
Date: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import re
from typing import Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TaglishSpamDetector:
    """
    A spam detection system for Taglish (English + Filipino) SMS messages.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load the merged spam/ham dataset.
        
        Args:
            filepath: Path to the CSV file with 'label' and 'text' columns
            
        Returns:
            DataFrame with 'message' and 'label' columns (0=ham, 1=spam)
        """
        print("Loading Taglish spam/ham dataset...")
        
        try:
            df = pd.read_csv(filepath)
            
            # Check for required columns
            if 'label' not in df.columns or 'text' not in df.columns:
                raise ValueError("Expected 'label' and 'text' columns in dataset")
            
            # Create standardized dataframe
            result_df = pd.DataFrame()
            result_df['message'] = df['text']
            
            # Convert labels to numeric (0=ham, 1=spam)
            result_df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            
            # Remove any rows with NaN values
            result_df = result_df.dropna()
            
            # Remove any rows with invalid labels
            result_df = result_df[result_df['label'].isin([0, 1])]
            
            print(f"Dataset loaded successfully: {len(result_df)} messages")
            print(f"   Ham (legitimate): {sum(result_df['label'] == 0)}")
            print(f"   Spam: {sum(result_df['label'] == 1)}")
            
            # Calculate class distribution
            spam_ratio = sum(result_df['label'] == 1) / len(result_df) * 100
            print(f"   Spam ratio: {spam_ratio:.1f}%")
            
            return result_df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame(columns=['message', 'label'])
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing.
        
        Args:
            text: Raw text message
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the Taglish spam detection model.
        
        Args:
            df: Dataset with 'message' and 'label' columns
            
        Returns:
            Dictionary with training results
        """
        print("Training Taglish spam detection model...")
        
        # Preprocess messages
        df['message'] = df['message'].apply(self.preprocess_text)
        
        # Prepare features and labels
        X = df['message']
        y = df['label']
        
        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Data split:")
        print(f"   Training: {len(X_train)} messages")
        print(f"   Testing: {len(X_test)} messages")
        
        # Initialize TF-IDF Vectorizer
        print("Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),         # Unigrams and bigrams
            max_features=5000,          # Limit to top 5000 features
            stop_words='english',       # Remove English stop words
            lowercase=True,             # Convert to lowercase
            strip_accents='unicode',    # Handle accented characters
            min_df=2,                   # Ignore terms that appear in less than 2 documents
            max_df=0.95                 # Ignore terms that appear in more than 95% of documents
        )
        
        # Transform training data
        print("Vectorizing messages...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train Logistic Regression model
        print("Training Logistic Regression classifier...")
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,                      # Regularization parameter
            solver='liblinear'          # Good for small datasets
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Make predictions on test set
        print("Evaluating model performance...")
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_count': X_train_tfidf.shape[1]
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, Any]):
        """
        Print detailed evaluation results.
        
        Args:
            results: Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print("TAGLISH SPAM DETECTION MODEL EVALUATION")
        print("="*60)
        
        print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        print(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
        print(f"F1-Score:  {results['f1']:.4f} ({results['f1']*100:.2f}%)")
        print(f"Features:  {results['feature_count']} TF-IDF features")
        
        print("\nConfusion Matrix:")
        print("           Predicted")
        print("         Ham    Spam")
        cm = results['confusion_matrix']
        print(f"Ham    {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"Spam   {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        print(f"\nDetailed Results:")
        print(f"True Negatives (Ham -> Ham):   {cm[0,0]}")
        print(f"False Positives (Ham -> Spam): {cm[0,1]}")
        print(f"False Negatives (Spam -> Ham): {cm[1,0]}")
        print(f"True Positives (Spam -> Spam): {cm[1,1]}")
        
        # Calculate error rates
        total = cm.sum()
        false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1]) * 100
        false_negative_rate = cm[1,0] / (cm[1,0] + cm[1,1]) * 100
        
        print(f"\nError Analysis:")
        print(f"False Positive Rate: {false_positive_rate:.2f}% (legitimate messages marked as spam)")
        print(f"False Negative Rate: {false_negative_rate:.2f}% (spam messages not caught)")
    
    def save_model(self, model_path: str = 'model_files/logistic_regression_taglish_spam_model.pkl', 
                   vectorizer_path: str = 'model_files/tfidf_vectorizer_taglish_spam_model.pkl'):
        """
        Save trained model and vectorizer to disk.
        
        Args:
            model_path: Path to save the model
            vectorizer_path: Path to save the vectorizer
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Cannot save untrained model.")
        
        # Create model_files directory if it doesn't exist
        os.makedirs('model_files', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"\nModel saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")


def main():
    """
    Main training function.
    """
    print("TAGLISH SPAM DETECTION SYSTEM - TRAINING MODULE")
    print("=" * 60)
    print("Training spam detection model for English and Filipino messages")
    print("=" * 60)
    
    # Initialize detector
    detector = TaglishSpamDetector()
    
    # Dataset path (adjusted for new folder structure)
    dataset_path = '../../dataset/final_spam_ham_dataset.csv'
    
    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Please ensure the final_spam_ham_dataset.csv file exists in the dataset/ folder.")
        return
    
    # Load dataset
    df = detector.load_dataset(dataset_path)
    
    # Check if dataset was loaded successfully
    if len(df) == 0:
        print("No data loaded. Please check your dataset file format.")
        print("Expected CSV with 'label' and 'text' columns.")
        return
    
    # Train model
    results = detector.train_model(df)
    
    # Print evaluation results
    detector.print_evaluation_results(results)
    
    # Save model for future use
    detector.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Model files saved:")
    print("   - model_files/logistic_regression_taglish_spam_model.pkl")
    print("   - model_files/tfidf_vectorizer_taglish_spam_model.pkl")
    print("\nYou can now run test_model.py to test the trained model!")
    print("Model ready to protect messages in English and Filipino!")


if __name__ == "__main__":
    main()