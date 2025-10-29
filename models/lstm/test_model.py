#!/usr/bin/env python3
"""
test_model.py - LSTM Spam Detection Testing Script

This script loads the trained LSTM model and provides an interactive interface
for spam detection with percentage predictions.

Author: AI Assistant
Date: 2024
"""

import os
import pickle
import numpy as np
import pandas as pd
import re
import string
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SpamDetector:
    """
    Spam Detection class for loading and using the trained LSTM model
    """
    
    def __init__(self):
        """Initialize the SpamDetector"""
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = None
        self.config = None
        self.stop_words = None
        self.setup_nltk()
        
    def setup_nltk(self):
        """Setup NLTK data"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Warning: Could not load NLTK stopwords. Using basic set.")
            self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                              'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                              'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                              'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
    
    def load_artifacts(self):
        """
        Load all saved model artifacts
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Loading model artifacts...")
        
        # Check if all required files exist
        required_files = [
            'lstm_spam_model.h5',
            'tokenizer.pkl',
            'label_encoder.pkl', 
            'model_config.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("❌ Error: Missing required files:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nPlease run 'train_model.py' first to train the model.")
            return False
        
        try:
            # Load model
            print("  Loading LSTM model...")
            self.model = load_model('lstm_spam_model.h5')
            
            # Load tokenizer
            print("  Loading tokenizer...")
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load label encoder
            print("  Loading label encoder...")
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            # Load configuration
            print("  Loading model configuration...")
            with open('model_config.pkl', 'rb') as f:
                self.config = pickle.load(f)
                self.max_length = self.config['max_length']
            
            print("✅ All artifacts loaded successfully!")
            
            # Display model info
            print(f"\nModel Information:")
            print(f"  Model version: {self.config.get('model_version', 'Unknown')}")
            print(f"  Vocabulary size: {self.config.get('vocab_size', 'Unknown')}")
            print(f"  Max sequence length: {self.max_length}")
            print(f"  Classes: {list(self.label_encoder.classes_)}")
            
            if 'metrics' in self.config:
                metrics = self.config['metrics']
                print(f"  Training accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  Training F1-score: {metrics.get('f1_score', 0):.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            return False
    
    def clean_text(self, text):
        """
        Clean and preprocess text data (same as training)
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            return ' '.join(tokens)
        except:
            # Fallback if tokenization fails
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            return ' '.join(words)
    
    def predict_message(self, message):
        """
        Predict spam probability for a message
        
        Args:
            message (str): Input message to classify
            
        Returns:
            dict: Prediction results with probabilities
        """
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded. Please load artifacts first."}
        
        # Clean the message
        cleaned_message = self.clean_text(message)
        
        # Handle empty messages after cleaning
        if not cleaned_message.strip():
            return {
                'original_message': message,
                'cleaned_message': cleaned_message,
                'prediction': 'ham',
                'spam_probability': 0.0,
                'ham_probability': 100.0,
                'confidence': 'Low',
                'note': 'Message is empty after cleaning'
            }
        
        # Tokenize and pad sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_message])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, 
                                      padding='post', truncating='post')
        
        # Make prediction
        spam_prob = float(self.model.predict(padded_sequence, verbose=0)[0][0])
        ham_prob = 1.0 - spam_prob
        
        # Convert to percentages
        spam_percentage = spam_prob * 100
        ham_percentage = ham_prob * 100
        
        # Determine prediction
        prediction = self.label_encoder.inverse_transform([1 if spam_prob > 0.5 else 0])[0]
        
        # Determine confidence level
        max_prob = max(spam_prob, ham_prob)
        if max_prob >= 0.9:
            confidence = "Very High"
        elif max_prob >= 0.75:
            confidence = "High"
        elif max_prob >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            'original_message': message,
            'cleaned_message': cleaned_message,
            'prediction': prediction,
            'spam_probability': spam_percentage,
            'ham_probability': ham_percentage,
            'confidence': confidence
        }
    
    def interactive_mode(self):
        """
        Interactive mode for continuous message testing
        """
        print("\n" + "="*70)
        print("🔍 INTERACTIVE SPAM DETECTION SYSTEM")
        print("="*70)
        print("Enter messages to check if they are SPAM or HAM")
        print("Commands:")
        print("  - Type any message to analyze")
        print("  - Type 'sample' to test with sample messages")
        print("  - Type 'quit', 'exit', or 'q' to stop")
        print("="*70)
        
        while True:
            try:
                # Get user input
                message = input("\n📝 Enter message: ").strip()
                
                # Handle commands
                if message.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using the Spam Detection System!")
                    break
                
                if message.lower() == 'sample':
                    self.test_sample_messages()
                    continue
                
                if not message:
                    print("⚠️ Please enter a message or type 'quit' to exit.")
                    continue
                
                # Make prediction
                result = self.predict_message(message)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
                    continue
                
                # Display results
                self.display_prediction_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Spam Detection System!")
                break
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                continue
    
    def display_prediction_result(self, result):
        """
        Display prediction results in a formatted way
        
        Args:
            result (dict): Prediction result dictionary
        """
        print("\n" + "-"*60)
        print("📊 ANALYSIS RESULTS")
        print("-"*60)
        
        # Show original and cleaned message
        print(f"Original: '{result['original_message']}'")
        if result['cleaned_message'] != result['original_message']:
            print(f"Cleaned:  '{result['cleaned_message']}'")
        
        # Main prediction
        prediction_emoji = "🚨" if result['prediction'] == 'spam' else "✅"
        print(f"\n{prediction_emoji} PREDICTION: {result['prediction'].upper()}")
        print(f"🔒 Confidence: {result['confidence']}")
        
        # Probabilities
        print(f"\n📈 PROBABILITIES:")
        print(f"🚨 Spam: {result['spam_probability']:.2f}%")
        print(f"✅ Ham:  {result['ham_probability']:.2f}%")
        
        # Visual bar representation
        spam_bar_length = int(result['spam_probability'] // 5)
        ham_bar_length = int(result['ham_probability'] // 5)
        
        spam_bar = "█" * spam_bar_length + "░" * (20 - spam_bar_length)
        ham_bar = "█" * ham_bar_length + "░" * (20 - ham_bar_length)
        
        print(f"\n📊 Visual Representation:")
        print(f"Spam: [{spam_bar}] {result['spam_probability']:.1f}%")
        print(f"Ham:  [{ham_bar}] {result['ham_probability']:.1f}%")
        
        # Risk assessment
        spam_prob = result['spam_probability']
        if spam_prob >= 80:
            print("⚠️  HIGH SPAM RISK - Be very cautious!")
        elif spam_prob >= 60:
            print("⚠️  MODERATE SPAM RISK - Exercise caution")
        elif spam_prob >= 40:
            print("⚠️  LOW SPAM RISK - Probably safe")
        else:
            print("✅ LIKELY LEGITIMATE MESSAGE")
        
        # Show note if present
        if 'note' in result:
            print(f"📝 Note: {result['note']}")
        
        print("-"*60)
    
    def test_sample_messages(self):
        """
        Test the model with predefined sample messages
        """
        print("\n" + "="*60)
        print("🧪 TESTING WITH SAMPLE MESSAGES")
        print("="*60)
        
        sample_messages = [
            "Congratulations! You won $1000! Call now to claim your prize!",
            "Hey John, are we still meeting for lunch tomorrow at 1pm?",
            "FREE MONEY!!! No credit check needed! Apply now!",
            "Your meeting has been rescheduled to 3pm. Please confirm.",
            "URGENT: Your account will be suspended. Click here immediately!",
            "Thanks for the great presentation today. Well done!",
            "Limited time offer! 90% discount on all products. Buy now!",
            "Can you pick up some milk on your way home?",
            "You have been selected for a special promotion. Call 555-0123",
            "Don't forget about the team meeting at 10am tomorrow."
        ]
        
        for i, message in enumerate(sample_messages, 1):
            print(f"\n--- Sample {i} ---")
            result = self.predict_message(message)
            
            # Simplified display for samples
            prediction_emoji = "🚨" if result['prediction'] == 'spam' else "✅"
            print(f"Message: '{message}'")
            print(f"Result: {prediction_emoji} {result['prediction'].upper()} "
                  f"(Spam: {result['spam_probability']:.1f}%, Ham: {result['ham_probability']:.1f}%)")
        
        print("\n" + "="*60)
    
    def single_prediction(self, message):
        """
        Make a single prediction (for programmatic use)
        
        Args:
            message (str): Message to classify
            
        Returns:
            dict: Prediction result
        """
        return self.predict_message(message)

def test_specific_message(detector, message):
    """
    Test a specific message and display results
    
    Args:
        detector: SpamDetector instance
        message (str): Message to test
    """
    print(f"\nTesting message: '{message}'")
    result = detector.predict_message(message)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    detector.display_prediction_result(result)

def main():
    """
    Main function for testing the spam detection model
    """
    print("="*70)
    print("🧪 LSTM SPAM DETECTION - TESTING SYSTEM")
    print("="*70)
    
    # Initialize detector
    detector = SpamDetector()
    
    # Load model artifacts
    if not detector.load_artifacts():
        print("\n❌ Failed to load model artifacts.")
        print("Please run 'train_model.py' first to train the model.")
        return
    
    print("\n" + "="*50)
    print("MODEL READY FOR TESTING")
    print("="*50)
    
    # Check if user provided a command line argument for single prediction
    import sys
    if len(sys.argv) > 1:
        # Single message mode
        message = ' '.join(sys.argv[1:])
        test_specific_message(detector, message)
        return
    
    # Interactive mode selection
    print("\nChoose testing mode:")
    print("1. Interactive mode (continuous testing)")
    print("2. Sample messages test")
    print("3. Single message test")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                detector.interactive_mode()
                break
            elif choice == '2':
                detector.test_sample_messages()
                
                # Ask if user wants to continue
                continue_choice = input("\nWould you like to try another mode? (y/N): ").lower()
                if continue_choice not in ['y', 'yes']:
                    break
            elif choice == '3':
                message = input("\nEnter the message to test: ").strip()
                if message:
                    test_specific_message(detector, message)
                else:
                    print("No message provided.")
                
                # Ask if user wants to continue
                continue_choice = input("\nWould you like to try another mode? (y/N): ").lower()
                if continue_choice not in ['y', 'yes']:
                    break
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Example usage functions for programmatic access
def predict_single_message(message):
    """
    Convenience function for single message prediction
    
    Args:
        message (str): Message to classify
        
    Returns:
        dict: Prediction result or None if error
    """
    detector = SpamDetector()
    if detector.load_artifacts():
        return detector.predict_message(message)
    return None

def batch_predict(messages):
    """
    Predict multiple messages at once
    
    Args:
        messages (list): List of messages to classify
        
    Returns:
        list: List of prediction results
    """
    detector = SpamDetector()
    if not detector.load_artifacts():
        return None
    
    results = []
    for message in messages:
        result = detector.predict_message(message)
        results.append(result)
    
    return results

if __name__ == "__main__":
    main()