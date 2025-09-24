#!/usr/bin/env python3
"""
Taglish Spam Detection System - Testing Module
==============================================
Test the trained spam detection model for English and Filipino (Taglish) SMS messages.

Author: Claude
Date: September 2025
"""

import pandas as pd
import numpy as np
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
    Testing module - loads pre-trained model.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
    
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
    
    def load_model(self, model_path: str = 'model_files/logistic_regression_taglish_spam_model.pkl', 
                   vectorizer_path: str = 'model_files/tfidf_vectorizer_taglish_spam_model.pkl') -> bool:
        """
        Load trained model and vectorizer from disk.
        
        Args:
            model_path: Path to load the model from
            vectorizer_path: Path to load the vectorizer from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            print(f"✅ Model loaded from: {model_path}")
            print(f"✅ Vectorizer loaded from: {vectorizer_path}")
            return True
        else:
            print(f"❌ Model files not found:")
            print(f"   • {model_path}")
            print(f"   • {vectorizer_path}")
            print("Please run train_model.py first to train the model.")
            return False
    
    def predict_message(self, message: str) -> Tuple[int, float]:
        """
        Predict if a single message is spam.
        
        Args:
            message: The message to classify
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please load the model first.")
        
        # Preprocess message
        processed_message = self.preprocess_text(message)
        
        # Vectorize
        message_tfidf = self.vectorizer.transform([processed_message])
        
        # Predict
        prediction = self.model.predict(message_tfidf)[0]
        confidence = self.model.predict_proba(message_tfidf)[0].max()
        
        return prediction, confidence
    
    def test_sample_messages(self):
        """
        Test the model with predefined sample messages.
        """
        print("\n" + "="*60)
        print("🧪 TESTING SAMPLE MESSAGES")
        print("="*60)
        
        # Test messages covering various scenarios
        test_messages = [
            # English spam
            "Claim your ₱1000 load now!",
            "Congratulations! You won 50000 pesos! Text WIN to 2346.",
            "Free GCash promo! Click here to claim your prize!",
            "URGENT: Your account will be suspended. Click link to verify.",
            
            # Filipino spam
            "Congrats! Nanalo ka ng ₱1,000 GCash load! Click this link now to claim.",
            "Mag-avail ng murang pautang! Walang collateral. Apply na ngayon!",
            "You receive P5,000 from DSWD. Confirm today or forfeited bukas.",
            
            # English legitimate
            "Hey, how are you doing today?",
            "Meeting is at 3pm tomorrow in conference room A.",
            "Thanks for the dinner last night, it was great!",
            
            # Filipino legitimate  
            "Tara, kita tayo bukas.",
            "Kumusta? Anong oras tayo magkikita bukas?",
            "Tapos na klase namin, pauwi na ako.",
            "Pakisabi kay mama na late ako uuwi.",
            "Bro, kita tayo sa Starbucks later? Libre na lang kita kape.",
            
            # Mixed/Taglish
            "Pre, may meeting tayo bukas at 2pm sa office ha.",
            "Sorry late ako, traffic sa EDSA. Wait lang."
        ]
        
        print("Testing various message types:\n")
        
        for i, message in enumerate(test_messages, 1):
            prediction, confidence = self.predict_message(message)
            label = "🚫 SPAM" if prediction == 1 else "📧 HAM "
            confidence_bar = "█" * int(confidence * 10)
            
            print(f"{i:2d}. 📱 '{message}'")
            print(f"    🔍 {label} (confidence: {confidence:.3f}) {confidence_bar}")
            print()
    
    def batch_test_messages(self, messages: list) -> list:
        """
        Test multiple messages at once.
        
        Args:
            messages: List of messages to test
            
        Returns:
            List of tuples (message, prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please load the model first.")
        
        results = []
        for message in messages:
            prediction, confidence = self.predict_message(message)
            results.append((message, prediction, confidence))
        
        return results
    
    def interactive_testing(self):
        """
        Interactive testing mode where user can input messages.
        """
        print("\n" + "="*60)
        print("💬 INTERACTIVE TESTING MODE")
        print("="*60)
        print("Enter messages to test (type 'quit', 'exit', or 'q' to stop):")
        print("You can also type 'sample' to run sample tests again.")
        
        while True:
            try:
                user_input = input("\n📱 Enter message: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q', '']:
                    break
                
                if user_input.lower() == 'sample':
                    self.test_sample_messages()
                    continue
                
                if user_input:
                    prediction, confidence = self.predict_message(user_input)
                    label = "🚫 SPAM" if prediction == 1 else "📧 HAM "
                    confidence_bar = "█" * int(confidence * 10)
                    
                    print(f"🔍 Result: {label} (confidence: {confidence:.3f}) {confidence_bar}")
                    
                    # Provide additional context
                    if prediction == 1:
                        print("   ⚠️  This message appears to be spam. Be cautious!")
                    else:
                        print("   ✅ This message appears to be legitimate.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_file(self, filepath: str):
        """
        Test messages from a file.
        
        Args:
            filepath: Path to file containing messages (one per line)
        """
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        print(f"\n📂 Testing messages from: {filepath}")
        print("="*60)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                messages = [line.strip() for line in f if line.strip()]
            
            if not messages:
                print("❌ No messages found in file.")
                return
            
            results = self.batch_test_messages(messages)
            
            spam_count = 0
            ham_count = 0
            
            for i, (message, prediction, confidence) in enumerate(results, 1):
                label = "🚫 SPAM" if prediction == 1 else "📧 HAM "
                confidence_bar = "█" * int(confidence * 10)
                
                print(f"{i:3d}. {label} | {message[:50]}{'...' if len(message) > 50 else ''}")
                print(f"      Confidence: {confidence:.3f} {confidence_bar}")
                
                if prediction == 1:
                    spam_count += 1
                else:
                    ham_count += 1
            
            print(f"\n📊 SUMMARY:")
            print(f"Total messages: {len(messages)}")
            print(f"🚫 Spam detected: {spam_count} ({spam_count/len(messages)*100:.1f}%)")
            print(f"📧 Ham detected: {ham_count} ({ham_count/len(messages)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")


def main():
    """
    Main testing function.
    """
    print("🚀 TAGLISH SPAM DETECTION SYSTEM - TESTING MODULE")
    print("=" * 60)
    print("📱 Testing spam detection for English and Filipino messages")
    print("=" * 60)
    
    # Initialize detector
    detector = TaglishSpamDetector()
    
    # Load trained model
    if not detector.load_model():
        print("\n❌ Cannot proceed without trained model.")
        print("🎯 Please run 'python train_model.py' first to train the model.")
        return
    
    print("\n🎯 Model loaded successfully! Ready for testing.")
    
    # Show menu options
    while True:
        print("\n" + "="*60)
        print("📋 TESTING OPTIONS")
        print("="*60)
        print("1. 🧪 Test with sample messages")
        print("2. 💬 Interactive testing mode")
        print("3. 📂 Test messages from file")
        print("4. 🚪 Exit")
        print("="*60)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            detector.test_sample_messages()
        
        elif choice == '2':
            detector.interactive_testing()
        
        elif choice == '3':
            filepath = input("Enter path to messages file: ").strip()
            if filepath:
                detector.test_file(filepath)
        
        elif choice == '4':
            break
        
        else:
            print("❌ Invalid choice. Please select 1-4.")
    
    print("\n👋 Thanks for using the Taglish Spam Detection System!")
    print("🇵🇭🇺🇸 Stay safe from spam in English and Filipino!")


if __name__ == "__main__":
    main()