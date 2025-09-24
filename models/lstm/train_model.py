#!/usr/bin/env python3
"""
train_model.py - LSTM Spam Detection Training Script

This script trains an LSTM model for spam vs ham detection and saves all artifacts.
It includes data preprocessing, model training, evaluation, and artifact saving.

Author: AI Assistant
Date: 2024
"""

import os
import pandas as pd
import numpy as np
import re
import string
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Alternative import style to avoid Pylance warnings
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    # Fallback import style
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not present
def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

def check_gpu():
    """Check for GPU availability and configure TensorFlow"""
    print("Checking GPU availability...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU available: {len(gpus)} device(s) found")
            print("✓ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU found. Training will use CPU.")
    
    return len(gpus) > 0

def load_dataset(filepath='final_spam_ham_dataset.csv'):
    """
    Load the spam detection dataset
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        print(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['label', 'text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✓ Dataset loaded successfully!")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Display label distribution
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{filepath}' not found.")
        print("Please ensure the file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def clean_text(text, stop_words):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Input text to clean
        stop_words (set): Set of stopwords to remove
        
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
    
    # Remove phone numbers (basic pattern)
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
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    except:
        # Fallback if tokenization fails
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)

def preprocess_data(df):
    """
    Preprocess the dataset for training
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        tuple: (X, y, tokenizer, label_encoder)
    """
    print("\n" + "="*50)
    print("PREPROCESSING DATA")
    print("="*50)
    
    # Setup stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        print("Warning: Could not load stopwords. Using basic stopwords.")
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                     'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                     'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                     'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
    
    # Remove rows with missing data
    initial_size = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"Removed {initial_size - len(df)} rows with missing data")
    
    # Clean text data
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x, stop_words))
    
    # Remove empty texts after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"Final dataset size: {len(df)} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    
    print(f"Label encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} → {i}")
    
    print(f"Label distribution after encoding:")
    unique, counts = np.unique(y, return_counts=True)
    for label_idx, count in zip(unique, counts):
        original_label = label_encoder.inverse_transform([label_idx])[0]
        print(f"  {original_label} ({label_idx}): {count} samples ({count/len(y)*100:.1f}%)")
    
    # Tokenize text
    print("\nTokenizing text...")
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['cleaned_text'])
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    
    # Analyze sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Sequence length statistics:")
    print(f"  Mean: {np.mean(seq_lengths):.1f}")
    print(f"  Median: {np.median(seq_lengths):.1f}")
    print(f"  Max: {np.max(seq_lengths)}")
    print(f"  95th percentile: {np.percentile(seq_lengths, 95):.1f}")
    
    # Pad sequences (use 95th percentile as max length)
    max_length = min(int(np.percentile(seq_lengths, 95)), 100)
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    print(f"Using max sequence length: {max_length}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
    
    return X, y, tokenizer, label_encoder, max_length

def build_lstm_model(vocab_size, max_length, embedding_dim=128):
    """
    Build the LSTM model architecture
    
    Args:
        vocab_size (int): Size of vocabulary
        max_length (int): Maximum sequence length
        embedding_dim (int): Embedding layer dimension
        
    Returns:
        tensorflow.keras.models.Sequential: Compiled model
    """
    print("\n" + "="*50)
    print("BUILDING LSTM MODEL")
    print("="*50)
    
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_length,
                 name='embedding'),
        
        # Spatial dropout for regularization
        SpatialDropout1D(0.2, name='spatial_dropout'),
        
        # LSTM layer with dropout
        LSTM(units=64, 
             dropout=0.2, 
             recurrent_dropout=0.2, 
             return_sequences=False,
             name='lstm'),
        
        # Dense layers with dropout
        Dense(32, activation='relu', name='dense_1'),
        Dropout(0.5, name='dropout_1'),
        
        Dense(16, activation='relu', name='dense_2'),
        Dropout(0.3, name='dropout_2'),
        
        # Output layer for binary classification
        Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the LSTM model
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        History object
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            filepath='best_model_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # Train the model
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    print("Starting training...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        label_encoder: Label encoder for class names
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Print results
    print("Test Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - LSTM Spam Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    
    return metrics

def save_artifacts(model, tokenizer, label_encoder, metrics, max_length):
    """
    Save all model artifacts
    
    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        label_encoder: Fitted label encoder
        metrics: Model performance metrics
        max_length: Maximum sequence length used
    """
    print("\n" + "="*50)
    print("SAVING MODEL ARTIFACTS")
    print("="*50)
    
    # Save model
    model.save('lstm_spam_model.h5')
    print("✓ Model saved as 'lstm_spam_model.h5'")
    
    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("✓ Tokenizer saved as 'tokenizer.pkl'")
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✓ Label encoder saved as 'label_encoder.pkl'")
    
    # Save model configuration and metrics
    config = {
        'max_length': max_length,
        'vocab_size': len(tokenizer.word_index) + 1,
        'metrics': metrics,
        'model_version': '1.0'
    }
    
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    print("✓ Model configuration saved as 'model_config.pkl'")
    
    print("\nAll artifacts saved successfully!")
    print("Files created:")
    print("  - lstm_spam_model.h5 (trained model)")
    print("  - tokenizer.pkl (text tokenizer)")
    print("  - label_encoder.pkl (label encoder)")
    print("  - model_config.pkl (configuration)")
    print("  - confusion_matrix.png (evaluation plot)")

def check_existing_artifacts():
    """
    Check if model artifacts already exist
    
    Returns:
        bool: True if all artifacts exist
    """
    required_files = [
        'lstm_spam_model.h5',
        'tokenizer.pkl', 
        'label_encoder.pkl',
        'model_config.pkl'
    ]
    
    existing_files = [f for f in required_files if os.path.exists(f)]
    
    if len(existing_files) == len(required_files):
        print("✓ All model artifacts found:")
        for file in existing_files:
            size = os.path.getsize(file) / (1024*1024)  # Size in MB
            print(f"  - {file} ({size:.2f} MB)")
        return True
    elif len(existing_files) > 0:
        print("⚠ Some artifacts found, but incomplete set:")
        for file in existing_files:
            print(f"  ✓ {file}")
        missing = [f for f in required_files if f not in existing_files]
        for file in missing:
            print(f"  ❌ {file} (missing)")
        return False
    else:
        print("No existing model artifacts found.")
        return False

def main():
    """
    Main training function
    """
    print("="*60)
    print("🚀 LSTM SPAM DETECTION - TRAINING PIPELINE")
    print("="*60)
    
    # Setup
    setup_nltk()
    gpu_available = check_gpu()
    
    # Check for existing artifacts
    print("\n" + "="*50)
    print("CHECKING FOR EXISTING MODEL")
    print("="*50)
    
    if check_existing_artifacts():
        user_input = input("\nModel artifacts already exist. Retrain? (y/N): ").lower()
        if user_input not in ['y', 'yes']:
            print("✓ Using existing model. Run test_model.py to test predictions.")
            return
        else:
            print("🔄 Proceeding with retraining...")
    
    # Load dataset
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    
    df = load_dataset()
    if df is None:
        return
    
    # Preprocess data
    X, y, tokenizer, label_encoder, max_length = preprocess_data(df)
    
    # Split data
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Data split completed:")
    print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    model = build_lstm_model(vocab_size, max_length)
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save artifacts
    save_artifacts(model, tokenizer, label_encoder, metrics, max_length)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Next steps:")
    print("1. Run 'python test_model.py' to test predictions")
    print("2. Check 'confusion_matrix.png' for model performance visualization")
    print(f"3. Final model accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

if __name__ == "__main__":
    main()