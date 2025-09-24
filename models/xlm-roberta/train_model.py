import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    print("Loading dataset...")
    
    # Load the CSV file
    try:
        df = pd.read_csv('final_spam_ham_dataset.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'final_spam_ham_dataset.csv' not found in current directory.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check dataset structure
    if 'label' not in df.columns or 'text' not in df.columns:
        print("Error: Dataset must contain 'label' and 'text' columns.")
        return
    
    # Clean the dataset
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    
    print(f"Dataset info:")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Encode labels: spam=1, ham=0
    label_mapping = {'spam': 1, 'ham': 0}
    df['encoded_label'] = df['label'].map(label_mapping)
    
    # Check if all labels were mapped correctly
    if df['encoded_label'].isnull().any():
        print("Warning: Some labels could not be mapped. Checking unique labels...")
        print("Unique labels:", df['label'].unique())
        # Handle case variations
        df['label'] = df['label'].str.lower().str.strip()
        df['encoded_label'] = df['label'].map(label_mapping)
        
        if df['encoded_label'].isnull().any():
            print("Error: Invalid labels found. Expected 'spam' or 'ham' only.")
            return
    
    # Split the dataset
    X = df['text'].tolist()
    y = df['encoded_label'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize tokenizer and model
    print("Loading XLM-RoBERTa tokenizer and model...")
    
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1}
    )
    
    # Create datasets
    train_dataset = SpamDataset(X_train, y_train, tokenizer)
    test_dataset = SpamDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Make predictions on test set for detailed metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # Save the model and tokenizer
    save_directory = "saved_model"
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Saving model and tokenizer to {save_directory}...")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # Save label mapping
    import json
    label_info = {
        "id2label": {0: "ham", 1: "spam"},
        "label2id": {"ham": 0, "spam": 1}
    }
    
    with open(os.path.join(save_directory, "label_mapping.json"), "w") as f:
        json.dump(label_info, f)
    
    print("Training completed successfully!")
    print(f"Model saved in: {save_directory}")
    print(f"Final accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Final F1-score: {eval_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()