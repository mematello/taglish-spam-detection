import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import json
import os
import warnings
warnings.filterwarnings('ignore')

class SpamDetector:
    def __init__(self, model_path="saved_model"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_mapping = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model, tokenizer, and label mapping"""
        try:
            print("Loading model and tokenizer...")
            
            # Load tokenizer
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            label_path = os.path.join(self.model_path, "label_mapping.json")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    self.label_mapping = json.load(f)
            else:
                # Default mapping
                self.label_mapping = {
                    "id2label": {"0": "ham", "1": "spam"},
                    "label2id": {"ham": 0, "spam": 1}
                }
            
            print("Model loaded successfully!")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model has been trained and saved in the 'saved_model' directory.")
            raise
    
    def predict(self, text, max_length=512):
        """
        Predict whether the given text is spam or ham
        
        Args:
            text (str): Input text to classify
            max_length (int): Maximum sequence length for tokenization
            
        Returns:
            dict: Contains prediction, confidence, and probabilities for both classes
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "error": "Invalid input. Please provide a non-empty text string."
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                confidence_scores = probabilities.cpu().numpy()[0]
                
                # Get prediction
                predicted_class_id = torch.argmax(logits, dim=-1).item()
                predicted_label = self.label_mapping["id2label"][str(predicted_class_id)]
                
                # Get confidence for predicted class
                confidence = confidence_scores[predicted_class_id] * 100
                
                return {
                    "prediction": predicted_label,
                    "confidence": f"{confidence:.2f}%",
                    "probabilities": {
                        "ham": f"{confidence_scores[0] * 100:.2f}%",
                        "spam": f"{confidence_scores[1] * 100:.2f}%"
                    },
                    "text": text
                }
                
        except Exception as e:
            return {
                "error": f"Error during prediction: {e}"
            }

def main():
    """Interactive testing interface"""
    try:
        # Initialize the spam detector
        detector = SpamDetector()
        
        print("\n" + "="*60)
        print("XLM-RoBERTa Spam Detection Model - Testing Interface")
        print("="*60)
        print("Instructions:")
        print("- Enter text to classify as spam or ham")
        print("- Type 'quit', 'exit', or 'q' to stop")
        print("- The model supports both English and Tagalog text")
        print("="*60)
        
        while True:
            # Get user input
            user_input = input("\nEnter text to classify: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                print("Please enter some text to classify.")
                continue
            
            # Make prediction
            result = detector.predict(user_input)
            
            # Display results
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("\n" + "-"*40)
                print("CLASSIFICATION RESULTS")
                print("-"*40)
                print(f"Text: {result['text']}")
                print(f"Prediction: {result['prediction'].upper()}")
                print(f"Confidence: {result['confidence']}")
                print("\nClass Probabilities:")
                print(f"  Ham: {result['probabilities']['ham']}")
                print(f"  Spam: {result['probabilities']['spam']}")
                print("-"*40)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()