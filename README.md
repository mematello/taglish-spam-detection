# Taglish Spam Detection System 🇵🇭🇺🇸

A comprehensive spam detection system for English and Filipino (Taglish) SMS messages using multiple machine learning approaches: Logistic Regression (TF‑IDF), LSTM, and XLM‑RoBERTa.

## 📊 Compare Models

After training, run the unified evaluator to generate metrics and plots:
```bash
python evaluate_models.py
```
Outputs created in the project root:
- `metrics.json`, `metrics_summary.csv`
- `model_comparison.png`, `confusion_matrices_comparison.png`

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset
Place `dataset/final_spam_ham_dataset.csv` with columns:
- `label`: 'ham' or 'spam'
- `text`: message content

### Train and Test Models

#### 1) Logistic Regression (TF‑IDF)
```bash
cd models/logistic_regression
python train_model.py   # trains and saves to model_files/
python test_model.py    # interactive tester (uses saved model_files)
```
Artifacts: `models/logistic_regression/model_files/`

#### 2) LSTM
```bash
cd models/lstm
python train_model.py   # saves artifacts to model_files/

# Run tester from the artifacts folder so it can find the files
cd model_files
python ../test_model.py
```
Artifacts: `models/lstm/model_files/` (includes `lstm_spam_model.h5`, `tokenizer.pkl`, `label_encoder.pkl`, `model_config.pkl`)

#### 3) XLM‑RoBERTa
```bash
cd models/xlm-roberta
python train_model.py   # trains and saves to saved_model/
python test_model.py    # interactive tester
```
Artifacts: `models/xlm-roberta/saved_model/`

### Web Interface
```bash
cd web_ui
python app.py
```
Visit `http://localhost:5000`.

## 📁 Project Structure

```
taglish-spam-detection/
├── models/
│   ├── logistic_regression/      # Traditional ML (TF‑IDF + LR)
│   ├── lstm/                     # Deep learning sequence model
│   └── xlm-roberta/              # Transformer-based model
├── dataset/                      # Training data CSV
├── web_ui/                       # Simple web app for testing
├── evaluate_models.py            # Unified evaluation/plots
├── metrics.json | metrics_summary.csv | *.png
```

## 🎯 Features

### Logistic Regression
- Fast training, TF‑IDF with 1–2 grams, ~5000 features
- Interactive testing and batch file testing

### LSTM
- Tokenization, padding, embeddings, dropout regularization
- Saves all artifacts for reproducible inference

### XLM‑RoBERTa
- Multilingual transformer (English/Filipino)
- Robust probabilities with softmax outputs

## 🔬 Technical Details

### Preprocessing
- Lowercasing, punctuation/URL/number removal, whitespace cleanup
- LSTM uses NLTK (downloads required data at first run)

### Evaluation
- Consistent train/test split
- Accuracy, Precision, Recall, F1
- Confusion matrices and model comparison plots

## 🧪 Quick Test Examples

After training LR:
```bash
cd models/logistic_regression
python test_model.py
```

After training LSTM:
```bash
cd models/lstm/model_files
python ../test_model.py
```

After training XLM‑RoBERTa:
```bash
cd models/xlm-roberta
python test_model.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m "Add some AmazingFeature"`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Dataset Info

- Languages: English, Filipino, Taglish
- Labels: `ham`, `spam`

## 📄 License

MIT License — see `LICENSE`.

## 🔮 Future Work

- [ ] Real-time SMS integration
- [ ] Mobile app
- [ ] Additional Filipino dialects
- [ ] Ensemble of all models
- [ ] Production API deployment

---

Protecting Filipino messages from spam, one algorithm at a time. 🇵🇭
