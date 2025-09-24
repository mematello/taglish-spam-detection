# Taglish Spam Detection System 🇵🇭🇺🇸

A comprehensive spam detection system for English and Filipino (Taglish) SMS messages using multiple machine learning approaches.

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Logistic Regression** | 97.33% | 99.36% | 90.41% | 94.67% | 0.25s |
| **LSTM** | TBD | TBD | TBD | TBD | TBD |
| **XLM-RoBERTa** | TBD | TBD | TBD | TBD | TBD |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset
Place your `final_spam_ham_dataset.csv` in the `dataset/` folder with columns:
- `label`: 'ham' or 'spam'
- `text`: message content

### Training Models

#### 1. Logistic Regression
```bash
cd models/logistic_regression/
python train_model.py
python test_model.py
```

#### 2. LSTM (Coming Soon)
```bash
cd models/lstm/
python train_lstm.py
python test_lstm.py
```

#### 3. XLM-RoBERTa (Coming Soon)
```bash
cd models/xlm_roberta/
python train_xlm_roberta.py
python test_xlm_roberta.py
```

### Web Interface
```bash
cd web_ui/
python app.py
```
Visit `http://localhost:5000` to compare all models interactively.

## 📁 Project Structure

```
taglish-spam-detection/
├── models/
│   ├── logistic_regression/    # Traditional ML approach
│   ├── lstm/                   # Deep learning approach
│   └── xlm_roberta/           # Transformer-based approach
├── web_ui/                    # Interactive comparison interface
├── dataset/                   # Training data
└── results/                   # Performance comparisons
```

## 🎯 Features

### Logistic Regression Model
- ✅ Fast training (< 1 second)
- ✅ High precision (99.36%)
- ✅ TF-IDF vectorization with n-grams
- ✅ Interactive testing mode
- ✅ Batch processing support

### LSTM Model (In Development)
- 🔄 Sequential pattern learning
- 🔄 Word embedding support
- 🔄 Handles variable message lengths

### XLM-RoBERTa Model (In Development)
- 🔄 Multilingual transformer
- 🔄 Pre-trained on Filipino text
- 🔄 State-of-the-art performance

## 📊 Sample Results

### Test Messages
```
✅ "Hey, how are you doing today?" → HAM (96.1% confidence)
❌ "Congratulations! You won 50000 pesos!" → SPAM (91.3% confidence)
✅ "Tara, kita tayo bukas." → HAM (80.4% confidence)
❌ "Free GCash promo! Click here!" → SPAM (94.7% confidence)
```

## 🌐 Web Interface Features

- **Real-time Comparison**: Test messages across all three models
- **Performance Metrics**: Visual comparison of accuracy, speed, and confidence
- **Interactive Dashboard**: Charts and graphs showing model performance
- **Batch Testing**: Upload files for bulk message analysis

## 🔬 Technical Details

### Preprocessing
- Text normalization and cleaning
- Lowercasing and whitespace removal
- Support for Filipino diacritics

### Feature Engineering
- **Logistic Regression**: TF-IDF with 1-2 grams, 5000 features
- **LSTM**: Word embeddings, sequence padding
- **XLM-RoBERTa**: Transformer tokenization

## 📈 Performance Analysis

### Confusion Matrix (Logistic Regression)
```
           Predicted
         Ham    Spam
Ham      963       2
Spam      33     311
```

### Error Analysis
- **False Positive Rate**: 0.21% (very conservative)
- **False Negative Rate**: 9.59% (acceptable for spam detection)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Dataset Information

- **Total Messages**: 6,542
- **Ham Messages**: 4,825 (73.8%)
- **Spam Messages**: 1,717 (26.2%)
- **Languages**: English, Filipino, Taglish (mixed)

## 🏆 Team

- **Marcus Oliver**: Logistic Regression implementation
- **Dominic Vilog**: LSTM implementation  
- **Ian Placencia**: XLM-RoBERTa implementation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset contributors
- Filipino NLP community
- Open source ML libraries

## 🔮 Future Work

- [ ] Real-time SMS integration
- [ ] Mobile app development
- [ ] Additional Filipino dialects support
- [ ] Ensemble model combining all three approaches
- [ ] API deployment for production use

---

*Protecting Filipino messages from spam, one algorithm at a time! 🇵🇭*# taglish-spam-detection
