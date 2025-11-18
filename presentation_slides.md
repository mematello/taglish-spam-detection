## TAGLISH-SPAM_DETECTION – Presentation Notes

This document captures the slide contents for the Taglish spam detection project, focusing on:

- Discussion of the unprepared and prepared datasets  
- Explanation of the codes used for data preparation, model training, and testing  
- Presentation of results using visual aids (charts/graphs) based on `@presentation_viz`  
- Summary of findings and recommendations  

All content is based strictly on the current `taglish-spam-detection` project.

---

## Unprepared vs Prepared Datasets

### Unprepared (Raw) Dataset

- **Sources**
  - `dataset/unprepared/english_spam_dataset.csv`
  - `dataset/unprepared/filipino_spam_dataset.csv`

- **Provenance**
  - English spam collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
    - Columns: `v1`, `v2`, `,,` (label, text, unused placeholders)
  - Filipino spam collection: https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages  
    - Columns: `masked_celphone_number`, `hashed_celphone_number`, `date`, `text`, `carrier`

- **Format and language**
  - English source: classic SMS spam/ham mix (mostly English with some slang).
  - Filipino source: promotional and phishing-style SMS in Filipino/Taglish, labeled spam-only.

- **Issues observed in the raw data**
  - **Missing or invalid values**
    - Some rows have empty or null `text` or `label` values.
  - **Inconsistent labels**
    - Case and whitespace differences such as `"Spam"`, `"SPAM"`, `"spam "`.
  - **Noise in the text**
    - Embedded URLs (`http`, `https`, `www`), email addresses, and phone numbers.
    - Numbers for amounts and dates (e.g., `1000`, `12/25`).
    - Punctuation, emojis, and special characters.
  - **Class imbalance**
    - As in most SMS spam datasets, **ham messages are more frequent than spam**, which affects naive metrics like accuracy.

### Prepared Dataset (Project-Wide)

Across the pipeline, several common steps are applied before the data reaches each model:

- **Global label cleanup**
  - Standardize labels via lowercasing and stripping whitespace.
  - Map consistently: **`ham → 0`**, **`spam → 1`** in all training and evaluation scripts (`evaluate_models.py` and model-specific training files).

- **Handling missing data**
  - Rows with missing or invalid `text` or `label` are dropped.
  - Text fields are coerced to `str` to avoid type issues during tokenization/vectorization.

### Prepared Dataset for Logistic Regression

- **Text normalization**
  - Lowercase the message text.
  - Collapse multiple spaces to a single space and strip leading/trailing whitespace.

- **Feature representation**
  - Use **TF‑IDF vectorization** over the cleaned text to convert each message into a numeric feature vector.
  - TF‑IDF captures term importance across the corpus and is suitable for short SMS-style messages.

- **Train-test split**
  - The logistic regression model’s evaluation within `evaluate_models.py` uses a **stratified 80/20 split**, shared across all models for fair comparison.

### Prepared Dataset for LSTM

- **Text cleaning (`models/lstm/train_model.py`)**
  - Lowercase the text.
  - Remove:
    - URLs (`http\S+`, `www\.\S+`, `https\S+`)
    - Email addresses
    - Phone number patterns
    - All digits
    - Punctuation
  - Collapse multiple whitespace characters into a single space.

- **Tokenization and sequence construction**
  - Use Keras `Tokenizer(num_words=10000, oov_token='<OOV>')` to build a word index from the cleaned text.
  - Convert each message to a sequence of integer word indices.
  - Analyze sequence lengths (mean, median, max, 95th percentile) and set `max_length` to the 95th percentile capped at 100.
  - Apply `pad_sequences` to obtain fixed-length sequences of size `max_length`.

- **Label encoding**
  - Apply `LabelEncoder` to map text labels (`ham`, `spam`) to integer targets.

### Prepared Dataset for XLM-RoBERTa

- **Tokenizer-driven preparation**
  - Use HuggingFace `XLMRobertaTokenizer` / `AutoTokenizer` from `models/xlm-roberta/saved_model`.
  - For each message:
    - Tokenize with subword units.
    - Apply truncation and padding up to `max_length=512` tokens.
  - No aggressive manual cleaning is required; the multilingual transformer is trained to cope with noisy text.

---

## Code Explanation – Data Preparation, Training, and Testing

### Logistic Regression + TF‑IDF

- **Training & preparation file**: `models/logistic_regression/train_model.py`

- **Key steps**
  - Load dataset from `dataset/final_spam_ham_dataset.csv`.
  - Validate presence of required columns (`label`, `text`).
  - Clean and normalize text (lowercase, trim, compress whitespace).
  - Encode labels to 0 (ham) and 1 (spam).
  - Split data into train and test sets (stratified split).
  - Fit a **TF‑IDF vectorizer** on training text and transform both splits.
  - Train a **Logistic Regression** classifier on the TF‑IDF features.
  - Evaluate using scikit-learn:
    - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, and `classification_report`.
  - Save artifacts for deployment:
    - `logistic_regression_taglish_spam_model.pkl`
    - `tfidf_vectorizer_taglish_spam_model.pkl`

### LSTM (RNN)

- **Training & preparation file**: `models/lstm/train_model.py`

- **Data preparation**
  - `load_dataset()` reads the CSV, ensures required columns, and reports label distribution.
  - `preprocess_data()`:
    - Cleans text (removes URLs, emails, phone numbers, digits, punctuation, excess whitespace).
    - Uses NLTK stopwords and tokenization for further filtering.
    - Encodes labels with `LabelEncoder` and prints mapping and distribution.
    - Fits a Keras `Tokenizer`, converts text to integer sequences, and calculates sequence length statistics.
    - Pads/truncates sequences to a chosen `max_length` (based on 95th percentile, capped at 100).

- **Model architecture and training**
  - `build_lstm_model()`:
    - Embedding layer: `Embedding(vocab_size, 128, input_length=max_length)`.
    - Regularization: `SpatialDropout1D(0.2)`.
    - Recurrent layer: `LSTM(64 units, dropout=0.2, recurrent_dropout=0.2)`.
    - Dense layers: 32- and 16-unit ReLU layers with dropout.
    - Output: Dense(1, Sigmoid) for spam probability.
  - `train_model()`:
    - Optimizer: `Adam(learning_rate=0.001)`; loss: `binary_crossentropy`; metric: `accuracy`.
    - `EarlyStopping` on validation accuracy (with patience and restore-best-weights).
    - `ModelCheckpoint` saving best model weights to `best_model_checkpoint.h5`.

- **Evaluation and artifacts**
  - `evaluate_model()`:
    - Predicts probabilities on test set; thresholds at 0.5 during training evaluation.
    - Computes standard metrics (accuracy, precision, recall, F1) and confusion matrix.
    - Creates and saves confusion matrix plot.
  - `save_artifacts()`:
    - Saves the trained model (`lstm_spam_model.h5`), tokenizer, label encoder, and `model_config.pkl` (with max_length, vocab_size, and metrics) in `models/lstm/model_files/`.

- **Deployment-specific runtime**
  - In `web_ui/app.py`, `LSTMModel` reimplements the LSTM forward pass in pure NumPy:
    - Loads embeddings and LSTM/dense weights from the H5 file using `h5py`.
    - Performs manual LSTM cell computations and dense-layer forward passes.
    - Uses a tuned decision threshold (from `thresholds.json`) instead of a hard-coded 0.5.

### XLM-RoBERTa (Transformer)

- **Training & preparation file**: `models/xlm-roberta/train_model.py`

- **Key steps**
  - Load dataset and perform the same label encoding (0 for ham, 1 for spam).
  - Use `XLMRobertaTokenizer` or `AutoTokenizer` to encode texts:
    - `tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')`.
  - Fine-tune `XLMRobertaForSequenceClassification` on the training set for binary classification.
  - Evaluate on the common test split with standard metrics (accuracy, precision, recall, F1).
  - Save the model and tokenizer under `models/xlm-roberta/saved_model/`, including label mapping and metrics.

- **Deployment wrapper**
  - `XLMRobertaModel` in `web_ui/app.py`:
    - Loads model and tokenizer from `saved_model`.
    - Uses the label mapping (`label2id` or `id2label`) to map logits to spam/ham indices.
    - Applies softmax to logits, extracts spam and ham probabilities, and applies a tuned spam threshold from `thresholds.json`.

### Unified Evaluation and Testing

- **File**: `evaluate_models.py`

- **Common evaluation pipeline**
  - `load_and_split_data()`:
    - Reads the dataset, standardizes labels, and performs a **stratified train-test split** (`test_size=0.2`, `random_state=42`) shared by all models.
  - `evaluate_logistic_regression()`:
    - Loads saved logistic regression model and TF‑IDF vectorizer.
    - Generates test predictions and probability scores.
    - Computes accuracy, precision, recall, F1, confusion matrix, and best decision threshold (based on F1 over a precision–recall curve).
  - `evaluate_lstm()`:
    - Imports `LSTMModel` from `web_ui/app.py` to ensure evaluation uses the same runtime as the web UI.
    - Runs predictions on the test set, collects spam probabilities.
    - Computes the same metrics and best threshold from the precision–recall curve.
  - `evaluate_xlm_roberta()`:
    - Loads XLM-RoBERTa from the saved model directory.
    - Produces predictions and spam probability scores on the test set.
    - Computes the same metrics and a best threshold from the precision–recall curve.

- **Outputs**
  - `metrics.json`:
    - Contains detailed metrics per model (accuracy, precision, recall, F1, confusion matrix, and optionally thresholds and curve-based metrics).
  - `metrics_summary.csv`:
    - Compact table containing model name and main metrics for quick comparison.
  - `thresholds.json`:
    - Per-model best thresholds used in `web_ui/app.py` for spam/ham decisions.

---

## Presentation of Results Using Visual Aids (Charts/Graphs)

Visualization is implemented in the `@presentation_viz` module and uses the outputs of `evaluate_models.py` (`metrics.json`). The generated charts are saved in `presentation_viz/` and are suitable for slides.

### Overall Performance – `metrics_overview.png`

- **Script function**: `plot_metrics_overview()` in `presentation_viz/visualize_results.py`
- **Input**: `metrics.json` (`model_results` section).
- **What the chart shows**
  - A grouped bar chart with, for each model:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**
  - Models on the x-axis (`Logistic Regression`, `LSTM`, `XLM-RoBERTa`), metric values on the y-axis (0–1).

- **Interpretation**
  - Logistic Regression: high accuracy and very high precision, strong F1-score.  
  - LSTM: lower accuracy and F1 due to poor recall (misses many spam texts).  
  - XLM-RoBERTa: highest overall accuracy and F1, strong precision and recall balance.

### Detailed Metrics – `metrics_detailed.png`

- **Script function**: `plot_detailed_metrics()` in `presentation_viz/visualize_results.py`
- **Input**: `metrics.json` confusion matrices (`confusion_matrix` field for each model).  
- **What the chart shows**
  - **Top subplot – Per-class recall per model:**
    - Ham Recall (TN / (TN + FP)).
    - Spam Recall (TP / (TP + FN)).
  - **Bottom subplot – Error rates per model:**
    - False Positive Rate: Ham misclassified as Spam (FP / (TN + FP)).
    - False Negative Rate: Spam misclassified as Ham (FN / (FN + TP)).

- **Interpretation**
  - Logistic Regression:
    - High ham recall and good spam recall; low FP and FN rates.
  - LSTM:
    - Very low false positives (high precision) but high false negative rate; it tends to label many spam messages as ham.
  - XLM-RoBERTa:
    - High recall for both ham and spam with relatively low error rates, demonstrating robust performance.

### Confusion Matrices – `confusion_matrices_grid.png`

- **Script function**: `plot_confusion_matrices()` in `presentation_viz/visualize_results.py`
- **Input**: `metrics.json` confusion matrices for each model.
- **What the chart shows**
  - A row of confusion matrix heatmaps (Ham/Spam vs Predicted Ham/Spam), one for each model:
    - `(TN, FP; FN, TP)` cells annotated with counts.
  - Each subplot is titled with model name and accuracy (e.g., `Logistic Regression – Accuracy: 0.972`).

- **Interpretation for the presentation**
  - Logistic Regression:
    - Very few Ham→Spam misclassifications; moderate Spam→Ham misclassifications.
  - LSTM:
    - Considerable Spam→Ham misclassification, reflecting low recall despite high precision.
  - XLM-RoBERTa:
    - Confusion matrix closest to ideal: very high true positives and true negatives, minimal false positives/negatives.

### Web UI Visuals (Optional Live Demo or Screenshots)

- **Main elements to capture in slides**
  - **Per-model cards** showing:
    - Spam/Ham badge.
    - Confidence percentage and bar.
    - Spam vs Ham probabilities.
  - **Final verdict line** that uses the actual message and majority voting:
    - Example:  
      - `The message: "CONGRATS! You won 1M!"`  
        `is SPAM (3/3 models agreed)`  
      - `The message: "See you later, salamat!"`  
        `is HAM (2/3 models agreed)`.

- These UI screenshots complement the charts by showing how users see and interpret model outputs in real time.

---

## Summary of Findings and Recommendations

### Summary of Model Performance

- **Logistic Regression + TF‑IDF**
  - Accuracy ≈ **97.25%**
  - Precision ≈ **99.36%**
  - Recall ≈ **90.12%**
  - F1-Score ≈ **94.51%**
  - Strengths:
    - Very high precision and strong overall performance.
    - Lightweight and fast; suitable for limited-resource environments.
  - Weaknesses:
    - Slightly less flexible than transformer-based models for very nuanced or long messages.

- **LSTM**
  - Accuracy ≈ **83.42%**
  - Precision ≈ **99.22%**
  - Recall ≈ **37.21%**
  - F1-Score ≈ **54.12%**
  - Strengths:
    - Very high precision (rarely flags ham as spam).
  - Weaknesses:
    - Low recall: it misses a large portion of spam messages, making it less suitable as a standalone spam filter.

- **XLM-RoBERTa**
  - Accuracy ≈ **98.24%**
  - Precision ≈ **95.73%**
  - Recall ≈ **97.67%**
  - F1-Score ≈ **96.69%**
  - Strengths:
    - Best overall model: high precision and recall, robust to Taglish and noisy text.
    - Handles complex phrasing and code-switching effectively.
  - Weaknesses:
    - Heavier model; higher computational cost compared to logistic regression.

### Impact of Preprocessing and Evaluation

- **Consistent label encoding and cleaning** across all models enabled fair comparison using a unified test split.
- **Stratified splitting** ensured both ham and spam classes were well represented in the evaluation set.
- **Detailed metrics derived from confusion matrices** (via `@presentation_viz`) provided more insight than accuracy alone, especially for spam recall and error rates.
- Using the same **LSTM runtime** (NumPy-based) for both web UI and evaluation ensured that reported metrics matched real-world behavior.

### System Design and Interpretability

- The **web UI** presents each model’s prediction independently, which supports the study’s comparative nature.
- The **2-out-of-3 majority voting** final verdict improves reliability for end users while preserving model transparency.
  - Example: If at least 2 out of 3 models classify the message as spam, the system declares spam; similarly for ham.

### Recommendations

- **For practical deployment**
  - Use **XLM-RoBERTa** as the **primary model** for Taglish spam detection due to its best F1 and recall.
  - Maintain **Logistic Regression** as a **backup or low-resource model** where transformer latency or hardware is constrained.

- **For future research and improvements**
  - **Data expansion and augmentation**
    - Collect newer Taglish spam/ham messages to track evolving spam tactics.
    - Augment the dataset with adversarial variants (obfuscated text, leetspeak, mixed languages).
  - **Model enhancements**
    - Explore larger or more specialized multilingual transformers.
    - Investigate additional architectures or attention mechanisms for the LSTM baseline to improve recall.
  - **Threshold and cost-sensitive tuning**
    - Adjust model thresholds according to application requirements (e.g., prioritize spam recall in highly sensitive environments).
  - **Monitoring and retraining**
    - Implement periodic re-evaluation using new labeled data and monitor performance drift.

- **Overall conclusion**
  - The `taglish-spam-detection` project demonstrates that transformer-based models like **XLM-RoBERTa** significantly outperform traditional and RNN-based models on Taglish spam detection, especially in recall and F1-score, while the unified evaluation pipeline and web UI make these differences clear and interpretable for both technical and non-technical audiences.


