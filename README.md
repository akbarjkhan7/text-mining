# Text Mining — Product Review Rating Prediction

Repository overview
This repository contains a Jupyter notebook implementation (zvdr73.ipynb), the dataset (dataset.csv) and the assignment specification (Problem Question.pdf) for a coursework-style project: predicting an Amazon product review rating (1–5) from the review text. The notebook implements classical and deep models, evaluates them, and saves trained models for later inference.

Status (what's in the repo)
- zvdr73.ipynb — end-to-end notebook with preprocessing, modeling, evaluation and saving models.
- dataset.csv — Amazon product reviews CSV (columns: `Score`, `Text`).
- Problem Question.pdf — the assignment/problem specification (describes goals & requirements).
- README.md — this file.

Project goals (from Problem Question)
- Frame predicting review `Score` (1–5) as a multi-class classification problem.
- Implement and evaluate:
  - Naive Bayes (TF-IDF)
  - k-NN (TF-IDF, choose best k ∈ {1,3,5,7,9} via CV)
  - CNN (text → embeddings → Conv1D layers)
  - LSTM (stacked Bi-LSTM)
- Produce confusion matrices and compute accuracy, precision, recall, F1 for each model.
- Save trained models and provide a function `predict_product_rating(text, model_file)` for inference.

Quick notes about what I inspected in the notebook
- Preprocessing:
  - Lowercasing, punctuation removal via regex, splitting on whitespace,
  - Stopwords removal (NLTK English stopwords).
  - Drop rows with missing `Score` or `Text`, cast `Score` to int, drop duplicates and empty processed text.
- Train/test split:
  - 80/20 stratified split (random_state=42).
- Classical models:
  - TF-IDF vectorizer + MultinomialNB (saved as `naive_bayes_model.pkl`)
  - TF-IDF vectorizer + KNeighborsClassifier (best k selected by 5-fold CV; saved as `knn_model.pkl`)
- Deep models:
  - TextVectorization (max_tokens=20000, output_length=200), Embedding(dim=100)
  - CNN: two Conv1D layers + GlobalMaxPooling, trained with EarlyStopping, saved as `cnn_model.keras`
  - LSTM: two Bidirectional LSTM layers + Dense, trained with EarlyStopping, saved as `lstm_model.keras`
- Evaluation summary (values printed in the notebook)
  - Naive Bayes — Accuracy: 0.542585, F1 (macro): 0.145475
  - k-NN         — Accuracy: 0.572341, F1 (macro): 0.324776
  - CNN          — Accuracy: 0.646067, F1 (macro): 0.408047
  - LSTM         — Accuracy: 0.638477, F1 (macro): 0.397079

Important: dataset path in the notebook
- The notebook loads the dataset using an absolute path:
  `raw_df = pd.read_csv("C:/Users/Dell/Downloads/dataset.csv")`
  This will fail for other users. Change it to a relative path:
  `raw_df = pd.read_csv("dataset.csv")`
  or set a DATASET_PATH variable at top of the notebook.

Requirements (packages used in the notebook)
- Python 3.8+
- pandas, numpy
- scikit-learn
- nltk
- matplotlib, seaborn
- tensorflow (2.x) / keras
- pickle (builtin)

Minimal installation example
```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows (PowerShell)
pip install --upgrade pip
pip install pandas numpy scikit-learn nltk matplotlib seaborn tensorflow
```

Quick start — run the notebook
1. Make sure `dataset.csv` is in the repository root (or update the notebook path).
2. Start Jupyter:
```bash
jupyter lab
# or
jupyter notebook
```
3. Open `zvdr73.ipynb` and run all cells in order. Note training the CNN/LSTM is compute-intensive and will be much faster on a GPU.

Run the notebook non-interactively (execute all cells, save executed notebook):
```bash
jupyter nbconvert --to notebook --execute zvdr73.ipynb --output zvdr73.executed.ipynb
```

Saved models and inference
- The notebook saves models (filenames used in the notebook):
  - `naive_bayes_model.pkl`
  - `knn_model.pkl`
  - `cnn_model.keras`
  - `lstm_model.keras`
- The notebook defines a helper function:
  - `predict_product_rating(text, model_file)` — loads a saved model (.pkl or .keras) and returns the predicted rating (1–5).
- Example usage (in notebook):
```python
predict_product_rating("The product is absolutely amazing and I'll recommend it to my network.", "naive_bayes_model.pkl")
```

How the code represents text for each model
- Naive Bayes & k-NN: TfidfVectorizer (default params in notebook).
- CNN & LSTM: Keras TextVectorization (max_tokens=20_000, sequence_length=200) + Embedding (100d).

Evaluation & metrics
- The notebook reports accuracy, macro F1, macro precision, macro recall, and classification reports and confusion matrices for each model.
- The printed evaluation summary table shows that CNN and LSTM outperform the classical models on accuracy and macro-F1 in the current setup (see numbers above).

Contact / author
- Repo owner: akbarjkhan7 — thank you for sharing the notebook & dataset. If you want, I can prepare the code refactor (scripts + CLI + requirements) and update the README in the repo directly.

If you want me to update README in the repo now, tell me which of the next steps above you want me to take (for example: add requirements.txt and a runnable predict.py; or refactor notebook into scripts), and I will produce the files and suggested commits.
