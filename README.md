# Movie Review Text Classification

A complete NLP classification project built in a Jupyter notebook that prepares movie review text, extracts features, trains and compares multiple models, and saves the final classifier for reuse.

## Project Summary

This project demonstrates a real-world text classification pipeline for movie reviews. It includes:

- Loading and inspecting a movie review dataset
- Cleaning raw text and correcting encoding issues
- Exploring word frequency, bigrams, and trigrams
- Building TF-IDF features from both word and character n-grams
- Training and evaluating several machine learning models
- Performing hyperparameter optimization
- Saving the trained model and vectorizers for deployment

This is a strong internship resume project because it covers data preparation, NLP feature engineering, model training, evaluation, and reproducibility.

## Dataset

- Source: `movie/movie.csv`
- Key columns:
  - `text`: raw movie review content
  - `label`: target category for classification
- Initial data handling:
  - BOM-corrupted column renaming
  - Missing value checks
  - Duplicate detection
  - Label distribution analysis

## Preprocessing

Text cleaning and normalization steps include:

- Lowercasing all text
- Removing HTML tags
- Removing URLs
- Keeping only alphanumeric characters and apostrophes
- Removing extra whitespace

This ensures consistent input for model training and reduces noise from punctuation and formatting artifacts.

## Feature Engineering

Two complementary TF-IDF feature sets were created:

- Word-level TF-IDF using unigrams and bigrams
- Character-level TF-IDF using 3- to 5-gram character sequences

These features were combined into a single sparse matrix, capturing both semantic and subword patterns.

## Exploratory Analysis

- Frequent word counts by label
- Word cloud visualizations for each category
- Top bigrams and trigrams by label using `CountVectorizer`

This helped identify distinctive language patterns and supported model selection.

## Models Evaluated

Baseline models trained and compared:

- Logistic Regression
- LinearSVC
- SGDClassifier
- PassiveAggressiveClassifier
- RidgeClassifier
- Perceptron
- Multinomial Naive Bayes
- Complement Naive Bayes

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1-score

## Hyperparameter Tuning

- Used `RandomizedSearchCV` for `SGDClassifier`
- Tuned parameters such as `loss`, `penalty`, `alpha`, `learning_rate`, `eta0`, `max_iter`, and `early_stopping`
- Selected best performing hyperparameters and retrained the final model

## Final Model Persistence

Saved artifacts:

- SGD_Model.pkl
- Word_vector.pkl
- Char_vector.pkl

This enables fast reuse of the trained classifier and preprocessing pipeline.

## Tools and Libraries

- Python
- pandas
- scikit-learn
- matplotlib
- wordcloud
- scipy
- pickle

## How to Run

1. Open app.ipynb in Jupyter Notebook / Jupyter Lab
2. Ensure the dataset is available at `movie/movie.csv`
3. Install required Python packages
4. Run notebook cells sequentially from data loading through model training
5. Inspect final evaluation metrics and saved model files

## Resume-Ready Bullet Points

- Built a movie review classification pipeline using Python, pandas, and scikit-learn.
- Implemented NLP preprocessing and TF-IDF feature engineering for both word and character n-grams.
- Compared multiple classifiers and used randomized hyperparameter search to optimize performance.
- Serialized the final model and vectorizers for future deployment and reuse.
- Demonstrated strong practical skills in NLP, model evaluation, and data-driven workflow design.