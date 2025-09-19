![fake-news](https://github.com/user-attachments/assets/9fbf2a5a-7ccb-44c6-8dbf-3376081b6358)

# Fake and real news dataset | Kaggle
This project is developed for the Machine Learning exam at University of Cagliari.
- Author: Andrea Labelli
- Supervisor: Prof. Battista Biggio

## Overview
This project focuses on fake news classification using different machine learning models and text representation techniques.
We use the Fake and Real News Dataset available on [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
, which contains two CSV files:

- Fake.csv -> collection of fake news articles
- True.csv -> collection of real news articles

The main objective is to explore text preprocessing, feature extraction, and model comparison in order to evaluate which approaches are most effective for detecting fake news.

## How to run the code in Google Colab
- Open the file uploaded in the repo
- Upload the two files Fake.csv and Real.csv

## Implementation
### Text Preprocessing 
- Stopword removal
- Lemmatization
- Lowercasing
- Filtering out short words

### Data Representations tested:
- Tokenizer-based statistics -> Statistical features extracted from tokenized sequences (average token index, frequency, etc.).
- Count Vectorizer -> Bag-of-Words representation (word counts).
- TF-IDF Vectorizer -> Term Frequency - Inverse Document Frequency representation (weighted word importance).

### Models compared:
- Naive Bayes
- Perceptron
- Random Forest
- Support Vector Machine (SVM)

Each classifier was trained and tested using both simple train/test split and 5-fold cross validation.

In the 5-fold cross validation (K=5) computation:
- The dataset is divided into 5 folds
- Each fold is used once as a test set, while the remaining 4 folds are used for training.
- The process is repeated 5 times, and results are averaged.
- Simple split gives performance for only one partition of the data (risk of bias).
- K-Fold CV provides a more reliable estimate by considering multiple partitions.

### Evaluation Metrics
To measure performance, we used Precision, Recall, and F1-score, which are common in classification tasks.

- Average precision

$\large\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$

Precision measures how many of the articles classified as fake were actually fake.

- Average recall

$\large\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$

Recall measures how many of the actual positives were correctly identified.

- Average F1-score

$\large F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

F1-score is the harmonic mean of Precision and Recall.

## Results
- TF-IDF representation consistently produced the best results across all models.
- Naive Bayes + TF-IDF performed particularly well, reaching very high precision and recall.
- SVM also showed excellent performance, although more computationally expensive.
- TokenizerStatistics performs significantly worse — useful for baseline but not competitive when vocabulary and term weighting are available.
- Confusion matrices showed that both classes (Fake and True) were classified with high reliability.

## Key Takeaways
- Feature representation matters as much as the choice of classifier.
- TF-IDF is the most effective representation for linear models (SVM, Perceptron).
- Naive Bayes is strong and efficient with Count Vectorizer.
- Cross validation ensures that results are not dependent on a lucky or unlucky split.
