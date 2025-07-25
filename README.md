# Fake News Detection using NLP

This project uses Natural Language Processing (NLP) and Machine Learning to classify news articles as either "Real" or "Fake". The model is trained on a labeled dataset and utilizes TF-IDF for feature extraction and a Passive Aggressive Classifier for prediction.

---

## About The Project

In an era of information overload, distinguishing between credible news and misinformation is a significant challenge. This project provides a machine learning solution to automatically detect fake news. It processes a dataset of news articles, learns the linguistic patterns associated with real and fake news, and builds a predictive model to classify new, unseen articles.

The core of this project is the `PassiveAggressiveClassifier`, a model well-suited for text classification tasks, which is trained on Term Frequency-Inverse Document Frequency (TF-IDF) vectors derived from the text data.

---

## Tech Stack

* **Language:** Python
* **Libraries:**
    * **Data Handling:** pandas, numpy
    * **NLP & ML:** scikit-learn (for TfidfVectorizer, PassiveAggressiveClassifier, metrics)
    * **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

---

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+
* Jupyter Notebook or a similar environment

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/Sinchana731/Fake-news-detection-using-NLP.git](https://github.com/Sinchana731/Fake-news-detection-using-NLP.git)
    cd Fake-news-detection-using-NLP
    ```

2.  **Install required packages**
    It's recommended to use a virtual environment.
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

3.  **Download the Dataset**
    This project uses a dataset named `news.csv`. Make sure this file is present in your project directory.

4.  **Launch Jupyter Notebook**
    ```sh
    jupyter notebook
    ```
    Then, open the `nlp_proj2.ipynb` file.

---

## Usage

1.  Open the Jupyter Notebook (`nlp_proj2.ipynb`).
2.  Run the cells sequentially to:
    * Load and inspect the `news.csv` dataset.
    * Preprocess the text data and split it into training and testing sets.
    * Initialize and train the TF-IDF Vectorizer.
    * Train the Passive Aggressive Classifier model.
    * Evaluate the model's performance using the accuracy score and confusion matrix.

## Results

The model achieves a high accuracy score in classifying news articles.

* **Accuracy:** 92.8% (This may vary slightly with different train-test splits)
* **Confusion Matrix:** The matrix shows a low number of false positives and false negatives, indicating that the model is effective at distinguishing between real and fake news.

This demonstrates that the combination of TF-IDF and a Passive Aggressive Classifier is a powerful approach for this NLP task.
