# Flipkart-Reviews-Sentiment-Analysis-using-Python

 


# Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

# Flipkart Reviews Sentiment Analysis using Python

## Introduction
This article is based on the analysis of the reviews and ratings given by users on Flipkart. By analyzing this data, we can gain insights into product quality and user experiences. We will use machine learning to analyze the data and make it ready for sentiment prediction, specifically, to predict whether a review is positive or negative.

Before we begin, you can download the dataset for this project by clicking [this link](#) and placing it in the same directory as this notebook.

## Importing Libraries
We'll be using several Python libraries for this project:

- Pandas: For importing and manipulating the dataset.
- Scikit-learn: For importing machine learning models, accuracy metrics, and TF-IDF vectorization.
- NLTK (Natural Language Toolkit): For text analysis and stopwords.
- Matplotlib: For data visualization and plotting.
- eaborn For additional data visualization.
- WordCloud: For creating word clouds.

Here's how to import these libraries:
 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
 

## Importing Dataset
We'll start by importing the dataset. Make sure you've downloaded the dataset file and placed it in the same directory as this notebook. Here's how to import the dataset:

 
data = pd read_csv('flipkart_data.csv')
data.head()
 

## Exploratory Data Analysis (EDA)
Next, we'll perform exploratory data analysis (EDA) to understand the dataset better. We'll explore unique ratings and visualize them. We'll also convert the ratings into two classes: positive and negative. Here's the code:

 
# unique ratings
pd unique(data['rating'])
 

We'll also create a countplot to visualize the ratings.
 
sns countplot(data=data, x='rating', order=data.rating.value_counts().index)
 

## Feature Engineering
We will create additional features from the dataset, such as preprocessing the text data. We'll remove punctuation, convert text to lowercase, and remove stopwords. Here's the function to preprocess text:
 
from tqdm import tqdm

def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                          for token in nltk.word_tokenize(sentence)
                                          if token.lower() not in stopwords.words('english')))

    return preprocessed_text
 

We'll then implement this function for the dataset:
  
preprocessed_review = preprocess_text(data['review'].values)
data['review'] = preprocessed_review
 

## Model Development and Evaluation
Once the data is preprocessed, we'll convert the text data into vectors using TF-IDF vectorization. We'll split the dataset into training and testing sets. We'll train a machine learning model (Decision Tree Classifier) and evaluate it. Here's how to do it:

 
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Testing the model
pred = model.predict(X_train)
print(accuracy_score(y_train, pred))
 

We'll also visualize the confusion matrix for the results:

 
from sklearn import metrics
cm = confusion_matrix(y_train, pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=[False, True])

cm_display.plot()
 

## Conclusion
In this project, we performed sentiment analysis on Flipkart reviews. We successfully transformed the text data into vectors using TF-IDF and trained a Decision Tree Classifier model. This model achieved good accuracy for sentiment prediction. In the future, we can apply this approach to larger datasets and even scrape data directly from websites for analysis.
 
