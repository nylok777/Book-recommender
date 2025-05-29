import pandas as pd
import numpy as np
import spacy
from surprise import SVD, dump
from sklearn.feature_extraction.text import TfidfVectorizer

class NLPModel:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

class Books:
    def __init__(self, filepath):
        self.books = pd.read_csv(filepath_or_buffer=filepath, sep=';')
        self.books = self.books.dropna()
    
    def vectorize_authors(self):
        vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+")
        self.author_vectors = vectorizer.fit_transform(self.books['Book-Author'].map(lambda x: str(x).replace(' ', '')))

    def vectorize_summary(self):
        vectorizer = TfidfVectorizer()
        self.summary_vectors = vectorizer.fit_transform(self.books['Summary'])

    def recommend_svd(self, user_id, ratings, title=None):
        svd = dump.load('svd_trained')
        if title is None:
            pass
        else:
            isbn = self.books.query('Book-Title == @title')['ISBN']
        rated_books = ratings.ratings.query('UserID == @user_id')['ISBN']
        new_books = [book for book in self.books['ISBN'].unique() if book not in rated_books]
        preds = [svd[1].predict(user_id, book) for book in new_books]
        top_books = sorted(preds, key=lambda k: k.est, reverse=True)[0:10]
        return top_books

class Users:
    def __init__(self, filepath):
        self.users = pd.read_csv(filepath_or_buffer=filepath, encoding="cp1252", sep=';')
        self.users['Age'] = self.users['Age'].fillna(value=np.random.normal(loc=34.75, scale=14.43))
        self.users['Age'] = self.users['Age'].apply(lambda x: 100 if x > 100 else x)

class Ratings:
    def __init__(self, filepath):
        self.ratings = pd.read_csv(filepath_or_buffer=filepath, encoding="cp1252", sep=';')
        self.ratings.rename(columns={"Book-ID": "BookID", "User-ID": "UserID"}, inplace=True)
