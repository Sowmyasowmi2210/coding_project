from sklearn.feature_extraction.text import CountVectorizer

#scikit learn is library and CoutVectorizer is a method from the library

corpus = ["I like apples", "I hate apples"]
vectorizer = CountVectorizer()  # ← from scikit-learn
X = vectorizer.fit_transform(corpus)
print(X)