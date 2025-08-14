
# from textblob import download_corpora
# download_corpora.download_all()
from textblob import TextBlob

text = "Hello! I am learning NLP with TextBlob."
blob = TextBlob(text)

words = blob.words
print(words)