# perform basic operation of text using nltk
''' Perform all the text operation using nltk'''
# import nltk
# nltk.download('all')


''' nltk text operation'''
# from nltk import pos_tag
# from nltk import word_tokenize
# text = "GeeksforGeeks is a Computer Science platform."
# tokenized_text = word_tokenize(text)
# tags = tokens_tag = pos_tag(tokenized_text)
# print(tags)

# NNP: Proper noun, singular (e.g., 'GeeksforGeeks', 'Computer', 'Science')
# VBZ: Verb, 3rd person singular present (e.g., 'is')
# DT: Determiner (e.g., 'a')
# NN: Noun, singular or mass (e.g., 'platform')

'''NER'''

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
# Download the required resource for NER
nltk.download('maxent_ne_chunker_tab')
nltk.download('words') # This resource is also needed for the chunker

# Sample text
text = "Barack Obama was born in Hawaii in 1961."

# Tokenize and POS tag the sentence
tokens = word_tokenize(text)
tags = pos_tag(tokens)

# Apply Named Entity Recognition
entities = ne_chunk(tags)
print(entities)



