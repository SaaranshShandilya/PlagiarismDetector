# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def vectorise (text):
    vectoriser = TfidfVectorizer()
    return vectoriser.fit_transform(text).toarray()

def check_sim(d1, d2):
    return cosine_similarity([d1, d2])


texts = ['Hello this is t1', 'Hello this is t2']

vectors = vectorise(texts)

print(check_sim(vectors[0], vectors[1])[0][1]*100)

# plagiarism_results = set()


# for data in check_plagiarism():
#     print(data)

# stop_words = set(stopwords.words('english'))

# t1 = 'This osidjf this isodjofjos isdjfiojoijsdiof siodjfoisjdf is a word token test'
# t2 = ''

# token_text1 = word_tokenize(t1.lower())
# token_text1 = [token for token in token_text1 if token not in stop_words]
# print(token_text1)

# vectorizer = CountVectorizer(lowercase=True) #max_df is used to stem out the repetitive words

# vector1 = vectorizer.fit_transform([t1]).toarray()



# print(vector1)
