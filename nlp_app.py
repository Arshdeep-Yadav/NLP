import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure all necessary NLTK resources are available
resources = ["punkt", "wordnet", "stopwords"]

for resource in resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize_words(text):
    return word_tokenize(text) if text.strip() else []

def tokenize_sentences(text):
    return sent_tokenize(text) if text.strip() else []

def perform_stemming(text):
    tokens = word_tokenize(text) if text.strip() else []
    return [stemmer.stem(token) for token in tokens]

def perform_lemmatization(text):
    tokens = word_tokenize(text) if text.strip() else []
    return [lemmatizer.lemmatize(token) for token in tokens]

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text) if text.strip() else []
    return [token for token in tokens if token.lower() not in stop_words]

# Streamlit UI
st.title("NLP Processing App")
st.write("Enter your text below and choose an NLP operation:")

user_input = st.text_area("Input Text", height=150)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    word_tok_btn = st.button("Word Tokenization")
with col2:
    sent_tok_btn = st.button("Sentence Tokenization")
with col3:
    stemming_btn = st.button("Stemming")
with col4:
    lemma_btn = st.button("Lemmatization")
with col5:
    stopwords_btn = st.button("Stopword Removal")

if user_input.strip():
    if word_tok_btn:
        with st.expander("Word Tokenization Results", expanded=True):
            st.write(tokenize_words(user_input))

    if sent_tok_btn:
        with st.expander("Sentence Tokenization Results", expanded=True):
            st.write(tokenize_sentences(user_input))

    if stemming_btn:
        with st.expander("Stemming Results", expanded=True):
            st.write(perform_stemming(user_input))

    if lemma_btn:
        with st.expander("Lemmatization Results", expanded=True):
            st.write(perform_lemmatization(user_input))

    if stopwords_btn:
        with st.expander("Stopword Removal Results", expanded=True):
            st.write(remove_stopwords(user_input))
else:
    st.warning("Please enter some text before performing operations.")
