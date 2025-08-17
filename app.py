import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🔍", layout="wide")

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    sentiment = model.predict(text)
    return "😡 Negative" if sentiment == 0 else "😊 Positive"

def create_card(tweet_text, sentiment):
    color = "#e63946" if "Negative" in sentiment else "#2a9d8f"
    card_html = f"""
    <div style="background-color: {color}; 
                padding: 15px; 
                border-radius: 12px; 
                margin: 12px 0; 
                box-shadow: 2px 2px 10px rgba(0,0,0,0.15);">
        <h4 style="color: white; margin:0;">{sentiment}</h4>
        <p style="color: white; font-size:16px; margin-top:8px;">{tweet_text}</p>
    </div>
    """
    return card_html

def main():
    st.title("🔍 Twitter Sentiment Analysis")
    st.markdown("Analyze sentiments from custom text or live tweets in a **beautiful interface** 🚀")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    st.sidebar.header("Navigation")
    option = st.sidebar.radio("Choose an option:", ["✍️ Input Text", "🐦 Fetch Tweets"])

    if option == "✍️ Input Text":
        st.subheader("✍️ Enter text to analyze")
        text_input = st.text_area("Write your text below:", height=120, placeholder="Type something...")
        if st.button("🔮 Analyze Sentiment"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.markdown(create_card(text_input, sentiment), unsafe_allow_html=True)

    elif option == "🐦 Fetch Tweets":
        st.subheader("🐦 Get Tweets from User")
        username = st.text_input("Enter Twitter username (without @):")
        if st.button("📥 Fetch Tweets"):
            tweets_data = scraper.get_tweets(username, mode='user', number=5)
            if 'tweets' in tweets_data:
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet['text']
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                    st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
            else:
                st.warning("⚠️ No tweets found or an error occurred.")

if __name__ == "__main__":
    main()
