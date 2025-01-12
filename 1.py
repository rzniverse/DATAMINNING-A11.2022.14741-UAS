import streamlit as st
import googleapiclient.discovery
import pandas as pd
import re
import html
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from deep_translator import GoogleTranslator
from textblob import TextBlob
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Streamlit UI Configuration
st.set_page_config(
    page_title="Machine Learning 2024",
    layout="wide",
)
st.title("Machine Learning 2024")

# Sidebar Menu
st.sidebar.title("Menu")
menu_options = ["Crawl Dataset YT", "Preprocessing Data", "Process Text"]
selected_menu = st.sidebar.radio("Select a menu", menu_options)

if 'menu' not in st.session_state:
    st.session_state.menu = "Crawl Dataset YT"
else:
    st.session_state.menu = selected_menu

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    pattern = r"(?<=v=)[a-zA-Z0-9_-]+(?=&|\?|$)"
    match = re.search(pattern, url)
    if match:
        return match.group(0)
    else:
        return None

# Function to fetch YouTube comments
def fetch_youtube_comments(video_url):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyC2snPu9P_p95Qm9Ej3tzeGY3Bantdf0L0"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    video_id = extract_video_id(video_url)

    if video_id:
        comments = []
        page_token = None

        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'Timestamp': comment['publishedAt'],
                    'Username': comment['authorDisplayName'],
                    'VideoID': video_id,
                    'Comment': comment['textDisplay'],
                    'Date': comment['updatedAt'] if 'updatedAt' in comment else comment['publishedAt']
                })

            page_token = response.get('nextPageToken')

            if not page_token:
                break

        df = pd.DataFrame(comments)
        return df
    else:
        st.error("Invalid YouTube URL. Please ensure the URL is correct.")
        return None

# Preprocessing Functions
def clean_comment(text):
    text = html.unescape(text)
    text = re.sub(r'\.', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def replace_taboo_words(text, kamus_tidak_baku):
    if isinstance(text, str):
        words = text.split()
        replaced_words = [kamus_tidak_baku.get(word, word) for word in words]
        replaced_text = ' '.join(replaced_words)
    else:
        replaced_text = ''
    return replaced_text

def stem_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    stemmed_text = stemmer.stem(text)
    print(f"Stemming: '{text}' -> '{stemmed_text}'")  # Print statement for terminal output
    return stemmed_text

def convert_english(text):
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        print(f"Translating '{text}' to '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

def process_text(text):
    cleaned_text = clean_comment(text)
    normalized_text = replace_taboo_words(cleaned_text, kata_tidak_baku)
    tokenized_text = nltk.word_tokenize(normalized_text)
    stopwords_removed = [token for token in tokenized_text if token not in stop_words]
    stemmed_text = stem_text(stopwords_removed)
    translated_text = convert_english(stemmed_text)
    blob = TextBlob(translated_text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment = 'positive' if sentiment_polarity > 0 else ('neutral' if sentiment_polarity == 0 else 'negative')
    return sentiment, sentiment_polarity

# Initialize Stemmer and Load Stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
stop_words.add('ya')
kamus_data = pd.read_excel('kamuskatabaku.xlsx')
kata_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

# Display corresponding content based on menu selection
if st.session_state.menu == "Crawl Dataset YT":
    st.sidebar.header("Crawl Dataset YT")
    
    video_url = st.text_input("Enter YouTube video URL:")
    
    if st.button("Fetch Comments"):
        if video_url:
            with st.spinner("Fetching comments..."):
                df = fetch_youtube_comments(video_url)
                if df is not None:
                    st.write(f"Total comments fetched: {len(df)}")
                    st.dataframe(df.head())
                    if st.button("Save to CSV"):
                        df.to_csv('youtube_video_comments.csv', index=False)
                        st.success("File saved successfully.")
        else:
            st.error("Please enter a YouTube video URL.")

elif st.session_state.menu == "Preprocessing Data":
    st.sidebar.header("Preprocessing Data")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Step 2.1: Display Wordcloud
        st.subheader("Step 2.1: Display Wordcloud")
        text = ' '.join(df['Comment'])
        word_freq = Counter(text.split())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Step 2.2: Display Word Frequency
        st.subheader("Step 2.2: Display Word Frequency")
        tokens = text.split()
        counter = Counter(tokens)
        most_common = counter.most_common(10)
        words, counts = zip(*most_common)
        colors = plt.cm.Paired(range(len(words)))
        plt.figure(figsize=(10, 6))
        bars = plt.bar(words, counts, color=colors)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Most Common Words')
        plt.xticks(rotation=45)
        for bar, num in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')
        st.pyplot(plt)

        # Step 3: Preprocessing
        st.subheader("Step 3: Preprocessing")

        # Step 3.1: Clean Comment
        st.subheader("Step 3.1: Clean Comment")
        df['Cleaning'] = df['Comment'].str.lower().apply(clean_comment)
        df_cleaned = df[['Username', 'Comment', 'Cleaning']]
        df_cleaned.to_csv('cleaned_comments.csv', index=False)
        st.write(df_cleaned.head())

        # Step 3.2: Normalization
        st.subheader("Step 3.2: Normalization")
        df['Normalisasi'] = df['Cleaning'].apply(lambda x: replace_taboo_words(x, kata_tidak_baku))
        df['Normalisasi'] = df['Normalisasi'].str.lower()
        df.to_csv('normalized_comments.csv', index=False)
        st.write(df[['Username', 'Comment', 'Normalisasi']].head())

        # Plot Normalization Results
        st.subheader("Word Frequency after Normalization")
        text = " ".join(df['Normalisasi'])
        tokens = text.split()
        counter = Counter(tokens)
        most_common = counter.most_common(10)
        words, counts = zip(*most_common)
        colors = plt.cm.Paired(range(len(words)))
        plt.figure(figsize=(10, 6))
        bars = plt.bar(words, counts, color=colors)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Most Common Words after Normalization')
        plt.xticks(rotation=45)
        for bar, num in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')
        st.pyplot(plt)

        # Step 3.3: Tokenization
        st.subheader("Step 3.3: Tokenization")
        df['Tokenization'] = df['Normalisasi'].apply(nltk.word_tokenize)
        st.write(df[['Username', 'Comment', 'Tokenization']].head())

        # Step 3.4: Remove Stopwords
        st.subheader("Step 3.4: Remove Stopwords")
        df['Stopwords_Removal'] = df['Tokenization'].apply(lambda tokens: [token for token in tokens if token not in stop_words])
        st.write(df[['Username', 'Comment', 'Stopwords_Removal']].head())

        # Step 3.5: Stemming
        st.subheader("Step 3.5: Stemming")
        df['Stemming'] = df['Stopwords_Removal'].apply(stem_text)
        st.write(df[['Username', 'Comment', 'Stemming']].head())

        # Step 3.6: Translation
        st.subheader("Step 3.6: Translation")
        df['Translated'] = df['Stemming'].apply(convert_english)
        st.write(df[['Username', 'Comment', 'Translated']].head())

        # Step 3.7: Sentiment Analysis
        st.subheader("Step 3.7: Sentiment Analysis")
        df['Sentiment_polarity'] = df['Translated'].apply(lambda text: TextBlob(text).sentiment.polarity)
        df['Sentiment'] = df['Sentiment_polarity'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))
        st.write(df[['Username', 'Comment', 'Sentiment_polarity', 'Sentiment']].head())

        # Save preprocessed data
        df.to_csv('preprocessed_comments.csv', index=False)
        st.success("Preprocessed data saved successfully.")
        
        # Model Training and Accuracy Evaluation
        st.subheader("Model Training and Accuracy Evaluation")
        
        # Prepare data for training
        X = df['Translated']
        y = df['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train the model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)
        st.text("Confusion Matrix:")
        st.write(cm)

        # Plot sentiment polarity distribution
        st.subheader("Sentiment Polarity Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Sentiment_polarity'], bins=20, kde=True)
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sentiment Polarity')
        st.pyplot(plt)

elif st.session_state.menu == "Process Text":
    st.sidebar.header("Process Text")
    
    input_text = st.text_area("Enter text to process:")

    if st.button("Process Text"):
        if input_text:
            sentiment, polarity = process_text(input_text)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Sentiment Polarity: {polarity:.2f}")
        else:
            st.error("Please enter some text to process.")
