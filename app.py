
import os
import re
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------- CONFIG -------------------------
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
QDRANT_COLLECTION = "youtube-comments"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------------- INIT -------------------------
st.set_page_config(page_title="📊 YouTube Comment QA (Free LLM)", layout="wide")
st.title("📊 YouTube Comment Analyzer")

model = SentenceTransformer(EMBED_MODEL_NAME)
client = QdrantClient(":memory:")
analyzer = SentimentIntensityAnalyzer()

@st.cache_resource
def load_phi_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm_pipe = load_phi_model()

def get_video_id(url):
    match = re.search(r"(?:v=|youtu.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def fetch_comments(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page = None
    while len(comments) < 100:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page
        ).execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page = response.get("nextPageToken")
        if not next_page:
            break
    return list(set(comments))

def clean(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ------------------------- MAIN -------------------------
youtube_url = st.text_input("🔗 Enter YouTube video link:")

if youtube_url:
    video_id = get_video_id(youtube_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        comments = fetch_comments(video_id)
        comments = [clean(c) for c in comments if len(c.strip()) > 10]

        if len(comments) < 5:
            st.warning("⚠️ Not enough comments to analyze.")
        else:
            embeddings = model.encode(comments)

            if client.collection_exists(QDRANT_COLLECTION):
                client.delete_collection(QDRANT_COLLECTION)
            client.create_collection(QDRANT_COLLECTION, VectorParams(size=384, distance=Distance.COSINE))

            points = [
                PointStruct(id=i, vector=embeddings[i], payload={"text": comments[i]})
                for i in range(len(comments))
            ]
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)

            st.success("✅ Comments embedded and uploaded to vector DB!")

            query = st.text_input("🧠 Ask a question about the comments (LLM-powered):")
            if query:
                query_vec = model.encode([query])[0]
                results = client.search(QDRANT_COLLECTION, query_vector=query_vec, limit=10)
                top_comments = [r.payload["text"] for r in results if "text" in r.payload]

                st.subheader("📄 Top Retrieved Comments")
                for c in top_comments:
                    st.markdown(f"- {c}")

                context = "\n".join(f"- {c}" for c in top_comments)
                prompt = f"Based on the following YouTube comments, answer the question: '{query}'\n{context}\nAnswer:"

                st.subheader("💬 Phi's Answer (LLM)")
                result = llm_pipe(prompt, max_new_tokens=150, temperature=0.7)[0]["generated_text"]
                st.write(result.split("Answer:")[-1].strip())

                sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
                for c in top_comments:
                    score = analyzer.polarity_scores(c)["compound"]
                    if score > 0.2:
                        sentiment = "positive"
                    elif score < -0.2:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"
                    sentiment_counts[sentiment] += 1

                st.subheader("📊 Sentiment Breakdown")
                fig = px.pie(
                    names=sentiment_counts.keys(),
                    values=sentiment_counts.values(),
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig)

                vectorizer = CountVectorizer(stop_words='english', max_features=10)
                X = vectorizer.fit_transform(top_comments)
                keywords = vectorizer.get_feature_names_out()
                st.markdown("📝 **Keyword Summary**")
                st.write(", ".join(keywords))
