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
from openai import OpenAI

# ------------------------- CONFIG -------------------------
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

QDRANT_COLLECTION = "youtube-comments"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------------- INIT -------------------------
st.set_page_config(page_title="📊 YouTube Comment QA (OpenAI-powered)", layout="wide")
st.title(" YouTube Comment Analyzer")

model = SentenceTransformer(EMBED_MODEL_NAME)
client = QdrantClient(":memory:")
analyzer = SentimentIntensityAnalyzer()

# OpenAI setup
openai_client = OpenAI(api_key=OPENAI_API_KEY)
st.success("✅  QA model ready.")

# ------------------------- FUNCTIONS -------------------------
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
            st.warning("⚠ Not enough comments to analyze.")
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

            query = st.text_input("🧠 Ask a question about the comments (OpenAI-powered):")
            if query:
                query_vec = model.encode([query])[0]
                results = client.search(QDRANT_COLLECTION, query_vector=query_vec, limit=10)
                top_comments = [r.payload["text"] for r in results if "text" in r.payload]

                st.subheader("📄 Top Retrieved Comments")
                for c in top_comments:
                    st.markdown(f"- {c}")

                context = "\n".join(top_comments)
                st.subheader("💬 OpenAI's Answer")
                try:
                    prompt = f"""You are an expert at analyzing YouTube comments.
Use the following context to answer the question.
If the answer is not explicitly in the comments, say you cannot answer.

Question: {query}
Context (comments): {context}
"""
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert at analyzing YouTube comments."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    st.write(response.choices[0].message.content.strip())
                except Exception as e:
                    st.warning(f"⚠ OpenAI could not answer: {e}")

                # Sentiment Analysis
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

                # Keyword Extraction
                vectorizer = CountVectorizer(stop_words='english', max_features=10)
                X = vectorizer.fit_transform(top_comments)
                keywords = vectorizer.get_feature_names_out()
                st.markdown("📝 *Keyword Summary*")
                st.write(", ".join(keywords))
