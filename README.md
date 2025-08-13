# YouTube Comment Analyzer

An interactive web app that:
- Fetches comments from a YouTube video
- Embeds and stores them in a **Qdrant** vector database
- Lets you **ask questions** about the comments using **OpenAI GPT models**
- Runs **sentiment analysis** with VADER
- Extracts **top keywords**
- Displays results in a nice dashboard

---

## 🚀 Features

- **YouTube API Integration** — Fetch up to 100 comments from a video
- **Vector Search** — Store & search comment embeddings with Qdrant
- **AI Q&A** — Ask questions about the comment set (powered by GPT)
- **Sentiment Analysis** — Pie chart breakdown (positive / neutral / negative)
- **Keyword Extraction** — Top 10 keywords from retrieved comments

---

## 🛠 Tech Stack

- [Python 3.9+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Qdrant](https://qdrant.tech/)
- [OpenAI Python SDK](https://platform.openai.com/)
- [Google API Client](https://developers.google.com/api-client-library/python)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Plotly Express](https://plotly.com/python/plotly-express/)

---

## 📦 Installation

Install dependencies before running:

bash
pip install streamlit sentence-transformers scikit-learn qdrant-client \
    google-api-python-client vaderSentiment plotly transformers
this code


Step 1: Add Your API Key


Create a secrets file inside a folder in your project directory:

YOUTUBE_API_KEY = "your_actual_youtube_api_key"

Step 2: Run the App


In your terminal, from the root of your project folder:


streamlit run app.py


You will be able to see the page similar to the screenshots provided.






