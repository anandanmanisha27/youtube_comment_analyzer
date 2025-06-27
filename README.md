# 📊 YouTube Comment QA App

This app fetches YouTube comments, embeds them using `sentence-transformers`, stores them in Qdrant, and lets you:

- Ask questions (via Phi-1.5 LLM)
- Analyze sentiment
- Extract top keywords

### 🚀 How to Run

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



