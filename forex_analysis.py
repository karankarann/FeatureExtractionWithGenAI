import os
import json

import openai
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from gnews import GNews
from pydantic import BaseModel


# Load environment variables (ensure .env contains OPENAI_API_KEY)
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

# Define a Pydantic model to structure the response
class NewsAnalysis(BaseModel):
    sentiment: str
    future_looking: bool

# Initialize the GNews client
google_news = GNews()

# ---- ADAPTATION FOR FOREX ----
# For example, we might want to get news about EUR/USD.
# You could replace 'EUR USD Forex' with any relevant query or currency pair.
query = "EUR USD Forex"

# Retrieve news articles from Google News
news = google_news.get_news(query)

# Extract titles from the fetched news
news_titles = [article['title'] for article in news[:10]]

results = []

for title in news_titles:
    # System message to ensure the model returns valid JSON only
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that strictly returns valid JSON. "
            "The JSON must have only two keys: 'sentiment' (string) and 'future_looking' (boolean). "
            "Example: {\"sentiment\": \"positive\", \"future_looking\": true}\n\n"
            "No additional text or keys."
        )
    }

    # User message requesting an analysis of sentiment and future-looking nature
    user_message = {
        "role": "user",
        "content": (
            f"Analyze the following title for sentiment (positive, negative, or neutral) "
            f"and whether it provides future-looking financial insight, predictions, or "
            f"guidance on whether to buy/hold/sell the currency (True or False): {title}"
        )
    }

    messages = [system_message, user_message]

    # Create a chat completion request using the GPT-3.5 model (adjust temperature/ model as needed)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )

    # Extract the assistant's reply
    content = completion.choices[0].message.content.strip()

    # Attempt to parse the JSON returned by the model
    try:
        analysis_dict = json.loads(content)
    except json.JSONDecodeError:
        # Fallback in case there's any parsing error
        analysis_dict = {"sentiment": "unknown", "future_looking": False}

    # Validate against our NewsAnalysis model
    sentiment_analysis = NewsAnalysis(**analysis_dict)
    results.append({
        "title": title,
        "sentiment": sentiment_analysis.sentiment,
        "future_looking": sentiment_analysis.future_looking
    })

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("final_forex_output.csv", index=False)

print("Analysis complete. Results saved to final_forex_output.csv.")
