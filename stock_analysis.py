from openai import OpenAI
import os
import json

import pandas as pd
from gnews import GNews
from pydantic import BaseModel

from dotenv import load_dotenv


google_news = GNews()
news = google_news.get_news('NVDA')

news_titles = [article['title'] for article in news]

load_dotenv()  

client = OpenAI()

class NewsAnalysis(BaseModel):
    sentiment: str
    future_looking: bool


results =[]
for title in news_titles:

    # We add a system message to instruct GPT to return valid JSON:
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that strictly returns valid JSON. "
            "The JSON must have only two keys: 'sentiment' (string) and 'future_looking' (boolean). "
            "Example: {\"sentiment\": \"positive\", \"future_looking\": true}\n\n"
            "No additional text or keys."
        )
    }

    
    user_message = {
        "role": "user",
        "content": (
            f"Analyze the following title for sentiment (positive, negative, or neutral) "
            f"and whether it provides future-looking financial insight, predictions, or "
            f"guidance on whether to buy/hold/sell the stock (True or False): {title}"
        )
    }

    messages = [system_message, user_message]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    # Extract GPT's reply
    content = completion.choices[0].message.content.strip()
    
    try:
        analysis_dict = json.loads(content)
    except json.JSONDecodeError:
        analysis_dict = {"sentiment": "unknown", "future_looking": False}

    sentiment_analysis = NewsAnalysis(**analysis_dict)
    results.append({
        "title": title,
        "sentiment": sentiment_analysis.sentiment,
        "future_looking": sentiment_analysis.future_looking
    })

df = pd.DataFrame(results)
df.to_csv("final_ouput.csv")