# Filename: data_scraper.py
import praw
import pandas as pd
# from psaw import PushshiftAPI
from datetime import datetime
import json
import os
import time
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from bertopic import BERTopic
# from collections import Counter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List


class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str, subreddit):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # self.pushshift = PushshiftAPI(self.reddit)
        self.subreddit = subreddit
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.qdrant = QdrantClient("localhost", port=6333)
        self.qdrant.recreate_collection(
            collection_name=f"subreddit_memory_{subreddit}",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )

    def scrape_subreddit(self, start_date: str, end_date: str, limit: int = 500) -> dict:
        """Scrape all threads from a subreddit, including historical data if available."""
        start_epoch = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_epoch = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        subreddit_data = {"threads": {}}
        subreddit = self.reddit.subreddit(self.subreddit)

        print(f"Scraping r/{self.subreddit} from {start_date} to {end_date}...")
        
        # Use multiple listing types to maximize data within limits
        for listing in ["hot", "new", "top"]:  # Combine different listings to get more data
            print(f"Scraping {listing} listings...")
            try:
                if listing == "top":
                    submissions = subreddit.top(time_filter="all", limit=limit // 3)  # Distribute limit across listings
                else:
                    submissions = getattr(subreddit, listing)(limit=limit // 3)
                
                for submission in submissions:
                    if submission.created_utc < start_epoch or submission.created_utc > end_epoch:
                        continue  # Skip submissions outside the date range
                    
                    submission.comments.replace_more(limit=0)  # Fetch all comments, avoid 'MoreComments'
                    thread_data = {
                        "submission": {
                            "id": submission.id,
                            "title": submission.title,
                            "text": submission.selftext,
                            "author": str(submission.author),
                            "upvotes": submission.ups,
                            "downvotes": submission.downs,
                            "timestamp": submission.created_utc
                        },
                        "comments": []
                    }
                    for comment in submission.comments.list():
                        if comment.created_utc < start_epoch or comment.created_utc > end_epoch:
                            continue  # Skip comments outside the date range
                        thread_data["comments"].append({
                            "id": comment.id,
                            "text": comment.body,
                            "author": str(comment.author),
                            "parent_id": comment.parent_id.split("_")[1] if comment.parent_id != submission.id else None,
                            "upvotes": comment.ups,
                            "downvotes": comment.downs,
                            "timestamp": comment.created_utc
                        })
                    subreddit_data["threads"][submission.id] = thread_data
                    
                    # Respect rate limits by adding a small delay
                    time.sleep(1)  # Delay to avoid hitting Reddit API rate limits (60 requests per minute)
            except Exception as e:
                print(f"Error scraping {listing} listings: {e}")
                time.sleep(60)  # Wait longer if rate-limited
        
        print(f"Scraped {len(subreddit_data['threads'])} threads.")
        return subreddit_data

    def group_by_user(self, subreddit_data: dict) -> dict:
        """Group all actions by user across the subreddit."""
        user_actions = {}
        for thread_id, thread_data in subreddit_data["threads"].items():
            submission = thread_data["submission"]
            user_actions[submission["author"]] = user_actions.get(submission["author"], []) + [{
                "thread_id": thread_id,
                "type": "post",
                "text": submission["text"],
                "parent_id": None,
                "upvotes": submission["upvotes"],
                "downvotes": submission["downvotes"],
                "timestamp": submission["timestamp"]
            }]
            for comment in thread_data["comments"]:
                user_actions[comment["author"]] = user_actions.get(comment["author"], []) + [{
                    "thread_id": thread_id,
                    "type": "comment",
                    "text": comment["text"],
                    "parent_id": comment["parent_id"],
                    "upvotes": comment["upvotes"],
                    "downvotes": comment["downvotes"],
                    "timestamp": comment["timestamp"]
                }]
        return user_actions

    def store_subreddit_memory(self, subreddit_data: dict, batch_size: int = 500):
        """Store subreddit-level data in Qdrant for similarity retrieval."""
        points = []
        point_id = 0
        for thread_id, thread_data in subreddit_data["threads"].items():
            submission = thread_data["submission"]
            text = f"{submission['title']} {submission['text']}"
            vector = self.embedder.encode(text).tolist()
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "thread_id": thread_id,
                    "text": text,
                    "sentiment": self._analyze_sentiment(text),
                    "upvotes": submission["upvotes"],
                    "downvotes": submission["downvotes"]
                }
            ))
            point_id += 1
            for comment in thread_data["comments"]:
                truncated_text = comment["text"][:250] # Avoid exceeding JSON limits
                vector = self.embedder.encode(truncated_text).tolist()
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "thread_id": thread_id,
                        "text": truncated_text,
                        "sentiment": self._analyze_sentiment(truncated_text),
                        "upvotes": comment["upvotes"],
                        "downvotes": comment["downvotes"]
                    }
                ))
                point_id += 1
        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            # Manually serialize PointStruct to dict for size calculation
            batch_dicts = [
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
                for point in batch
            ]
            batch_size_bytes = len(json.dumps(batch_dicts).encode('utf-8'))
            print(f"Upserting batch {i // batch_size + 1} with {len(batch)} points ({batch_size_bytes} bytes)")
            if batch_size_bytes > 33_554_432:
                print(f"Warning: Batch size {batch_size_bytes} bytes exceeds limit. Reduce batch_size or truncate further.")
            self.qdrant.upsert(
                collection_name=f"subreddit_memory_{self.subreddit}",
                points=batch
            )

    def retrieve_similar_threads(self, text: str, limit: int = 5) -> List[dict]:
        """Retrieve threads most similar to the given text."""
        vector = self.embedder.encode(text).tolist()
        results = self.qdrant.search(
            collection_name=f"subreddit_memory_{self.subreddit}",
            query_vector=vector,
            limit=limit
        )
        return [hit.payload for hit in results]
    
    def save_data(self, data: dict, filename: str):
        """Save scraped data to a JSON file."""
        os.makedirs("data", exist_ok=True)
        with open(f"data/{filename}", "w") as f:
            json.dump(data, f, indent=2)

    def preprocess_for_finetuning(self, user_actions: dict, subreddit_data: dict) -> dict:
        """Convert user actions into instruction-response pairs for fine-tuning, augmented with subreddit data."""
        user_datasets = {}
        for user, actions in user_actions.items():
            dataset = []
            for action in actions:
                if action["type"] in ["post", "comment"]:
                    thread_data = subreddit_data["threads"][action["thread_id"]]
                    parent_text = thread_data["submission"]["text"] if action["parent_id"] is None else next(
                        (c["text"] for c in thread_data["comments"] if c["id"] == action["parent_id"]), "")
                    similar_threads = self.retrieve_similar_threads(parent_text)
                    subreddit_context = "\n".join([t["text"] for t in similar_threads])
                    dataset.append({
                        "instruction": (
                            f"Respond to this post/comment in the subreddit r/{self.subreddit}: '{parent_text}'\n"
                            f"Subreddit context (similar threads):\n{subreddit_context}"
                        ),
                        "response": action["text"],
                        "metadata": {
                            "sentiment": self._analyze_sentiment(action["text"]),
                            "upvotes": action["upvotes"]
                        }
                    })
            user_datasets[user] = dataset
        return user_datasets
    
    def _analyze_sentiment(self, text: str) -> float:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

if __name__ == "__main__":
    scraper = RedditScraper(
        client_id="your_client_id",
        client_secret="your_client_secret",
        user_agent="your_user_agent",
        subreddit="AmItheAsshole"
    )
    # Example: Scrape a specific thread
    subreddit_data = scraper.scrape_subreddit("submission_id_here")
    scraper.save_data(subreddit_data, "subreddit_data.json")
    
    # # Example: Scrape a subreddit (historical data)
    # subreddit_data = scraper.scrape_subreddit("AskReddit", "2023-01-01", "2023-12-31")
    # scraper.save_data(subreddit_data, "subreddit_data.json")
    user_actions = scraper.group_by_user(subreddit_data)
    scraper.save_data(user_actions, "user_actions.json")

    # Preprocess for fine-tuning
    finetune_data = scraper.preprocess_for_finetuning(user_actions)
    for user, data in finetune_data.items():
        scraper.save_data(data, f"finetune_data_{user}.json")