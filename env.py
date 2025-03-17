import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models
from textblob import TextBlob
import json
# Reddit thread node
@dataclass
class ThreadNode:
    id: int
    text: str
    author_id: str
    parent_id: Optional[int]
    upvotes: int = 0
    downvotes: int = 0
    timestamp: float = time.time()

# Environment simulating a Reddit thread
class SubredditEnvironment:
    def __init__(self, subreddit_data: dict = None, subreddit: str = "AmItheAsshole"):
        self.threads: Dict[str, Dict[int, ThreadNode]] = {}
        self.next_ids: Dict[str, int] = {}
        self.real_actions: Dict[str, List[Dict]] = {}
        self.subreddit = subreddit
        if subreddit_data:
            self.load_subreddit_data(subreddit_data)

    # Load a real thread from a JSON file
    def load_subreddit_data(self, subreddit_data: dict):
        """Load subreddit data into the environment and store actions."""
        for thread_id, thread_data in subreddit_data["threads"].items():
            self.threads[thread_id] = {}
            self.next_ids[thread_id] = 0
            self.real_actions[thread_id] = []
            submission = thread_data["submission"]
            self.add_post(thread_id, submission["text"], submission["author"], None, 
                         submission["upvotes"], submission["downvotes"], submission["timestamp"])
            self.real_actions[thread_id].append({
                "type": "post",
                "text": submission["text"],
                "author": submission["author"],
                "parent_id": None,
                "timestamp": submission["timestamp"]
            })
            for comment in thread_data["comments"]:
                self.add_post(thread_id, comment["text"], comment["author"], comment["parent_id"],
                             comment["upvotes"], comment["downvotes"], comment["timestamp"])
                self.real_actions[thread_id].append({
                    "type": "comment",
                    "text": comment["text"],
                    "author": comment["author"],
                    "parent_id": comment["parent_id"],
                    "timestamp": comment["timestamp"]
                })
            self.real_actions[thread_id].sort(key=lambda x: x["timestamp"])

    def add_post(self, thread_id: str, text: str, author_id: str, parent_id: Optional[int] = None,
                 upvotes: int = 0, downvotes: int = 0, timestamp: float = time.time()) -> int:
        if thread_id not in self.threads:
            self.threads[thread_id] = {}
            self.next_ids[thread_id] = 0
        node = ThreadNode(self.next_ids[thread_id], text, author_id, parent_id, upvotes, downvotes, timestamp)
        self.threads[thread_id][self.next_ids[thread_id]] = node
        self.next_ids[thread_id] += 1
        return node.id

    def update_votes(self, thread_id: str, post_id: int, upvotes: int, downvotes: int):
        if thread_id in self.threads and post_id in self.threads[thread_id]:
            self.threads[thread_id][post_id].upvotes += upvotes
            self.threads[thread_id][post_id].downvotes += downvotes

    def get_state_at_timestamp(self, thread_id: str, timestamp: float) -> Dict:
        """Return the thread state up to a given timestamp."""
        if thread_id not in self.threads:
            return {"thread": {}, "sentiment": 0.0}
        filtered_thread = {k: v for k, v in self.threads[thread_id].items() if v.timestamp <= timestamp}
        return {
            "thread": {k: vars(v) for k, v in filtered_thread.items()},
            "sentiment": self._compute_sentiment(filtered_thread)
        }

    def _compute_sentiment(self, thread: Dict) -> float:
        texts = [node.text for node in thread.values()]
        if not texts:
            return 0.0
        return TextBlob(" ".join(texts)).sentiment.polarity
    
    def save_state(self, filename: str):
        state = {thread_id: self.get_state_at_timestamp(thread_id, time.time()) for thread_id in self.threads}
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)


