# Filename: agent.py
import random
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from env import SubredditEnvironment
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RedditAgent:
    def __init__(self, user_id: str, real_actions: List[Dict], subreddit: str, model_path: str = None):
        self.user_id = user_id
        self.real_actions = sorted(real_actions, key=lambda x: x["timestamp"])  # User's real actions
        self.subreddit = subreddit
        self.short_term_memory: List[dict] = []
        self.qdrant = QdrantClient("localhost", port=6333)
        self.qdrant.recreate_collection(
            collection_name=f"user_memory_{user_id}",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = AutoModelForCausalLM.from_pretrained(model_path or "distilbert/distilgpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path or "distilbert/distilgpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.action_history: List[Dict] = []
        self.epsilon = 0.1  # For exploration
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._store_user_memory()

    def _store_user_memory(self):
        """Store user-specific interactions in Qdrant."""
        points = []
        for i, action in enumerate(self.real_actions):
            if "text" in action:
                vector = self.embedder.encode(action["text"]).tolist()
                points.append(models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "text": action["text"],
                        "type": action["type"],
                        "thread_id": action["thread_id"],
                        "sentiment": self._analyze_sentiment(action["text"])
                    }
                ))
        if points:
            self.qdrant.upsert(
                collection_name=f"user_memory_{self.user_id}",
                points=points
            )

    def observe(self, env: SubredditEnvironment, thread_id: str, timestamp: float):
        state = env.get_state_at_timestamp(thread_id, timestamp)
        self.short_term_memory = list(state["thread"].values())[-5:]  # Last 5 posts/comments
        return state

    def retrieve_subreddit_context(self, text: str, limit: int = 5) -> List[dict]:
        """Retrieve similar threads from subreddit memory."""
        from datascraper import RedditScraper
        scraper = RedditScraper("", "", "", self.subreddit)  # Credentials not needed for retrieval
        return scraper.retrieve_similar_threads(text, limit)

    def retrieve_user_memory(self, context: str) -> List[str]:
        """Retrieve user-specific interactions from memory."""
        vector = self.embedder.encode(context).tolist()
        results = self.qdrant.search(
            collection_name=f"user_memory_{self.user_id}",
            query_vector=vector,
            limit=3
        )
        return [hit.payload["text"] for hit in results]

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def select_action(self, state: Dict, thread_id: str, real_action: Optional[Dict] = None) -> Dict:
        context = " ".join([node["text"] for node in state["thread"].values()])
        subreddit_context = self.retrieve_subreddit_context(context)
        subreddit_context_text = "\n".join([t["text"] for t in subreddit_context])
        user_memory = self.retrieve_user_memory(context)
        user_memory_text = "\n".join(user_memory)
        thread_sentiment = state["sentiment"]

        if real_action and real_action["author"] == self.user_id:
            # If this is a real action by the user, aim to replicate it
            prompt = (
                f"Reddit thread context in r/{self.subreddit}: {context}\n"
                f"Subreddit context (similar threads):\n{subreddit_context_text}\n"
                f"User {self.user_id}'s past interactions:\n{user_memory_text}\n"
                f"Thread sentiment: {'positive' if thread_sentiment > 0 else 'negative' if thread_sentiment < 0 else 'neutral'}\n"
                f"Generate a response as user {self.user_id} that matches their style and the subreddit norms."
            )
            if random.random() < self.epsilon:
                action = {"action": real_action["type"], "text": real_action["text"], "parent_id": real_action["parent_id"]}
            else:
                content = self.generate_response(prompt)
                action = {"action": real_action["type"], "text": content, "parent_id": real_action["parent_id"]}
        else:
            # If the user did not act, decide whether to act based on subreddit norms and user tendencies
            prompt = (
                f"Reddit thread context in r/{self.subreddit}: {context}\n"
                f"Subreddit context (similar threads):\n{subreddit_context_text}\n"
                f"User {self.user_id}'s past interactions:\n{user_memory_text}\n"
                f"Thread sentiment: {'positive' if thread_sentiment > 0 else 'negative' if thread_sentiment < 0 else 'neutral'}\n"
                f"Choose an action: post, comment, upvote, downvote, ignore"
            )
            action_response = self.generate_response(prompt).strip().lower()
            if random.random() < self.epsilon:
                action_type = random.choice(["post", "comment", "upvote", "downvote", "ignore"])
            else:
                action_type = action_response.split("\n")[0] if "\n" in action_response else action_response
            
            if action_type in ["post", "comment"]:
                content_prompt = (
                    f"Generate a {action_type} for the thread context in r/{self.subreddit}: {context}\n"
                    f"Subreddit context (similar threads):\n{subreddit_context_text}\n"
                    f"User {self.user_id}'s past interactions:\n{user_memory_text}\n"
                    f"Ensure the response aligns with a {'positive' if thread_sentiment > 0 else 'negative' if thread_sentiment < 0 else 'neutral'} sentiment."
                )
                content = self.generate_response(content_prompt)
                action = {"action": action_type, "text": content, "parent_id": None if action_type == "post" else random.choice(list(state["thread"].keys()))}
            else:
                action = {"action": action_type}
        return action

    def update_memory(self, action: Dict, reward: float, real_response: Optional[str] = None):
        if reward > 0.5 and "text" in action:  # Store high-reward interactions
            vector = self.embedder.encode(action["text"]).tolist()
            self.qdrant.upsert(
                collection_name=f"user_memory_{self.user_id}",
                points=[models.PointStruct(
                    id=len(self.action_history),
                    vector=vector,
                    payload={
                        "text": action["text"],
                        "reward": reward,
                        "real_response": real_response or ""
                    }
                )]
            )
        self.action_history.append({"action": action, "reward": reward, "real_response": real_response or ""})

    def reflect(self, performance_threshold: float = 0.5):
        if len(self.action_history) < 1:
            return
        recent_history = self.action_history[-1:]
        avg_reward = np.mean([h["reward"] for h in recent_history])
        
        print(f"Agent {self.user_id} reflection:")
        print(f"  Avg reward: {avg_reward:.2f}")

        if avg_reward < performance_threshold:
            print("  Performance poor. Triggering fine-tuning...")
            self._trigger_finetuning(recent_history)

    def _trigger_finetuning(self, recent_history: List[Dict]):
        dataset = []
        for h in recent_history:
            if h["action"]["action"] in ["post", "comment"] and h["real_response"]:
                context = " ".join([node["text"] for node in self.short_term_memory])
                dataset.append({
                    "instruction": f"Respond to this thread context in r/{self.subreddit}: {context}",
                    "response": h["real_response"],
                    "metadata": {"reward": h["reward"]}
                })
        if dataset:
            from finetune import RedditFinetuner
            finetuner = RedditFinetuner(model_name=self.model.config._name_or_path)
            finetuner_dataset = finetuner.load_data("data/temp_finetune_data.json")
            finetuner.fine_tune(finetuner_dataset, output_dir=f"finetuned_model_{self.user_id}")
            self.model = AutoModelForCausalLM.from_pretrained(f"finetuned_model_{self.user_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(f"finetuned_model_{self.user_id}")

    def act(self, env: SubredditEnvironment, thread_id: str, real_action: Optional[Dict] = None) -> float:
        state = self.observe(env, thread_id, real_action["timestamp"] if real_action else time.time())
        action = self.select_action(state, thread_id, real_action)
        
        if action["action"] in ["post", "comment"]:
            post_id = env.add_post(thread_id, action["text"], self.user_id, action["parent_id"])
            action["post_id"] = post_id
            reward = self._calculate_reward(action, real_action, state) if real_action else 1.0  # Default reward if no real action
        elif action["action"] == "upvote":
            env.update_votes(thread_id, random.choice(list(state["thread"].keys())), 1, 0)
            reward = 1.0 if real_action and real_action["type"] == "upvote" else -1.0
        elif action["action"] == "downvote":
            env.update_votes(thread_id, random.choice(list(state["thread"].keys())), 0, 1)
            reward = 1.0 if real_action and real_action["type"] == "downvote" else -1.0
        else:  # ignore
            reward = 1.0 if real_action is None or real_action["type"] == "ignore" else -1.0

        self.update_memory(action, reward, real_action["text"] if real_action and "text" in real_action else None)
        self.reflect()
        return reward

    def _calculate_reward(self, action: Dict, real_action: Dict, state: Dict) -> float:
        if action["action"] != real_action["type"]:
            return -1.0  # Penalty for wrong action type
        if "text" not in action or "text" not in real_action:
            return 0.0
        action_vector = self.embedder.encode(action["text"]).tolist()
        real_vector = self.embedder.encode(real_action["text"]).tolist()
        similarity = np.dot(action_vector, real_vector) / (np.linalg.norm(action_vector) * np.linalg.norm(real_vector))
        action_sentiment = self.sentiment_analyzer.polarity_scores(action["text"])["compound"]
        real_sentiment = self.sentiment_analyzer.polarity_scores(real_action["text"])["compound"]
        sentiment_alignment = 1 - abs(action_sentiment - real_sentiment)
        subreddit_context = self.retrieve_subreddit_context(" ".join([node["text"] for node in state["thread"].values()]))
        subreddit_sentiment = np.mean([t["sentiment"] for t in subreddit_context])
        subreddit_alignment = 1 - abs(action_sentiment - subreddit_sentiment)
        return similarity + sentiment_alignment + subreddit_alignment

    def _analyze_sentiment(self, text: str) -> float:
        return TextBlob(text).sentiment.polarity