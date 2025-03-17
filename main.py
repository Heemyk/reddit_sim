# Filename: main.py
from env import SubredditEnvironment
from agent import RedditAgent
from datascraper import RedditScraper
import json
import random
import os
from dotenv import load_dotenv

load_dotenv()
client_id=os.getenv("CLIENT_ID")
client_secret=os.getenv("CLIENT_SECRET")
user_agent=os.getenv("USER_AGENT")

def load_split_json(base_filename: str) -> dict:
    """Load split JSON files into a single dict."""
    data = {}
    i = 0
    while os.path.exists(f"data/{base_filename}_part{i}.json"):
        with open(f"data/{base_filename}_part{i}.json", "r") as f:
            chunk = json.load(f)
            if "threads" in chunk:
                data["threads"] = data.get("threads", {}) | chunk["threads"]
            else:
                data.update(chunk)
        i += 1
    if i == 0:  # No split files, try single file
        if os.path.exists(f"data/{base_filename}.json"):
            with open(f"data/{base_filename}.json", "r") as f:
                data = json.load(f)
    return data

def main():
    subreddit = "AmItheAsshole"
    data_dir = "data"
    dont_scrape = True # set to true once we have data
    max_agents = 20
    max_threads = 5
    
    # Scrape subreddit data
    scraper = RedditScraper(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        subreddit=subreddit
    )

    subreddit_data_file = "subreddit_data"
    user_actions_file = "user_actions"

    if (os.path.exists(f"{data_dir}/{subreddit_data_file}.json") or os.path.exists(f"{data_dir}/{subreddit_data_file}_part0.json")) and dont_scrape:
        print(f"Loading existing subreddit data from {data_dir}/{subreddit_data_file}...")
        subreddit_data = load_split_json(subreddit_data_file)
    else:
        print("No existing subreddit data found. Scraping new data...")
        subreddit_data = scraper.scrape_subreddit("2023-01-01", "2023-12-31")
        scraper.save_data(subreddit_data, f"{subreddit_data_file}.json")

    if (os.path.exists(f"{data_dir}/{user_actions_file}.json") or os.path.exists(f"{data_dir}/{user_actions_file}_part0.json")) and dont_scrape:
        print(f"Loading existing user actions from {data_dir}/{user_actions_file}...")
        user_actions = load_split_json(user_actions_file)
    else:
        print("No existing user actions found. Generating from subreddit data...")
        user_actions = scraper.group_by_user(subreddit_data)
        scraper.save_data(user_actions, f"{user_actions_file}.json")

    # subreddit_data = scraper.scrape_subreddit("2023-01-01", "2023-12-31")
    # scraper.save_data(subreddit_data, "subreddit_data.json")

    # # Group actions by user
    # print("Grouping actions by user")
    # user_actions = scraper.group_by_user(subreddit_data)
    # scraper.save_data(user_actions, "user_actions.json")
    
    # # Store subreddit memory
    # print(f"Storing memory")
    # scraper.store_subreddit_memory(subreddit_data)
    
    # Store subreddit memory (only if not already stored or if new data)
    if not scraper.qdrant.count(collection_name=f"subreddit_memory_{subreddit}").count:
        print("Storing subreddit memory in Qdrant...")
        scraper.store_subreddit_memory(subreddit_data, batch_size=500)
    else:
        print("Subreddit memory already exists in Qdrant. Skipping storage.")

    # Limit user_actions to max_agents
    if len(user_actions) > max_agents:
        selected_users = random.sample(list(user_actions.keys()), max_agents)
        user_actions = {user: user_actions[user] for user in selected_users}

    # Initialize environment with subreddit data
    print("Initializing environment")
    env = SubredditEnvironment(subreddit_data, subreddit)
    print("Environment initialized")

    # Print number of agents to be created
    print(f"Number of agents to be created: {len(user_actions)}")

    # Initialize agents for each user
    agents = {}
    for user, actions in user_actions.items():
        model_path = f"finetuned_model_{user}" if os.path.exists(f"finetuned_model_{user}") else None
        print(f"Initialising agents for user {user}")
        agents[user] = RedditAgent(user, actions, subreddit, model_path)

    print(env.real_actions)
    # Simulate the subreddit by replaying real actions and adding new interactions
    for thread_id in env.real_actions:
        print(f"\nSimulating thread {thread_id}:")
        real_actions = env.real_actions[thread_id]
        for i, real_action in enumerate(real_actions):
            user = real_action["author"]
            if user not in agents:
                continue
            agent = agents[user]
            print(f"\nStep {i + 1} (User {user}):")
            reward = agent.act(env, thread_id, real_action)
            print(f"Agent {user} acted, reward: {reward}")
            print(f"Real action: {real_action}")
            print(f"Agent action: {agent.action_history[-1]['action']}")
        
        # Simulate additional interactions by randomly selecting users to act
        for _ in range(3):  # Add 3 simulated interactions per thread
            user = random.choice(list(agents.keys()))
            agent = agents[user]
            print(f"\nSimulated interaction (User {user}):")
            reward = agent.act(env, thread_id)
            print(f"Agent {user} acted, reward: {reward}")
            print(f"Agent action: {agent.action_history[-1]['action']}")

    env.save_state("simulated_subreddit.json")
    print("Simulation saved to simulated_subreddit.json")

if __name__ == "__main__":
    main()