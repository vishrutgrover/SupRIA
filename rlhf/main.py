import numpy as np
import torch as T
from chatbotenv import ChatbotEnv
from ppoAgent import PPOAgent

def simple_tokenizer(text):
    return np.array([ord(c) for c in text[:50]]) 

response_bank = {
    "What is my policy coverage?": "Your policy covers health and accidents.",
    "How can I renew my policy?": "You can renew your policy online via our portal.",
    "What is my premium amount?": "Your premium amount is 50,000 per month."
}

env = ChatbotEnv(tokenizer=simple_tokenizer, response_bank=response_bank)

n_actions = len(response_bank)  
input_dims = 50  
ppo_agent = PPOAgent(n_actions, input_dims)

num_episodes = 500

for episode in range(num_episodes):
    for query, expected_response in response_bank.items():
        state = env.encode_query(query) 
        action, prob, val = ppo_agent.choose_action(state) 

        chosen_response = list(response_bank.values())[action]

        reward = env.get_reward(chosen_response, expected_response)
        done = True  

        # Store in memory
        ppo_agent.remember(state, action, prob, val, reward, done)


    ppo_agent.learn()

    if episode % 50 == 0:
        print(f"Episode {episode}: Training in progress...")


ppo_agent.save_models()

print("Training complete. PPO chatbot model saved!")

# Chatbot interaction (Testing phase)
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    state = env.encode_query(user_input)
    action, _, _ = ppo_agent.choose_action(state)
    bot_response = list(response_bank.values())[action]

    print(f"Bot: {bot_response}")
