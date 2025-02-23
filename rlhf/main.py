import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=0.2):
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, log_probs, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)

        new_probs = self.policy(states)
        dist = Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        advantage = discounted_rewards - discounted_rewards.mean()
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(surrogate1, surrogate2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
class ChatbotEnv:
    def __init__(self, tokenizer, responses):
        self.tokenizer = tokenizer
        self.responses = responses

    def encode_query(self, query):
        return np.array(self.tokenizer(query))

    def get_reward(self, response, expected_response):
        return 1 if response == expected_response else -1

if __name__ == "__main__":
    def dummy_tokenizer(text):
        return [ord(c) for c in text][:10]  

    chatbot_env = ChatbotEnv(dummy_tokenizer, {"Hello": "Hi!", "How are you?": "I am fine."})
    agent = PPOAgent(input_dim=10, output_dim=2)
    
    
    for epoch in range(100):
        state = chatbot_env.encode_query("Hello")
        action, log_prob = agent.get_action(state)
        reward = chatbot_env.get_reward("Hi!", "Hi!")
        agent.update([state], [action], [log_prob], [reward])

    print("Training complete.")
