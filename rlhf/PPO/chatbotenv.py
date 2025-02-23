import numpy as np

class ChatbotEnv:
    def __init__(self, tokenizer, response_bank):
        """
        Args:
            tokenizer: Function that converts text queries into numerical form.
            response_bank: Dict mapping queries to expected responses.
        """
        self.tokenizer = tokenizer
        self.response_bank = response_bank

    def encode_query(self, query):
        """Convert a text query into a numerical representation."""
        return np.array(self.tokenizer(query))

    def get_reward(self, response, expected_response):
        """Assign a reward based on response correctness."""
        return 1.0 if response == expected_response else -1.0
