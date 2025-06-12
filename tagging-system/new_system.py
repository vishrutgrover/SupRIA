import os
import json
import spacy
import ast
from collections import Counter
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# LangChain Gemini model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    convert_system_message_to_human=True,
)

# Prompt template
prompt_template = PromptTemplate.from_template("""
You are an expert at user behavior profiling.
Given the topics: {topics}, generate a dictionary of behavior tags with weights (in snake_case).
Weights should reflect topic recency: recent topics → 8-10, older ones → 5-7.
Return only a valid Python dictionary. No extra text or comments.
""")

# Load previous tags
def load_previous_tags(file_path="previous_tags.json"):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as file:
        return json.load(file)

# Save updated tags
def save_updated_tags(tags, file_path="previous_tags.json"):
    with open(file_path, "w") as file:
        json.dump(tags, file, indent=4)

# Extract main topics from user queries
def extract_main_topics(user_queries, top_n=5):
    all_nouns = []
    for query in user_queries:
        doc = nlp(query)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                all_nouns.append(token.lemma_.lower())
    topic_counter = Counter(all_nouns)
    return [topic for topic, _ in topic_counter.most_common(top_n)]

import re

def generate_tags_with_weights(topics):
    prompt = prompt_template.format(topics=", ".join(topics))
    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    # Remove code block markdown if present
    if "```" in raw_text:
        raw_text = re.findall(r"```(?:python)?\n?(.*?)```", raw_text, re.DOTALL)
        raw_text = raw_text[0].strip() if raw_text else ""

    try:
        return ast.literal_eval(raw_text)
    except Exception as e:
        print("Error parsing LLM response:", e)
        print("Raw LLM Output:", response.content.strip())
        return {}


# Adjust tag weights
def adjust_weights_and_add_new_tags(previous_tags, new_tags, similarity_threshold=80, weight_threshold=2, max_weight=15):
    updated_tags = previous_tags.copy()

    for new_tag, weight in new_tags.items():
        matched_tag = None
        for prev_tag in previous_tags:
            if fuzz.ratio(new_tag, prev_tag) >= similarity_threshold:
                matched_tag = prev_tag
                break

        if matched_tag:
            updated_tags[matched_tag] = min(updated_tags[matched_tag] + weight, max_weight)
        else:
            updated_tags[new_tag] = min(weight, max_weight)

    for tag in list(updated_tags):
        if tag not in new_tags:
            updated_tags[tag] -= 1

    return {tag: wt for tag, wt in updated_tags.items() if wt >= weight_threshold}

# Main processing function
def process_user_queries(user_queries, previous_tags_file="previous_tags.json"):
    previous_tags = load_previous_tags(previous_tags_file)

    topics = extract_main_topics(user_queries)
    print("\nExtracted Topics:", topics)

    new_tags_with_weights = generate_tags_with_weights(topics)
    print("\nGenerated Tags with Weights:", new_tags_with_weights)

    updated_tags = adjust_weights_and_add_new_tags(previous_tags, new_tags_with_weights)
    save_updated_tags(updated_tags, previous_tags_file)

    return updated_tags

# Test run
if __name__ == "__main__":
    user_queries = [
        "Tell me about health insurance plans",
        "Benefits of critical illness cover",
        "Family floater medical insurance",
        "Coverage for hospital bills and emergencies",
    ]

    updated_tags = process_user_queries(user_queries)
    print("\nFinal Updated Tags with Weights:", updated_tags)
