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
You are an expert user-behavior profiler for an insurance chatbot.  
Given a list of topics: {topics} in order from most recent (first) to oldest (last), produce *only* a valid Python dictionary literal mapping each behavior tag (snake_case) to an integer weight (5–10).  

*Rules:*  
1. *Recency tiers:*  
   - Top 1-2 topics → weights 9-10  
   - Next 2-3 topics → weights 7-8  
   - Remaining topics → weights 5-6  
2. *Tag naming:*  
   - Convert each topic to a concise tag by extracting its core noun(s).  
   - Optionally append “_focused”, “_aware”, or “_planner” to clarify intent.  
   - All tags must be lowercase snake_case, with no spaces, punctuation, or comments.  
3. *Deduplication:*  
   - If two topics yield the same tag (e.g., “health insurance” vs. “healthcare”), merge into one entry with the higher weight.  
4. *Formatting:*  
   - *Only* output a single-line (or minimal multi-line) Python dict literal 
   - Do *not* include any markdown, code fences, commentary, or extra text.
""")

# Load previous tags
def load_previous_tags(file_path="..\Django-Server\static\previous_tags.json"):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as file:
        return json.load(file)

# Save updated tags
def save_updated_tags(tags, file_path="..\Django-Server\static\previous_tags.json"):
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
def adjust_weights_and_add_new_tags(previous_tags, new_tags, similarity_threshold=50, weight_threshold=5, max_weight=10):
    updated_tags = previous_tags.copy()
    refreshed = set()

    for new_tag, weight in new_tags.items():
        matched_tag = None
        for prev_tag in previous_tags:
            if fuzz.ratio(new_tag, prev_tag) >= similarity_threshold:
                matched_tag = prev_tag
                break

        if matched_tag:
            updated_tags[matched_tag] = max(updated_tags[matched_tag] , weight)
            refreshed.add(matched_tag)
        else:
            updated_tags[new_tag] = min(weight, max_weight)
            refreshed.add(new_tag)

    for tag in list(updated_tags):
        if tag not in refreshed:
            updated_tags[tag] -= 1
    updated_tags = {t: w for t, w in updated_tags.items() if w >= weight_threshold}

    return updated_tags

# Main processing function
def process_user_queries(user_queries, previous_tags_file="..\Django-Server\static\previous_tags.json"):
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
    # user_queries = [
    #     "Benefits of a pension fund for retirement",
    #     "What are the features of a retirement annuity?",
    #     "Tell me about the different retirement income options",
    #     "Advantages of starting retirement planning early",
    #     "How do retirement corpus withdrawal rules work?"
    # ]
    user_queries = [
        "What are the benefits of a health insurance plan?",
    "Which features should I look for in a health policy?",
    "How does cashless hospitalization work in health insurance?",
    "What is the difference between individual and family floater health plans?",
    "Explain the waiting period in health insurance policies",
    "Can I claim health insurance for pre-existing diseases?",
    "Is maternity coverage included in standard health plans?",
    "What are the tax benefits of buying a health insurance policy?",
    "How do top-up health plans work?",
    "What documents are required to claim medical expenses under a health plan?"
    ]

    updated_tags = process_user_queries(user_queries)
    print("\nFinal Updated Tags with Weights:", updated_tags)
