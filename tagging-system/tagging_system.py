import google.genai as genai
import spacy
import json
from fuzzywuzzy import fuzz
from collections import Counter
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

nlp = spacy.load("en_core_web_sm")
model = genai.Client(api_key=GEMINI_API_KEY).models


def load_previous_tags(
    file_path="previous_tags.json",
):
    with open(file_path, "r") as file:

        try:
            return json.load(file)
        except Exception as e:
            print("Exception:", e)


def save_updated_tags(tags, file_path="previous_tags.json"):
    with open(file_path, "w") as file:
        json.dump(tags, file, indent=4)


def extract_main_topics(user_queries):
    all_nouns = []

    for query in user_queries:
        doc = nlp(query)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                all_nouns.append(token.lemma_.lower())

    topic_counter = Counter(all_nouns)
    main_topics = [topic for topic, _ in topic_counter.most_common(5)]
    return main_topics


def generate_dynamic_tags_with_weights(topics):
    prompt = f"""You are an expert at user behavior profiling.
Based on these topics: {', '.join(topics)}, suggest a list of behavior tags with weights.
The weights should reflect the importance of these tags, with recent topics assigned higher weights (8, 9, 10) and older topics assigned lower weights (5, 6, 7). 
Provide the tags in snake_case format with their respective weights as a dictionary and dont add any comments."""

    response = model.generate_content(model="gemini-2.0-flash", contents=prompt)

    candidates = response.candidates
    content_text = candidates[0].content.parts[0].text

    try:

        content_text = (
            content_text.strip().split("```python\n")[-1].split("\n```")[0].strip()
        )
        tags_with_weights = eval(content_text)
    except Exception as e:
        print("Error parsing Gemini response:", e)
        tags_with_weights = {}

    return tags_with_weights


def adjust_weights_and_add_new_tags(
    previous_tags, new_tags, similarity_threshold=80, weight_threshold=2, max_weight=15
):
    updated_tags = previous_tags.copy()

    for new_tag, weight in new_tags.items():

        matched_tag = None
        for prev_tag in previous_tags:
            if fuzz.ratio(new_tag, prev_tag) >= similarity_threshold:
                matched_tag = prev_tag
                break

        if matched_tag:
            updated_tags[matched_tag] = min(
                updated_tags.get(matched_tag, 0) + weight, max_weight
            )
        else:
            updated_tags[new_tag] = min(weight, max_weight)

    for tag in list(updated_tags.keys()):
        if tag not in new_tags:
            updated_tags[tag] -= 1

    updated_tags = {
        tag: weight
        for tag, weight in updated_tags.items()
        if weight >= weight_threshold
    }

    return updated_tags


def process_user_queries(
    user_queries,
    previous_tags_file=r"D:\RAJ ARYAN\Codec\SupRIA\tagging-system\previous_tags.json",
):
    previous_tags = load_previous_tags(previous_tags_file)

    topics = extract_main_topics(user_queries)
    print("Extracted Topics:", topics)

    new_tags_with_weights = generate_dynamic_tags_with_weights(topics)
    print("Generated Tags with Weights:", new_tags_with_weights)

    updated_tags = adjust_weights_and_add_new_tags(previous_tags, new_tags_with_weights)

    save_updated_tags(updated_tags)

    return updated_tags


user_queries = [
    "Tell me about health insurance plans",
    "Benefits of critical illness cover",
    "Family floater medical insurance",
    "Coverage for hospital bills and emergencies",
]

updated_tags = process_user_queries(user_queries)
print("Updated Tags with Weights:", updated_tags)
