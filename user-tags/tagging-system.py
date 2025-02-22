import spacy
import json
import subprocess
import sys
from typing import List, Set, Dict

def install_spacy_model(model_name="en_core_web_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)

install_spacy_model()
nlp = spacy.load("en_core_web_sm")

TAGS_FILE = "user_tags.json"

def load_user_tags() -> Dict[str, List[str]]:
    try:
        with open(TAGS_FILE, "r") as file:
            data = json.load(file)
            if isinstance(data, dict) and "recent" in data and "older" in data:
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"recent": [], "older": []}

def save_user_tags(recent: List[str], older: List[str]):
    with open(TAGS_FILE, "w") as file:
        json.dump({"recent": recent, "older": older}, file)

tags_data = load_user_tags()
recent_tags: List[Set[str]] = [set(tags.split(", ")) for tags in tags_data["recent"]]
older_tags: Set[str] = set(tags_data["older"])

def extract_tags(message: str) -> Set[str]:
    doc = nlp(message)
    tags = set()
    
    for ent in doc.ents:
        if ent.label_ not in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
            tags.add(ent.text.lower())

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha and not token.is_stop:
            tags.add(token.lemma_.lower())

    return tags

def process_message(message: str):
    global recent_tags, older_tags

    new_tags = extract_tags(message)

    if len(recent_tags) == 3:
        older_tags.update(recent_tags.pop(0))

    recent_tags.append(new_tags)
    save_user_tags([", ".join(tags) for tags in recent_tags], list(older_tags))

    return f"Recent tags: {recent_tags}\nOlder tags: {older_tags}"

if __name__ == "__main__":
    messages = [
        "I am interested in a term insurance policy.",
        "Tell me about investment plans and retirement benefits.",
        "Do you offer health insurance?",
        "What are the best investment options for 2025?",
        "I want to apply for a car loan and personal loan."
    ]
    
    for msg in messages:
        print(process_message(msg))