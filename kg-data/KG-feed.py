import asyncio
import os
from datetime import datetime
from hashlib import md5
from dotenv import load_dotenv
from typing import Dict, List
import tiktoken
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel, Field


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(url=NEO4J_URI, 
                   username=NEO4J_USERNAME, 
                   password=NEO4J_PASSWORD, 
                   refresh_schema=False)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

construction_system = """Your role is to extract key elements and atomic facts from a given text.
1. Key Elements: Essential nouns, verbs, and adjectives pivotal to the text.
2. Atomic Facts: Smallest, indivisible facts in concise sentence format.
"""

construction_prompt = ChatPromptTemplate.from_messages([
    ("system", construction_system),
    ("human", "Extract information from the following input: {input}")
])

class AtomicFact(BaseModel):
    key_elements: List[str] = Field(description="List of key elements.")
    atomic_fact: str = Field(description="The atomic fact as a sentence.")

class Extraction(BaseModel):
    atomic_facts: List[AtomicFact] = Field(description="List of extracted atomic facts.")

# Google Gemini API Configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GEMINI_API_KEY")
)
structured_llm = llm.with_structured_output(Extraction)
construction_chain = construction_prompt | structured_llm

import_query = """
MERGE (d:Document {id:$document_name})
WITH d
UNWIND $data AS row
MERGE (c:Chunk {id: row.chunk_id})
SET c.text = row.chunk_text,
    c.index = row.index,
    c.document_name = row.document_name
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, row
UNWIND row.atomic_facts AS af
MERGE (a:AtomicFact {id: af.id})
SET a.text = af.atomic_fact
MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
WITH c, a, af
UNWIND af.key_elements AS ke
MERGE (k:KeyElement {id: ke})
MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
"""

def encode_md5(text):
    return md5(text.encode("utf-8")).hexdigest()

async def process_document(text, document_name, chunk_size=2000, chunk_overlap=200):
    start = datetime.now()
    print(f"Started extraction at: {start}")
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    print(f"Total text chunks: {len(texts)}")
    tasks = [
        asyncio.create_task(construction_chain.ainvoke({"input": chunk_text}))
        for index, chunk_text in enumerate(texts)
    ]
    results = await asyncio.gather(*tasks)
    print(f"Finished LLM extraction after: {datetime.now() - start}")
    docs = [el.model_dump() for el in results]
    for index, doc in enumerate(docs):
        doc['chunk_id'] = encode_md5(texts[index])
        doc['chunk_text'] = texts[index]
        doc['index'] = index
        for af in doc["atomic_facts"]:
            af["id"] = encode_md5(af["atomic_fact"])
    graph.query(import_query, params={"data": docs, "document_name": document_name})
    graph.query("""
    MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
    WHERE d.id = $document_name
    WITH c ORDER BY c.index WITH collect(c) AS nodes
    UNWIND range(0, size(nodes) -2) AS index
    WITH nodes[index] AS start, nodes[index + 1] AS end
    MERGE (start)-[:NEXT]->(end)
    """, params={"document_name": document_name})
    print(f"Finished import at: {datetime.now() - start}")

# Run the script
if __name__ == "__main__":
    data = """SBI Life Insurance Policies Overview

1. SBI Life - Smart Shield
----------------------------------
Policy Type: Term Insurance
Key Features:
• Affordable premiums with a high sum assured
• Two death benefit options: Level Cover and Increasing Cover
• Flexible policy terms (ranging from 5 to 80 years)
• Option to add riders such as critical illness protection
Benefits:
• Provides financial security for family members in case of untimely demise
• Tax benefits under Sections 80C and 10(10D) of the Income Tax Act
• Hassle-free online application and claim settlement process

2. SBI Life - eShield
----------------------------------
Policy Type: Pure Term Insurance
Key Features:
• Entirely online application and policy management process
• Competitive premium rates with substantial coverage
• Quick claim processing with minimal documentation
• Option to enhance coverage with additional riders for comprehensive protection
Benefits:
• Ensures financial safety for dependents with pure protection focus
• Attractive tax benefits as per prevailing tax regulations
• Streamlined procedures for ease of access and management

3. SBI Life - Smart Wealth Builder
----------------------------------
Policy Type: Endowment/Money Back Plan
Key Features:
• Dual advantage of protection and wealth accumulation
• Guaranteed maturity benefits along with participating bonuses
• Flexible premium payment options (single or regular premium modes)
• Option to receive benefits as a lump sum or as periodic payouts
Benefits:
• Helps in building a corpus for future financial goals such as education, marriage, or retirement
• Provides life cover during the policy term for added security
• Tax benefits under relevant income tax provisions

4. SBI Life - Smart Retire
----------------------------------
Policy Type: Retirement/Annuity Plan
Key Features:
• Designed specifically to secure financial independence post-retirement
• Offers a variety of annuity options tailored to different retirement needs
• Regular income stream guaranteed after retirement
• Flexible premium payment modes to suit individual financial planning
Benefits:
• Provides a stable and predictable income during retirement years
• Aids in managing post-retirement expenses and maintaining lifestyle quality
• Structurally tax-efficient to maximize retirement income

5. SBI Life - Smart Protect Plus (Additional Policy)
----------------------------------
Policy Type: Comprehensive Protection Plan
Key Features:
• Combines life coverage with additional benefits such as accidental death and disability cover
• Customizable cover amounts and flexible policy terms
• Optional riders for enhanced protection against critical illnesses
• Competitive premium rates with a focus on overall financial protection
Benefits:
• Offers a well-rounded safety net for unforeseen life events
• Helps policyholders manage unexpected financial burdens
• Provides tax benefits as per applicable sections of the Income Tax Act

These SBI Life Insurance policies are designed to cater to diverse financial needs and life stages. 
Whether the focus is on pure protection, wealth accumulation, or ensuring a secure retirement, 
SBI Life's product suite offers comprehensive solutions that blend affordability with robust benefits.
"""

    asyncio.run(process_document(data, "SBI Life Policies", chunk_size=500, chunk_overlap=100))