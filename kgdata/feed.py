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

if __name__ == "__main__":
    data = """
1. SBI Life - Smart Shield
----------------------------------
Policy Name: SBI Life - Smart Shield
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
Age at entry: Minimum - 18 years, Maximum - 60 years
Maximum age at Maturity: 80 years
Basic Sum Assured: Minimum - 20 Lakhs, Maximum - No Limit
Policy Term: 5 to 80 years
Premium amount: Depends on age, sum assured, and term chosen

2. SBI Life - eShield Next
----------------------------------
Policy Name: SBI Life - eShield Next
Policy Type: Pure Term Insurance
Key Features:
• Comprehensive online term plan
• Increasing cover options to match life stages
• Optional riders for critical illness and accidental death benefits
Benefits:
• Ensures financial security for loved ones
• Tax benefits under Sections 80C and 10(10D)
• Affordable and customizable coverage
Age at entry: Minimum - 18 years, Maximum - 65 years
Maximum age at Maturity: 85 years
Basic Sum Assured: Minimum - 50 Lakhs, Maximum - No Limit
Policy Term: 10 to 85 years
Premium amount: Varies based on coverage and age

3. SBI Life - Smart Scholar Plus
----------------------------------
Policy Name: SBI Life - Smart Scholar Plus
Policy Type: Child Plan
Key Features:
• Ensures financial planning for child's future
• Premium waiver benefit in case of policyholder's death
• Multiple fund options for investment
Benefits:
• Helps secure child's education and future goals
• Life cover during the policy term
• Tax benefits under applicable laws
Age at entry: Minimum - 18 years, Maximum - 57 years
Maximum age at Maturity: 65 years
Basic Sum Assured: Minimum - 1 Lakh, Maximum - No Limit
Policy Term: 8 to 25 years
Premium amount: Flexible as per chosen coverage

4. SBI Life - Retire Smart Plus
----------------------------------
Policy Name: SBI Life - Retire Smart Plus
Policy Type: Retirement/Annuity Plan
Key Features:
• Secures post-retirement financial independence
• Guaranteed additions throughout the policy term
• Option to receive annuities as per choice
Benefits:
• Ensures steady retirement income
• Tax-efficient retirement corpus accumulation
• Flexibility in premium payments
Age at entry: Minimum - 30 years, Maximum - 70 years
Maximum age at Maturity: 80 years
Basic Sum Assured: Minimum - 2 Lakhs, Maximum - No Limit
Policy Term: 10 to 40 years
Premium amount: Based on annuity plan selected

5. SBI Life - eWealth Plus
----------------------------------
Policy Name: SBI Life - eWealth Plus
Policy Type: Unit Linked Insurance Plan (ULIP)
Key Features:
• Combines wealth creation with life insurance
• No premium allocation charges
• Automatic asset allocation between equity and debt
Benefits:
• Helps grow investments with market-linked returns
• Provides life cover for financial security
• Tax benefits under prevailing tax laws
Age at entry: Minimum - 18 years, Maximum - 50 years
Maximum age at Maturity: 70 years
Basic Sum Assured: Minimum - 1.25 Lakhs, Maximum - 10 times annual premium
Policy Term: 10 to 30 years
Premium amount: Starts at 30,000 per annum

6. SBI Life - Smart Elite Plus
----------------------------------
Policy Name: SBI Life - Smart Elite Plus
Policy Type: Unit Linked Insurance Plan (ULIP)
Key Features:
• Wealth enhancement with life cover
• Choice of investment funds
• Option to increase or decrease sum assured
Benefits:
• Facilitates wealth accumulation
• Provides financial protection for family
• Tax advantages as per applicable laws
Age at entry: Minimum - 18 years, Maximum - 60 years
Maximum age at Maturity: 70 years
Basic Sum Assured: Minimum - 10 Lakhs, Maximum - No Limit
Policy Term: 5 to 30 years
Premium amount: Minimum 1 Lakh per annum

7. SBI Life - Smart Platina Supreme
----------------------------------
Policy Name: SBI Life - Smart Platina Supreme
Policy Type: Guaranteed Return Plan
Key Features:
• Guaranteed maturity benefits
• Flexible premium payment terms
• Life cover during policy term
Benefits:
• Ensures risk-free wealth growth
• Provides financial security
• Eligible for tax benefits
Age at entry: Minimum - 18 years, Maximum - 55 years
Maximum age at Maturity: 75 years
Basic Sum Assured: Minimum - 2 Lakhs, Maximum - No Limit
Policy Term: 10 to 30 years
Premium amount: Depends on sum assured

8. SBI Life - Smart Platina Plus
----------------------------------
Policy Name: SBI Life - Smart Platina Plus
Policy Type: Guaranteed Return Plan
Key Features:
• Assured returns at maturity
• Premium payment flexibility
• Life cover during the policy term
Benefits:
• Helps build a secure corpus
• Ensures family's financial well-being
• Tax benefits available
Age at entry: Minimum - 18 years, Maximum - 60 years
Maximum age at Maturity: 75 years
Basic Sum Assured: Minimum - 1 Lakh, Maximum - No Limit
Policy Term: 10 to 20 years
Premium amount: Varies by sum assured

9. SBI Life - Smart Platina Assure
----------------------------------
Policy Name: SBI Life - Smart Platina Assure
Policy Type: Savings Plan
Key Features:
• Guaranteed maturity benefits
• Life cover throughout the policy term
• Regular payouts after maturity
Benefits:
• Ensures safe savings growth
• Provides financial protection for dependents
• Offers tax benefits
Age at entry: Minimum - 18 years, Maximum - 60 years
Maximum age at Maturity: 75 years
Basic Sum Assured: Minimum - 2 Lakhs, Maximum - No Limit
Policy Term: 8 to 20 years
Premium amount: Customizable based on plan

10. SBI Life - Smart Annuity Plus
----------------------------------
Policy Name: SBI Life - Smart Annuity Plus
Policy Type: Annuity Plan
Key Features:
• Multiple annuity options
• Lifetime guaranteed income
• Flexibility to choose payout frequency
Benefits:
• Provides steady post-retirement income
• Helps maintain lifestyle after retirement
• Tax benefits as per regulations
Age at entry: Minimum - 40 years, Maximum - 80 years
Maximum age at Maturity: Lifetime
Basic Sum Assured: Based on annuity option
Policy Term: Lifetime
Premium amount: Decided by annuity plan

11. SBI Life - Smart Swadhan Supreme
----------------------------------
Policy Name: SBI Life - Smart Swadhan Supreme
Policy Type: Term Insurance
Key Features:
• Return of premium at maturity
• Fixed life cover for term duration
• Affordable premiums
Benefits:
• Ensures protection and savings
• Refund of premiums at maturity
• Tax benefits under 80C
Age at entry: Minimum - 18 years, Maximum - 60 years
Maximum age at Maturity: 75 years
Basic Sum Assured: Minimum - 5 Lakhs, Maximum - No Limit
Policy Term: 10 to 30 years
Premium amount: Depends on sum assured

12. SBI Life - Saral Jeevan Bima
----------------------------------
Policy Type: Term Insurance
Key Features:
• Simple and affordable pure protection plan
• Provides life cover without any maturity benefit
• No medical tests for lower sum assured
Benefits:
• Financial security for family in case of policyholder's demise
• Tax benefits under Sections 80C and 10(10D)
Age at Entry: 18 to 65 years
Maximum Age at Maturity: 70 years
Basic Sum Assured: Minimum - 5 Lakhs, Maximum - 25 Lakhs
Policy Term: 5 to 40 years
Premium Amount: Based on sum assured and age

13. SBI Life - Smart Swadhan Neo
----------------------------------
Policy Type: Term Insurance with Return of Premium
Key Features:
• Life cover with guaranteed return of premium on maturity
• Affordable premiums with flexible policy terms
• Option to enhance protection with riders
Benefits:
• Ensures both life protection and maturity benefit
• Tax benefits under Sections 80C and 10(10D)
Age at Entry: 18 to 55 years
Maximum Age at Maturity: 75 years
Basic Sum Assured: Minimum - 5 Lakhs, Maximum - No Limit
Policy Term: 10 to 30 years
Premium Amount: Based on sum assured and term

14. SBI Life - Smart Privilege Plus
----------------------------------
Policy Type: Unit Linked Insurance Plan (ULIP)
Key Features:
• Wealth creation along with life cover
• Multiple fund options with free switches
• Loyalty additions for long-term policyholders
Benefits:
• Combines investment growth with life insurance
• Tax benefits under Sections 80C and 10(10D)
Age at Entry: 8 to 60 years
Maximum Age at Maturity: 75 years
Basic Sum Assured: Minimum - 10 Lakhs, Maximum - No Limit
Policy Term: 10 to 30 years
Premium Amount: Based on fund selection and premium term

15. SBI Life - New Smart Samriddhi
----------------------------------
Policy Type: Endowment Plan
Key Features:
• Guaranteed returns with life cover
• Single premium and regular premium options
• Maturity benefit along with bonuses
Benefits:
• Helps achieve long-term financial goals
• Tax benefits under Sections 80C and 10(10D)
Age at Entry: 18 to 50 years
Maximum Age at Maturity: 65 years
Basic Sum Assured: Minimum - 1 Lakh, Maximum - No Limit
Policy Term: 10 to 25 years
Premium Amount: Based on sum assured and payment mode

16. SBI Life - Saral Pension
----------------------------------
Policy Type: Pension Plan
Key Features:
• Immediate annuity plan with lifelong income
• Multiple annuity options available
• Flexibility to choose annuity payout frequency
Benefits:
• Ensures regular income post-retirement
• Tax benefits under prevailing pension rules
Age at Entry: 40 to 80 years
Maximum Age at Maturity: Lifetime annuity
Basic Sum Assured: Minimum - 1 Lakh, Maximum - No Limit
Policy Term: Lifetime
Premium Amount: Based on selected annuity option
"""

    asyncio.run(process_document(data, "SBI Life Policies", chunk_size=500, chunk_overlap=100))