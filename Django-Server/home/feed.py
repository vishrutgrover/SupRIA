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

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    refresh_schema=False,
)

graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

construction_system = """Your role is to extract key elements and atomic facts from a given text.
1. Key Elements: Essential nouns, verbs, and adjectives pivotal to the text.
2. Atomic Facts: Smallest, indivisible facts in concise sentence format.
"""

construction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", construction_system),
        ("human", "Extract information from the following input: {input}"),
    ]
)


class AtomicFact(BaseModel):
    key_elements: List[str] = Field(description="List of key elements.")
    atomic_fact: str = Field(description="The atomic fact as a sentence.")


class Extraction(BaseModel):
    atomic_facts: List[AtomicFact] = Field(
        description="List of extracted atomic facts."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GEMINI_API_KEY"),
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
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
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
        doc["chunk_id"] = encode_md5(texts[index])
        doc["chunk_text"] = texts[index]
        doc["index"] = index
        for af in doc["atomic_facts"]:
            af["id"] = encode_md5(af["atomic_fact"])
    graph.query(import_query, params={"data": docs, "document_name": document_name})
    graph.query(
        """
    MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
    WHERE d.id = $document_name
    WITH c ORDER BY c.index WITH collect(c) AS nodes
    UNWIND range(0, size(nodes) -2) AS index
    WITH nodes[index] AS start, nodes[index + 1] AS end
    MERGE (start)-[:NEXT]->(end)
    """,
        params={"document_name": document_name},
    )
    print(f"Finished import at: {datetime.now() - start}")


if __name__ == "__main__":
    data = """
    [
        {
            "Plan Name": "SBI Life – eShield Next",
            "Plan Type": "Term Plan",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "50 years",
            "Maturity Age (Min)": "18 years",
            "Maturity Age (Max)": "65 years",
            "Policy Term Options": "Pure protection (up to 35 years, subject to life assured age)",
            "Premium Payment Term Options": "Yearly, Half-yearly, Quarterly, Monthly",
            "Minimum Premium Amount": "Rs. 3,600 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Multiple plan options to suit your needs; enhanced protection with optional Accident Benefit Rider; Better-Half Benefit (sum assured paid to spouse if insured dies without policy loans); choice of death benefit payment mode.",
            "Key Features": "A new-age online pure term plan designed to meet both present and future protection needs.",
            "Sum Assured Options": "Level or increasing cover (basic sum assured typically higher of 10× annual premium or 105% of total premiums paid).",
            "Riders Available": "Optional Accident Death & Total and Permanent Disability Rider.",
            "Surrender Value": "Nil (pure term plan; no surrender benefit)",
            "Eligibility Criteria": "Life assured age 18–50 years; sum assured as per IRDAI norms.",
            "USPs": "Instant online term cover with flexible options and Better-Half benefit.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – eShield Insta",
            "Plan Type": "Term Plan (Digital)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "25 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "15–30 years (subject to entry and maturity ages)",
            "Premium Payment Term Options": "Single, Regular, Limited (as per variant)",
            "Minimum Premium Amount": "Rs. 2,259 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Choice of two benefit options (level or increasing sum assured); easy digital enrollment with instant issue; full life cover throughout term.",
            "Key Features": "A 100% online term insurance plan for quick, hassle-free purchase and instant approval.",
            "Sum Assured Options": "Level or increasing term cover as per chosen option.",
            "Riders Available": "Standard riders like Accidental Death & Disability may be available separately.",
            "Surrender Value": "Nil (pure term plan)",
            "Eligibility Criteria": "Life assured age 18–65 years; healthy status required for online issuance.",
            "USPs": "Fully digital term insurance with instant issuance and affordable premiums.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Saral Jeevan Bima",
            "Plan Type": "Term Plan (Standard)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "50 years",
            "Maturity Age (Min)": "25 years",
            "Maturity Age (Max)": "65 years",
            "Policy Term Options": "10–30 years (limited by maturity age)",
            "Premium Payment Term Options": "Regular (equal to policy term), Single-pay",
            "Minimum Premium Amount": "Rs. 1,415 per annum",
            "Maximum Premium Amount": "Rs. 1,01,025 per annum",
            "Key Benefits": "Simple term cover at affordable cost; sum assured payable on death; flexible premium payment options including Single Pay and Limited Pay.",
            "Key Features": "No-frills plan with straightforward terms and multiple premium terms to suit affordability.",
            "Sum Assured Options": "Level sum assured; death benefit equals basic sum assured.",
            "Riders Available": "Optional riders like Accidental Death Benefit (per norms).",
            "Surrender Value": "Nil (pure term plan)",
            "Eligibility Criteria": "Life assured age 18–50 years; subject to age constraints.",
            "USPs": "Mass-affordable term plan with easy terms and flexible pay mode.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Shield Premier",
            "Plan Type": "Term Plan (Enhanced)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "25 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "10–25 years (subject to ages)",
            "Premium Payment Term Options": "Regular, Limited, Single Pay",
            "Minimum Premium Amount": "Rs. 9,500 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "High coverage with Level or Increasing Cover; financial protection at affordable premium; optional Accident Benefit Rider.",
            "Key Features": "Exclusive plan offering higher sum assured limits and flexible payment terms.",
            "Sum Assured Options": "Level or increasing (10% annual increase up to max).",
            "Riders Available": "Accidental Death & Total Permanent Disability Rider (optional).",
            "Surrender Value": "Nil (pure term plan)",
            "Eligibility Criteria": "Life assured age 18–65 years; sum assured per IRDAI guidelines.",
            "USPs": "Higher coverage and flexible options compared to standard plans.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Shield",
            "Plan Type": "Term Plan",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "23 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "10–25 years (subject to ages)",
            "Premium Payment Term Options": "Regular",
            "Minimum Premium Amount": "Rs. 3,000 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Basic term cover; choice of Level or Increasing cover; affordable premiums.",
            "Key Features": "Simple assurance plan with optional Level or Increasing cover.",
            "Sum Assured Options": "Level or increasing; death benefit equals basic sum assured.",
            "Riders Available": "Optional Accidental Death & Disability Rider.",
            "Surrender Value": "Nil (pure term plan)",
            "Eligibility Criteria": "Life assured age 18–65 years; standard underwriting applies.",
            "USPs": "Low entry premium with flexible benefit options.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Swadhan Neo",
            "Plan Type": "Term Plan with Return of Premium",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "23 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "15–20 years (subject to ages)",
            "Premium Payment Term Options": "Regular, Single Pay",
            "Minimum Premium Amount": "Rs. 5,000 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Protection plus return of premium at maturity; sum assured on death and premiums back on survival.",
            "Key Features": "Refund of all premiums if insured survives term.",
            "Sum Assured Options": "Level cover; death benefit equals basic sum assured; maturity benefit equals total premiums paid.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "Refund of fund value after 5 years.",
            "Eligibility Criteria": "Life assured age 18–65 years; subject to underwriting.",
            "USPs": "Combines risk cover with savings; unique in its category.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Swadhan Supreme",
            "Plan Type": "Term Plan with Return of Premium",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "23 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "20–25 years (ensures maturity by 70)",
            "Premium Payment Term Options": "Regular",
            "Minimum Premium Amount": "Rs. 6,000 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Affordable protection with full return of premium at maturity; sum assured on death.",
            "Key Features": "Refund of all premiums paid upon maturity.",
            "Sum Assured Options": "Level cover; death benefit equals basic sum assured; maturity benefit equals total premiums paid.",
            "Riders Available": "Optional Accident Benefit.",
            "Surrender Value": "Refund of fund value after 5 years.",
            "Eligibility Criteria": "Life assured age 18–65 years; subject to underwriting.",
            "USPs": "Provides protection plus guaranteed return of premiums.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Saral Swadhan Supreme",
            "Plan Type": "Return of Premium Endowment",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "60 years",
            "Maturity Age (Min)": "18 years",
            "Maturity Age (Max)": "65 years",
            "Policy Term Options": "20 years",
            "Premium Payment Term Options": "Regular",
            "Minimum Premium Amount": "Rs. 8,050 per annum",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Protection plus full refund of premiums on maturity; death benefit payable if death occurs before maturity.",
            "Key Features": "Endowment plan with return of all premiums at 20-year maturity.",
            "Sum Assured Options": "Level cover; death benefit equals basic sum assured; maturity benefit equals total premiums paid.",
            "Riders Available": "None.",
            "Surrender Value": "Payable after 5 policy years as per plan rules.",
            "Eligibility Criteria": "Life assured age 18–60 years; sum assured per IRDAI norms.",
            "USPs": "Simple endowment with guaranteed return of premiums; minimal underwriting.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/protection-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Platina Plus",
            "Plan Type": "Guaranteed Income Plan (Traditional Savings)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "60 years",
            "Maturity Age (Min)": "65 years",
            "Maturity Age (Max)": "95 years",
            "Policy Term Options": "20, 25, 30 years",
            "Premium Payment Term Options": "Regular, Single Pay",
            "Minimum Premium Amount": "Rs. 50,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed yearly income during payout and return of 110% of total premiums at maturity; bonuses declared.",
            "Key Features": "Participating savings plan providing fixed income post-retirement and maturity payouts.",
            "Sum Assured Options": "110% of total premiums at maturity; increasing life cover initially.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "30% of total premiums after 3 years, increasing thereafter.",
            "Eligibility Criteria": "Life assured age 18–60 years; minimum maturity age 65.",
            "USPs": "Long-term savings with guaranteed income and bonus additions.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/savings-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Platina Assure",
            "Plan Type": "Traditional Participating Endowment",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "23 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "10, 15, 20 years",
            "Premium Payment Term Options": "Limited Pay (5 or 7 years), Single Pay",
            "Minimum Premium Amount": "Rs. 50,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Flexibility in premium term; life cover with risk and savings benefits; loyalty additions on maturity.",
            "Key Features": "Participating savings with guaranteed additions and flexible pay options.",
            "Sum Assured Options": "10× annual premium for pay plans; 1.25× single premium for single pay.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "Guaranteed surrender after 2 years plus bonuses.",
            "Eligibility Criteria": "Life assured age 18–65 years; term per maturity age chosen.",
            "USPs": "Guaranteed additions with flexible premium payment.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/savings-plans"
        },
        {
            "Plan Name": "SBI Life – New Smart Samriddhi",
            "Plan Type": "Non-Participating Savings Plan",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "28 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "10, 15, 20, 25, 30 years",
            "Premium Payment Term Options": "Regular, Limited Pay, Single Pay",
            "Minimum Premium Amount": "Rs. 25,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed loyalty additions; return of premium on survival; life cover during premium term.",
            "Key Features": "Traditional non-participating with fixed loyalty additions and single premium option.",
            "Sum Assured Options": "125% of single premium; multiples of annual premium.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "Guaranteed value based on premiums paid after 2 years.",
            "Eligibility Criteria": "Life assured age 18–65 years; term up to age 70.",
            "USPs": "Fixed guarantees with premium flexibility.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/savings-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Lifetime Saver",
            "Plan Type": "Guaranteed Income Plan (Traditional Savings)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "65 years",
            "Maturity Age (Max)": "95 years",
            "Policy Term Options": "10, 15, 20 years",
            "Premium Payment Term Options": "Regular, Single Pay",
            "Minimum Premium Amount": "Rs. 51,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed regular income for life starting from maturity; return of 110% of total premiums at maturity.",
            "Key Features": "Lifetime annuity with capital return guarantee; flexible payout modes.",
            "Sum Assured Options": "110% of total premiums at maturity; life annuity thereafter.",
            "Riders Available": "None.",
            "Surrender Value": "No surrender value.",
            "Eligibility Criteria": "Life assured age 18–65 years; maturity at least 65.",
            "USPs": "Steady lifelong income with capital guarantee.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/retirement-pension-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Bachat Plus",
            "Plan Type": "Traditional Participating Savings Plan",
            "Entry Age (Min)": "45 days",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "18 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "10, 12, 15 years",
            "Premium Payment Term Options": "Limited Pay (4 or 6 years), Single Pay",
            "Minimum Premium Amount": "Rs. 25,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed loyalty additions; choice of lump sum or income payout; life cover and premium waiver on death.",
            "Key Features": "Participating with bonuses and flexible payout options.",
            "Sum Assured Options": "10× annual premium; 1.25× single premium.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "Guaranteed surrender plus bonuses after 2 years.",
            "Eligibility Criteria": "Life assured age 45 days–65 years.",
            "USPs": "Dual benefit of life cover and savings with bonuses.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/savings-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Platina Supreme",
            "Plan Type": "Guaranteed Income Plan (Traditional Savings)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Maturity Age (Min)": "75 years",
            "Maturity Age (Max)": "95 years",
            "Policy Term Options": "20, 25, 30 years",
            "Premium Payment Term Options": "Regular, Single Pay",
            "Minimum Premium Amount": "Rs. 50,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed annuity from payout age; return of 110% of total premiums at maturity; life cover.",
            "Key Features": "Lifetime income plus capital return; higher minimum premium for higher cover.",
            "Sum Assured Options": "110% of total premiums at maturity.",
            "Riders Available": "Optional Accidental Death Rider.",
            "Surrender Value": "Guaranteed surrender plus bonuses after 3 years.",
            "Eligibility Criteria": "Life assured age 18–65 years; maturity age 75–95 years.",
            "USPs": "High coverage guaranteed income for affluent customers.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/savings-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Money Back Gold",
            "Plan Type": "Money Back (Participating)",
            "Entry Age (Min)": "14 years",
            "Entry Age (Max)": "55 years",
            "Maturity Age (Min)": "18 years",
            "Maturity Age (Max)": "65 years",
            "Policy Term Options": "14 years",
            "Premium Payment Term Options": "Regular, Single Pay",
            "Minimum Premium Amount": "Rs. 9,500 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Periodic survival benefits at 5th, 8th, 11th years; total survival benefit 110% of sum assured; full life cover.",
            "Key Features": "Money-back plan with protection and periodic payouts.",
            "Sum Assured Options": "Level basic sum assured; total payouts equal 110% of sum assured.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "Guaranteed surrender plus bonuses after 2 years.",
            "Eligibility Criteria": "Life assured age 14–55 years; pay term options 4 or 5 years.",
            "USPs": "Protection with regular payouts and high survival benefit.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/money-back-income-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Money Planner",
            "Plan Type": "Income Plan (Participating)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "60 years",
            "Maturity Age (Min)": "28 years",
            "Maturity Age (Max)": "75 years",
            "Policy Term Options": "10, 12, 15 years",
            "Premium Payment Term Options": "Regular, Single Pay",
            "Minimum Premium Amount": "Varies by sum assured",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Life cover; regular income payouts for chosen benefit period; sum assured on death.",
            "Key Features": "Savings with guaranteed income component over payout phase.",
            "Sum Assured Options": "Level sum assured chosen by policyholder.",
            "Riders Available": "Standard riders as per norms.",
            "Surrender Value": "Paid-up and guaranteed surrender value after 2 years.",
            "Eligibility Criteria": "Life assured age 18–60 years; benefit period of 5/10/15 years.",
            "USPs": "Life cover plus customizable income stream.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/money-back-income-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Income Protect",
            "Plan Type": "Participating Savings Plan",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "60 years",
            "Maturity Age (Min)": "23 years",
            "Maturity Age (Max)": "70 years",
            "Policy Term Options": "15 years",
            "Premium Payment Term Options": "Regular",
            "Minimum Premium Amount": "Depends on sum assured",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Regular annual income during term; full life cover; bonuses.",
            "Key Features": "Dual-benefit plan with life cover and periodic income.",
            "Sum Assured Options": "Level sum assured used for income calculation.",
            "Riders Available": "Optional Accidental Death Benefit.",
            "Surrender Value": "Guaranteed surrender plus bonuses after 2 years.",
            "Eligibility Criteria": "Life assured age 18–60 years; fixed term 15 years.",
            "USPs": "Guaranteed annual income plus maturity bonus.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/money-back-income-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Scholar Plus",
            "Plan Type": "Child ULIP Plan",
            "Entry Age (Min)": "Child: 0 years; Parent: 18 years",
            "Entry Age (Max)": "Child: 17 years; Parent: 50 years",
            "Maturity Age (Min)": "Child: 18 years",
            "Maturity Age (Max)": "Child: 25 years; Parent: 65 years",
            "Policy Term Options": "8–25 years (ensuring child ≥18 at maturity)",
            "Premium Payment Term Options": "Single Pay; Limited Pay; Regular Pay",
            "Minimum Premium Amount": "Rs. 75,000 single; Rs. 50,000 annual",
            "Maximum Premium Amount": "No limit (subject to underwriting)",
            "Key Benefits": "Builds fund for child’s future with market-linked returns; premium waiver and death benefit.",
            "Key Features": "ULIP with wealth creation, 10 equity funds, loyalty additions, partial withdrawals from 6th year.",
            "Sum Assured Options": "Higher of 10× annual premium or 105% of total premiums on death.",
            "Riders Available": "In-built Accident Benefit.",
            "Surrender Value": "Fund value after 5 years.",
            "Eligibility Criteria": "Parent age 18–50; child age 0–17; child ≥18 at maturity.",
            "USPs": "Child cover plus savings: corpus for education with premium waiver.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/ulip/smart-scholar-plus"
        },
        {
            "Plan Name": "SBI Life – Smart Future Star",
            "Plan Type": "Child Endowment Plan",
            "Entry Age (Min)": "Child: 30 days; Parent: 18 years",
            "Entry Age (Max)": "Child: 12 years; Parent: 60 years",
            "Maturity Age (Min)": "Child: 18 years",
            "Maturity Age (Max)": "Child: 70 years",
            "Policy Term Options": "15–30 years",
            "Premium Payment Term Options": "Regular, Limited Pay",
            "Minimum Premium Amount": "Rs. 40,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Secure corpus for child’s future with bonuses; premium waiver on parent’s death.",
            "Key Features": "Participating child plan with bonuses, payout options, premium waiver.",
            "Sum Assured Options": "Lump sum = basic sum assured + bonuses; child death benefit equals sum assured.",
            "Riders Available": "None.",
            "Surrender Value": "Paid-up plus bonuses after 2 years.",
            "Eligibility Criteria": "Parent age 18–60; child age 30 days–12 years.",
            "USPs": "Traditional child savings with guaranteed bonuses and waiver.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/child-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Platina Young Achiever",
            "Plan Type": "Child Endowment Plan",
            "Entry Age (Min)": "Child: 30 days; Parent: 18 years",
            "Entry Age (Max)": "Child: 12 years; Parent: 60 years",
            "Maturity Age (Min)": "Child: 18 years",
            "Maturity Age (Max)": "Child: 70 years",
            "Policy Term Options": "12–30 years",
            "Premium Payment Term Options": "Regular",
            "Minimum Premium Amount": "Rs. 50,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed milestone benefits; premium waiver on parent’s death/TPD; flexible maturity payout.",
            "Key Features": "Non-participating with assured benefits and premium waiver.",
            "Sum Assured Options": "100% of premiums at maturity; 105% of premiums on child’s death.",
            "Riders Available": "None.",
            "Surrender Value": "50% of premiums after 2 years.",
            "Eligibility Criteria": "Parent age 18–60; child age 30 days–12 years.",
            "USPs": "Zero-risk child savings with guaranteed outcomes and waiver.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/child-plans"
        },
        {
            "Plan Name": "SBI Life – Retire Smart Plus",
            "Plan Type": "Retirement (ULIP)",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "60 years",
            "Maturity Age (Min)": "65 years",
            "Maturity Age (Max)": "95 years",
            "Policy Term Options": "10–25 years",
            "Premium Payment Term Options": "Regular, Limited Pay, Single Pay",
            "Minimum Premium Amount": "Rs. 30,000 per annum",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Build retirement corpus via market-linked funds; loyalty and terminal additions; flexibility in top-ups.",
            "Key Features": "ULIP for retirement with multiple fund choices and bonuses.",
            "Sum Assured Options": "125% single premium or 10× annual premium for death.",
            "Riders Available": "Optional Accident Benefit.",
            "Surrender Value": "Fund value after 2 years minus charges; no charges thereafter.",
            "Eligibility Criteria": "Life assured age 18–60 years.",
            "USPs": "Focused retirement ULIP with loyalty rewards and flexible investments.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/retirement-pension-plans"
        },
        {
            "Plan Name": "SBI Life – Smart Annuity Plus",
            "Plan Type": "Annuity Plan (Immediate/Deferred)",
            "Entry Age (Min)": "40 years",
            "Entry Age (Max)": "80 years",
            "Policy Term Options": "Lifetime annuity",
            "Premium Payment Term Options": "Single Premium",
            "Minimum Premium Amount": "As per guidelines",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed lifelong income; joint life option; incentive bonus for high purchase price.",
            "Key Features": "Immediate/deferred annuity with multiple payout options.",
            "Sum Assured Options": "Not applicable.",
            "Riders Available": "None.",
            "Surrender Value": "None (purchase price returned on death if applicable).",
            "Eligibility Criteria": "Entry age 40–80 years.",
            "USPs": "Multiple annuity options with bonus for higher premiums.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/pension/smart-annuity-plus"
        },
        {
            "Plan Name": "SBI Life – Smart Annuity Income",
            "Plan Type": "NPS Annuity Plan",
            "Entry Age (Min)": "18 years",
            "Entry Age (Max)": "65 years",
            "Policy Term Options": "Lifetime annuity",
            "Premium Payment Term Options": "Single Premium",
            "Minimum Premium Amount": "Per NPS norms",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Guaranteed lifelong annuity; multiple options; tax benefits.",
            "Key Features": "Non-linked annuity for NPS subscribers with assured income.",
            "Sum Assured Options": "Not applicable.",
            "Riders Available": "None.",
            "Surrender Value": "None.",
            "Eligibility Criteria": "Per NPS annuitization criteria.",
            "USPs": "Exclusively for NPS subscribers; steady annuity from corpus.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/pension/smart-annuity-income"
        },
        {
            "Plan Name": "SBI Life – Saral Pension",
            "Plan Type": "Immediate Pension (Single Premium)",
            "Entry Age (Min)": "40 years",
            "Entry Age (Max)": "80 years",
            "Policy Term Options": "Immediate lifetime annuity",
            "Premium Payment Term Options": "Single Premium",
            "Minimum Premium Amount": "Per annuity rate tables",
            "Maximum Premium Amount": "No limit",
            "Key Benefits": "Regular lifetime income; return of purchase price for certain options; CI surrender.",
            "Key Features": "Single premium annuity with immediate payouts; CI surrender benefit.",
            "Sum Assured Options": "Not applicable.",
            "Riders Available": "None (CI surrender built-in).",
            "Surrender Value": "Purchase price return on CI diagnosis.",
            "Eligibility Criteria": "Entry age 40–80 years.",
            "USPs": "Immediate pension with purchase price return and CI surrender.",
            "Official Brochure": "https://www.sbilife.co.in/en/individual-life-insurance/pension/saral-pension"
        }
    ]
"""

    asyncio.run(
        process_document(data, "SBI Life Policies", chunk_size=500, chunk_overlap=100)
    )
