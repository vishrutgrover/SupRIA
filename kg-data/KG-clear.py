from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph

load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    refresh_schema=False
)

graph.query("MATCH (n) DETACH DELETE n")
print("Database emptied successfully.")