import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, refresh_schema=False)

graph.query("MATCH (n) DETACH DELETE n") # Clear the knowledge graph
print("Knowledge graph cleared successfully.")