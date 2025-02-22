from setup import langgraph

#sample of fetching
def ask_question(question: str):
    response = langgraph.invoke({"question": question})
    try:
        return response["answer"]
    except:
        return "I cannot answer this currently. Please ask questions related to SBI Life policies."

# driver
# answer = ask_question("spouse ke liye insurance dikhaana and also for mera baccha")
# print("Final Answer: ", answer)