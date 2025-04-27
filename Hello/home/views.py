from django.shortcuts import render, HttpResponse, redirect

from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer

from django.views.decorators.csrf import csrf_exempt

from home.setup import langgraph, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Function to check if question is related to SBI Life policies
def is_sbi_life_related(question: str) -> bool:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that determines if a question is related to SBI Life insurance policies.
        Return 'yes' if the question is about SBI Life insurance policies, products, services, or related information, even if it's just about something which can be related to policies.
        Return 'no' if the question is about anything else.
        Be strict in your judgment - only return 'yes' for questions directly related to SBI Life insurance.
        Do not reveal your identity or capabilities. Simply return 'yes' or 'no'."""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    return response.lower().strip() == "yes"

def ask_question(question: str):
    if is_sbi_life_related(question):
        response = langgraph.invoke({"question": question})
        try:
            return response["answer"]
        except:
            return "I cannot answer this currently. Please ask questions related to SBI Life policies."
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an SBI Life Insurance Agent. Your role is to help customers with their insurance-related queries.
            If the question is not related to insurance or SBI Life, politely redirect them to ask insurance-related questions. 
            Don't give answer in markdown format.
            Do not reveal your identity as an AI or LLM. Always maintain the persona of an SBI Life Insurance Agent.
            If asked about your identity, simply say you are an SBI Life Insurance Agent and focus on helping with insurance queries."""),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})

@csrf_exempt
# Django view to handle chatbot requests
def chatbot_response(request):
    if request.method == "POST":
        question = request.POST.get("question", "")
        if question:
            response = ask_question(question)
            return JsonResponse({"response": response})
        return JsonResponse({"error": "No question provided"})
    return JsonResponse({"error": "Invalid request"})

# Create your views here.
def index(request):
    return render(request, 'new.html')
    # return HttpResponse("This is Homepage")
def about(request):
    return redirect("https://www.sbilife.co.in/en/about-us")
def services(request):
    return redirect("https://www.sbilife.co.in/en/services")
def contacts(request):
    return redirect("https://www.sbilife.co.in/en/about-us/contact-us")
def chatbot(request):
    return render(request, 'real_chatbot.html')