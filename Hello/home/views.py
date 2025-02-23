from django.shortcuts import render, HttpResponse, redirect

from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer

from django.views.decorators.csrf import csrf_exempt


# Function to generate a response
from home.setup import langgraph

#sample of fetching
def ask_question(question: str):
    response = langgraph.invoke({"question": question})
    try:
        return response["answer"]
    except:
        return "I cannot answer this currently. Please ask questions related to SBI Life policies."

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








