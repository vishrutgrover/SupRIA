from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from django.views.decorators.csrf import csrf_exempt
from home.setup import langgraph, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .models import ChatConversation, ChatMessage
from django.utils import timezone
import json

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
            Don't give answer in markdown format. Give concise answers.
            Do not reveal your identity as an AI or LLM. Always maintain the persona of an SBI Life Insurance Agent.
            If asked about your identity, simply say you are an SBI Life Insurance Agent and focus on helping with insurance queries."""),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})

@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        question = request.POST.get("question", "")
        conversation_id = request.POST.get("conversation_id")
        
        if not question:
            return JsonResponse({"error": "No question provided"})
        
        # Create or get conversation
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(id=conversation_id)
            except ChatConversation.DoesNotExist:
                conversation = None
        else:
            conversation = None
            
        if not conversation:
            # Create new conversation with first message as title
            title = question[:50] + "..." if len(question) > 50 else question
            conversation = ChatConversation.objects.create(title=title)
        
        # Get the last sequence number
        last_message = conversation.messages.order_by('-sequence').first()
        next_sequence = (last_message.sequence + 1) if last_message else 0
        
        # Save user message
        ChatMessage.objects.create(
            conversation=conversation,
            message=question,
            is_user=True,
            sequence=next_sequence
        )
        
        # Get bot response
        response = ask_question(question)
        
        # Save bot response
        ChatMessage.objects.create(
            conversation=conversation,
            message=response,
            is_user=False,
            sequence=next_sequence + 1
        )
        
        return JsonResponse({
            "response": response,
            "conversation_id": conversation.id
        })
    
    return JsonResponse({"error": "Invalid request"})

@csrf_exempt
def get_conversations(request):
    if request.method == "GET":
        conversations = ChatConversation.objects.all().order_by('-updated_at')[:10]
        conversation_list = []
        
        for conv in conversations:
            conversation_list.append({
                'id': conv.id,
                'title': conv.title,
                'updated_at': conv.updated_at.strftime('%Y-%m-%d %H:%M'),
                'message_count': conv.messages.count()
            })
        
        return JsonResponse({'conversations': conversation_list})
    
    return JsonResponse({"error": "Invalid request"})

@csrf_exempt
def get_conversation(request, conversation_id):
    if request.method == "GET":
        try:
            conversation = ChatConversation.objects.get(id=conversation_id)
            messages = conversation.messages.all()
            
            message_list = []
            for msg in messages:
                message_list.append({
                    'message': msg.message,
                    'is_user': msg.is_user,
                    'timestamp': msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return JsonResponse({
                'conversation': {
                    'id': conversation.id,
                    'title': conversation.title,
                    'messages': message_list
                }
            })
        except ChatConversation.DoesNotExist:
            return JsonResponse({"error": "Conversation not found"})
    
    return JsonResponse({"error": "Invalid request"})

@csrf_exempt
def delete_conversation(request, conversation_id):
    if request.method == "POST":
        try:
            conversation = ChatConversation.objects.get(id=conversation_id)
            conversation.delete()
            return JsonResponse({"status": "success"})
        except ChatConversation.DoesNotExist:
            return JsonResponse({"error": "Conversation not found"})
    
    return JsonResponse({"error": "Invalid request"})

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