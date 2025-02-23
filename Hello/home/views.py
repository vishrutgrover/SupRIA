from django.shortcuts import render, HttpResponse, redirect

from django.http import JsonResponse
from transformers import T5ForConditionalGeneration, T5Tokenizer

from django.views.decorators.csrf import csrf_exempt

# Load your fine-tuned T5 model and tokenizer
model_path = "D:\My Files\django\Hello\home\my_t5_chatbot"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


# Function to generate a response
def generate_response(question, max_length=250, num_beams=10):
    """
    Generate a coherent and concise answer using the fine-tuned T5 model.
    
    Args:
    - question (str): User's question.
    - max_length (int): Max length for generated answers.
    - num_beams (int): Number of beams for beam search.
    
    Returns:
    - str: The chatbot's answer.
    """
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate response with improved decoding options
    outputs = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=num_beams, 
        early_stopping=True,
        repetition_penalty=2.0,  # Stronger penalty to avoid repetitive phrases
        length_penalty=1.2,       # Encourage slightly longer, meaningful responses
        no_repeat_ngram_size=3,   # Prevent repeating n-grams (like 'non-participating' spam)
        temperature=0.9,          # Add randomness for more human-like variety
        top_k=50,                 # Consider top 50 tokens at each step
        top_p=0.9                 # Nucleus sampling for diverse outputs
    )
    
    # Decode and return the generated answer
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
@csrf_exempt
# Django view to handle chatbot requests
def chatbot_response(request):
    if request.method == "POST":
        question = request.POST.get("question", "")
        if question:
            response = generate_response(question)
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








