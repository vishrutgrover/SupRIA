import google.generativeai as genai
import json

genai.configure(api_key="AIzaSyBcWbt9c7GhA_TgDR4uJ_cG7C-J-B1Gdu4")
model = genai.GenerativeModel("gemini-2.0-flash")
def handle_gemini_response(response):
    try:
        # Directly access the candidates field in the response
        candidates = response.candidates
        
        if candidates:
            # Extract the content from the first candidate using dot notation
            content = candidates[0].content.parts[0].text
            print("Raw Content Extracted:", content)
            
            return content
        else:
            print("No candidates found in the response.")
            return None
        
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return None

# Example of calling the function
prompt="Please generate a tag for health-conscious user."
response = model.generate_content(prompt)
parsed_response = handle_gemini_response(response)
