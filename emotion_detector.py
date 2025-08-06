import google.generativeai as genai    #To interact with the Gemini models
import os    #It is to get the access of the environment variables like API Keys

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")   #This model is a lightweight and fast model and suitable for emotion classification tasks

def detect_emotion(text):
    prompt = f"Detect the dominant emotion from this text: '{text}'. Respond with one word like 'fear', 'joy', 'sadness', 'anger', or 'surprise'."
    response = model.generate_content(prompt)
    return response.text.strip().lower()
