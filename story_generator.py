import google.generativeai as genai
import os  #It is used to fecth the environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")   #Used for fast and efficient generation of story telling tasks

def continue_story(scene, emotion):#The initial user prompt and the emotion
    prompt = (
        f"Continue this story in the tone of {emotion}. Make it emotionally rich and realistic.\n\n"
        f"Scene: {scene}\n\n"
        f"Continuation:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()
