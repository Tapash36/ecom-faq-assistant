import os
from dotenv import load_dotenv
from google import genai

# Load API key from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file!")
    exit()

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send a test prompt
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say hello and confirm you are ready to help with our e-commerce FAQ chatbot project!"
)

print("\n--- GEMINI RESPONSE ---")
print(response.text)
print("------------------------\n")