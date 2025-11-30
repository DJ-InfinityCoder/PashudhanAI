import google.generativeai as genai

# 🔑 Replace with your actual API key
genai.configure(api_key="YOUR_GEMINI_API_KEY")

print("Fetching available models...\n")

for m in genai.list_models():
    print(m.name)
