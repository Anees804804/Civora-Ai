import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

print('Available embedding models:')
found = False
for m in genai.list_models():
    if 'embed' in m.name.lower():
        found = True
        print(f'  - {m.name}')
        supported_methods = [method.name for method in m.supported_generation_methods]
        print(f'    Supported methods: {supported_methods}')

if not found:
    print('  No embedding models found! Listing ALL models:')
    for m in genai.list_models():
        print(f'  - {m.name}')
