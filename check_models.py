import google.generativeai as genai
import os
from dotenv import load_dotenv

# Cargar claves
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("🔍 Preguntando a Google qué modelos puedo usar...")

try:
    # Listar todos los modelos disponibles para tu cuenta
    for m in genai.list_models():
        # Solo queremos los que sirven para chatear (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ MODELO DISPONIBLE: {m.name}")
            
except Exception as e:
    print(f"❌ Error conectando con Google: {e}")