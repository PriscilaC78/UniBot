import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv


# 1. Configuración inicial
load_dotenv()
app = FastAPI()

# Configuración de permisos para que la web pueda hablar con el bot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" significa que permite conexiones desde cualquier sitio (ideal para pruebas)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión a Servicios
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not google_api_key:
    raise ValueError("❌ Faltan las variables de entorno en el archivo .env")

supabase: Client = create_client(supabase_url, supabase_key)
genai.configure(api_key=google_api_key)

# Modelo de datos para recibir la pregunta
class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo" # Para identificar al usuario (opcional por ahora)

def buscar_contexto(pregunta_usuario: str):
    """
    1. Convierte la pregunta en vectores.
    2. Busca en Supabase los fragmentos más parecidos.
    """
    try:
        # Generamos el embedding de la pregunta
        # Revisa que esta línea tenga "models/" al principio
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=pregunta_usuario,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']

        # Llamamos a la función 'match_documents' que creamos con SQL en Supabase
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.5, # Qué tan parecida debe ser la info (0 a 1)
            "match_count": 5        # Cuántos fragmentos traer
        }).execute()
        
        # Unimos todos los fragmentos de texto encontrados en uno solo
        contexto = "\n\n".join([item['content'] for item in response.data])
        return contexto
        
    except Exception as e:
        print(f"Error buscando contexto: {e}")
        return ""

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    """
    Endpoint principal: Recibe JSON {"pregunta": "..."} y devuelve la respuesta.
    """
    print(f"📩 Pregunta recibida: {query.pregunta}")
    
    # 1. Buscamos información relevante en la base de datos
    contexto = buscar_contexto(query.pregunta)
    
    if not contexto:
        # Si no encuentra nada en la BD, la IA no debe inventar.
        return {"respuesta": "Lo siento, no tengo información sobre ese tema específico en mi base de conocimientos de Alumnado."}

    # 2. Armamos el Prompt para la IA
    # Prompt mejorado con "personalidad"
        prompt = f"""
        Eres UniBot, el asistente virtual del área de Alumnado de UNCAUS. Tu tono es amable, profesional y claro.

        Instrucciones:
        1. Si el usuario saluda (ej: "hola", "buen día"), responde amablemente, preséntate brevemente y pregunta en qué puedes ayudar. NO uses el contexto para esto.
        2. Para cualquier pregunta sobre trámites o la universidad, responde basándote EXCLUSIVAMENTE en el siguiente contexto recuperado del PDF.
        3. Si la respuesta no está en el contexto, di amablemente que no tienes esa información específica.

        Contexto recuperado:
        {contexto}

        Pregunta del usuario:
        {pregunta_usuario}
        """

    # 3. Generamos la respuesta con Gemini
    # Usamos este modelo que SÍ está en tu lista confirmada
    # Usamos el alias 'latest' que siempre apunta al modelo activo y gratuio
    model = genai.GenerativeModel('models/gemini-flash-latest')# Modelo rápido y gratis
    response = model.generate_content(prompt)
    
    respuesta_final = response.text

    # 4. (Opcional) Guardamos el chat en el historial
    supabase.table("chat_logs").insert({
        "session_id": query.session_id,
        "user_input": query.pregunta,
        "bot_response": respuesta_final
    }).execute()

    return {"respuesta": respuesta_final}

@app.get("/")
def home():
    return {"status": "UniBot está vivo y funcionando 🤖"}