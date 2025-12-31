import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Configuración inicial
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Conexión a Servicios
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not google_api_key:
    raise ValueError("❌ Error: Faltan variables de entorno.")

try:
    supabase: Client = create_client(supabase_url, supabase_key)
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"❌ Error conexión servicios: {e}")

class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo"

def buscar_contexto(pregunta_usuario: str):
    try:
        # Modelo de embeddings estándar
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=pregunta_usuario,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']

        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.4,
            "match_count": 3
        }).execute()
        
        contexto = "\n\n".join([item['content'] for item in response.data])
        return contexto
    except Exception as e:
        print(f"⚠️ Error buscando contexto: {e}")
        return ""

# --- NUEVA FUNCIÓN: PROBAR MÚLTIPLES MODELOS ---
def generar_respuesta_inteligente(prompt):
    # Lista de modelos sacada de tu diagnóstico (en orden de preferencia)
    # Probamos con y sin el prefijo "models/" por seguridad
    lista_modelos = [
        "gemini-2.0-flash",           # La mejor opción
        "gemini-2.0-flash-exp",       # Experimental (suele estar libre)
        "gemini-flash-latest",        # El último estable genérico
        "models/gemini-2.0-flash",    # Con prefijo por si acaso
        "gemini-1.5-flash-latest"     # Respaldo antiguo confiable
    ]

    errores = []
    
    for nombre_modelo in lista_modelos:
        try:
            print(f"🔄 Intentando con modelo: {nombre_modelo}...")
            model = genai.GenerativeModel(nombre_modelo)
            response = model.generate_content(prompt)
            print(f"✅ ¡Éxito con {nombre_modelo}!")
            return response.text
        except Exception as e:
            print(f"❌ Falló {nombre_modelo}: {e}")
            errores.append(str(e))
            continue # Pasa al siguiente de la lista

    # Si llegamos aquí, fallaron TODOS
    print("💀 Todos los modelos fallaron.")
    return f"Lo siento, hubo un error técnico. Detalles: {errores[-1]}"

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"📩 Pregunta: {query.pregunta}")

    saludos = ["hola", "buen dia", "buenas", "que tal"]
    if any(s in query.pregunta.lower() for s in saludos) and len(query.pregunta) < 20:
        return {"respuesta": "¡Hola! 👋 Soy UniBot. ¿En qué puedo ayudarte?"}

    contexto = buscar_contexto(query.pregunta)
    
    prompt = f"""
    Eres UniBot de UNCAUS. Responde usando SOLO este contexto:
    "{contexto}"
    Si no está en el contexto, di que no sabes.
    """ 

    # Llamamos a la función "Todoterreno"
    respuesta_final = generar_respuesta_inteligente(prompt)

    try:
        supabase.table("chat_logs").insert({
            "session_id": query.session_id,
            "user_input": query.pregunta,
            "bot_response": respuesta_final
        }).execute()
    except:
        pass

    return {"respuesta": respuesta_final}

@app.get("/")
def home():
    return {"status": "UniBot v5.0 - AUTO-SELECTOR DE MODELOS 🤖"}