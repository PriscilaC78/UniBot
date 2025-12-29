import os
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv
from pypdf import PdfReader

# 1. Cargar las credenciales del archivo .env
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# 2. Configurar la IA y la Base de Datos
genai.configure(api_key=google_api_key)
supabase: Client = create_client(supabase_url, supabase_key)

def extract_text_from_pdf(pdf_path):
    """Lee el PDF y devuelve todo el texto como un string"""
    print(f"📖 Leyendo el archivo: {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=800, overlap=100):
    """
    Corta el texto en pedazos de 800 caracteres con 100 de solapamiento.
    El solapamiento ayuda a no cortar una frase importante a la mitad.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap # Movemos la ventana dejando un poco del anterior
    return chunks

def save_to_supabase(chunks):
    """Genera el vector (IA) para cada pedazo y lo guarda en la BD"""
    print(f"🧠 Procesando {len(chunks)} fragmentos de información...")
    
    total_guardados = 0
    
    for i, chunk in enumerate(chunks):
        try:
            # A. Generamos el 'Embedding' (la representación matemática del texto)
            # Usamos el modelo 'text-embedding-004' que es compatible con las 768 dimensiones que definimos
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']

            # B. Guardamos en Supabase
            data = {
                "content": chunk,
                "embedding": embedding,
                "metadata": {"source": "faq_alumnado.pdf"}
            }
            
            supabase.table("knowledge_base").insert(data).execute()
            print(f"✅ Fragmento {i+1}/{len(chunks)} guardado.")
            total_guardados += 1
            
        except Exception as e:
            print(f"❌ Error en fragmento {i+1}: {e}")

    print(f"\n🎉 ¡Listo! Se guardaron {total_guardados} fragmentos en la memoria de UniBot.")

# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # Asegúrate de que el nombre del archivo coincida con el que pusiste en la carpeta
    PDF_NAME = "faq.pdf" 
    
    if os.path.exists(PDF_NAME):
        raw_text = extract_text_from_pdf(PDF_NAME)
        text_chunks = split_text(raw_text)
        save_to_supabase(text_chunks)
    else:
        print(f"⚠️ Error: No encuentro el archivo '{PDF_NAME}'. Asegúrate de ponerlo en la carpeta.")