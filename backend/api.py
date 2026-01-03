from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import os
from pathlib import Path

from chat_engine import (
    ensure_index_loaded,
    buscar_similares,
    armar_contexto,
    preguntar_a_gpt,
    generar_embedding_local,
)

app = FastAPI(title="Chat Legal IA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚡ CONFIGURACIÓN DE RUTAS
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
PDFS_DIR = BASE_DIR / "PDFs"

# ⚡ Servir PDFs estáticos (opcional, si quieres servirlos desde el backend)
if PDFS_DIR.exists():
    app.mount("/pdfs", StaticFiles(directory=str(PDFS_DIR)), name="pdfs")


# ⚡ Precarga del índice al iniciar
@app.on_event("startup")
async def startup_event():
    """Carga el índice al iniciar el servidor (no en cada request)"""
    print("🚀 Precargando índice FAISS...")
    ensure_index_loaded()
    # Pre-calentar el modelo de embeddings con una consulta dummy
    print("🔥 Pre-calentando modelo de embeddings...")
    generar_embedding_local("test de calentamiento")
    print("✅ Sistema listo para recibir consultas")


class Question(BaseModel):
    pregunta: str


class Answer(BaseModel):
    respuesta: str
    documentos: list


# ⚡ ENDPOINT: Servir metadatos.json ordenados y con caché
@app.get("/metadatos")
async def get_metadatos():
    """
    Devuelve el archivo metadatos.json ordenado por número de ordenanza (descendente).
    El navegador lo cacheará automáticamente.
    """
    metadatos_path = DATA_DIR / "metadatos.json"

    if not metadatos_path.exists():
        raise HTTPException(
            status_code=404, detail="Archivo metadatos.json no encontrado"
        )

    # Cargar y ordenar metadatos en el servidor
    with open(metadatos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Función segura para convertir número de ordenanza
    def safe_ord_num(x):
        try:
            return int(x.get("numero_ordenanza", "0") or "0")
        except (ValueError, TypeError):
            return 0  # Ordenanzas con número inválido van al final

    # Ordenar por número de ordenanza descendente (8000, 7999, 7998...)
    data.sort(key=safe_ord_num, reverse=True)

    return JSONResponse(
        content=data,
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache por 1 hora
            "ETag": f"metadatos-{os.path.getmtime(metadatos_path)}",  # Versionado
        },
    )


# ⚡ ENDPOINT ESTÁNDAR (más rápido con las optimizaciones)
@app.post("/ask", response_model=Answer)
async def ask_question(q: Question):
    if not q.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    loop = asyncio.get_event_loop()
    resultados = await loop.run_in_executor(None, buscar_similares, q.pregunta)

    contexto = armar_contexto(resultados)
    respuesta = await loop.run_in_executor(
        None, preguntar_a_gpt, q.pregunta, contexto, resultados
    )

    documentos_info = []
    ordenanzas_vistas = set()
    for r in resultados:
        num_ord = r.get("numero_ordenanza", "desconocido")
        if num_ord not in ordenanzas_vistas:
            ordenanzas_vistas.add(num_ord)
            documentos_info.append(
                {
                    "nombre": r.get("nombre_archivo", ""),
                    "numero_ordenanza": num_ord,
                    "fecha_sancion": r.get("fecha_sancion"),
                    "fragmento": r["chunk_texto"][:250] + "...",
                    "pdf": r.get("nombre_archivo"),
                }
            )
            if len(documentos_info) >= 10:
                break

    return Answer(respuesta=respuesta, documentos=documentos_info)


# ⚡ ENDPOINT CON STREAMING (respuesta incremental)
@app.post("/ask-stream")
async def ask_question_stream(q: Question):
    """
    Endpoint que envía datos progresivamente:
    1. Primero envía los documentos
    2. Luego la respuesta de GPT
    """
    if not q.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    async def generate():
        loop = asyncio.get_event_loop()
        resultados = await loop.run_in_executor(None, buscar_similares, q.pregunta)

        documentos_info = []
        ordenanzas_vistas = set()
        for r in resultados:
            num_ord = r.get("numero_ordenanza", "desconocido")
            if num_ord not in ordenanzas_vistas:
                ordenanzas_vistas.add(num_ord)
                documentos_info.append(
                    {
                        "nombre": r.get("nombre_archivo", ""),
                        "numero_ordenanza": num_ord,
                        "fecha_sancion": r.get("fecha_sancion"),
                        "fragmento": r["chunk_texto"][:250] + "...",
                        "pdf": r.get("nombre_archivo"),
                    }
                )
                if len(documentos_info) >= 10:
                    break

        yield f"data: {json.dumps({'tipo': 'documentos', 'documentos': documentos_info})}\n\n"

        contexto = armar_contexto(resultados)
        respuesta = await loop.run_in_executor(
            None, preguntar_a_gpt, q.pregunta, contexto
        )

        yield f"data: {json.dumps({'tipo': 'respuesta', 'respuesta': respuesta})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Health check
@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)  # Solo encabezados
    # Si es GET, devuelve el cuerpo normalmente
    return {"status": "ok", "index_loaded": True}
