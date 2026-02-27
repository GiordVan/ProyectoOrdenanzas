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
    preguntar_a_gpt_stream,
    generar_embedding_local,
    MAX_DOCS_MOSTRADOS,
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
PDFS_ROOT_DIR = BASE_DIR.parent / "PDFs"
PDFS_FRONTEND_DIR = BASE_DIR.parent / "frontend" / "public" / "PDFs"

from fastapi.responses import FileResponse


# ⚡ Servir PDFs con fallback: primero en frontend/public/PDFs, luego en raíz /PDFs
@app.get("/pdfs/{filename}")
async def get_pdf(filename: str):
    # 1. Intentar en la carpeta del frontend
    frontend_file = PDFS_FRONTEND_DIR / filename
    if frontend_file.exists():
        return FileResponse(frontend_file)

    # 2. Intentar en la carpeta raíz
    root_file = PDFS_ROOT_DIR / filename
    if root_file.exists():
        return FileResponse(root_file)

    raise HTTPException(status_code=404, detail="PDF no encontrado")


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
    resultado_gpt = await loop.run_in_executor(
        None, preguntar_a_gpt, q.pregunta, contexto, resultados
    )
    respuesta_texto = resultado_gpt["respuesta"]
    ordenanzas_citadas = set(resultado_gpt.get("ordenanzas_citadas", []))

    # Construir documentos: solo los que GPT citó (o todos si no citó ninguno)
    documentos_info = []
    ordenanzas_vistas = set()
    for r in resultados:
        num_ord = r.get("numero_ordenanza", "desconocido")
        if num_ord in ordenanzas_vistas:
            continue
        # Si GPT citó ordenanzas específicas, filtrar. Si no especificó, mostrar todas
        if ordenanzas_citadas and num_ord not in ordenanzas_citadas:
            continue
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
        if len(documentos_info) >= MAX_DOCS_MOSTRADOS:
            break

    return Answer(respuesta=respuesta_texto, documentos=documentos_info)


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
        contexto = armar_contexto(resultados)

        # Enviar resultados encontrados como primer evento
        documentos_info = []
        ordenanzas_vistas = set()
        for r in resultados:
            num_ord = r.get("numero_ordenanza", "desconocido")
            if num_ord in ordenanzas_vistas:
                continue

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
            if len(documentos_info) >= MAX_DOCS_MOSTRADOS:
                break

        yield f"data: {json.dumps({'tipo': 'documentos', 'documentos': documentos_info})}\n\n"

        # Consumir el stream de OpenAI token por token
        async for chunk_text in preguntar_a_gpt_stream(
            q.pregunta, contexto, resultados
        ):
            # Escapar saltos de línea para el formato SSE
            safe_text = json.dumps({"tipo": "chunk", "texto": chunk_text})
            yield f"data: {safe_text}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Health check
@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)  # Solo encabezados
    # Si es GET, devuelve el cuerpo normalmente
    return {"status": "ok", "index_loaded": True}
