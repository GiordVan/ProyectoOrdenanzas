import asyncio
import json
import os
import re
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from chat_engine import (
    MAX_DOCS_MOSTRADOS,
    aclient,
    armar_contexto,
    buscar_similares,
    construir_pregunta_aclaratoria,
    detectar_tipo_pregunta,
    detectar_zona_propiedad,
    ensure_index_loaded,
    extraer_anios_de_texto,
    generar_embedding_local,
    preguntar_a_gpt,
    resolver_modalidad_pago_propiedad,
    resolver_pregunta_presupuesto,
    resolver_tarifaria_intenciones,
    resolver_tasa_propiedad,
)

app = FastAPI(title="Chat Legal IA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://demo-digestodigital.netlify.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CONVERSATION_MEMORY = {}
CONVERSATION_LOCK = threading.Lock()


@app.on_event("startup")
async def startup_event():
    """Carga el indice al iniciar el servidor."""
    print("Precargando indice FAISS...")
    ensure_index_loaded()
    print("Pre-calentando modelo de embeddings...")
    generar_embedding_local("test de calentamiento")
    print("Sistema listo para recibir consultas")


class Question(BaseModel):
    pregunta: str
    conversation_id: str | None = None


class Answer(BaseModel):
    respuesta: str
    documentos: list


def construir_documentos_info(resultados: list, ordenanzas_citadas: set | None = None):
    """Construye documentos unicos para mostrar en frontend."""
    documentos_info = []
    ordenanzas_vistas = set()
    ordenanzas_citadas = {str(x) for x in (ordenanzas_citadas or set())}

    for r in resultados:
        num_ord = str(r.get("numero_ordenanza", "desconocido"))
        if num_ord in ordenanzas_vistas:
            continue
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

    return documentos_info


def _get_or_create_conversation_id(conversation_id: str | None) -> str:
    if conversation_id and conversation_id.strip():
        return conversation_id.strip()
    return str(uuid4())


def _get_memory_state(conversation_id: str) -> dict:
    with CONVERSATION_LOCK:
        return CONVERSATION_MEMORY.setdefault(conversation_id, {})


def _aplicar_memoria_pregunta(pregunta: str, state: dict) -> str:
    """
    Si hay una aclaración pendiente y el usuario responde corto (ej: 'Zona 7'),
    reconstruye una pregunta completa para retrieval + respuesta.
    """
    if not state:
        return pregunta

    pending_topic = state.get("pending_topic")
    last_topic = state.get("last_topic")
    if pending_topic != "tasa_propiedad" and last_topic != "tasa_propiedad":
        return pregunta

    zona = detectar_zona_propiedad(pregunta)
    if zona is None:
        return pregunta

    anio = state.get("year", "2025")
    return f"¿Cuánto tengo que pagar de tasa anual por mi propiedad en zona {zona} para el año {anio}?"


def _actualizar_memoria(
    conversation_id: str, pregunta_usuario: str, respuesta_texto: str
):
    state = _get_memory_state(conversation_id)
    respuesta_norm = re.sub(r"\s+", " ", (respuesta_texto or "").strip().lower())
    pregunta_norm = re.sub(r"\s+", " ", (pregunta_usuario or "").strip().lower())

    # Recordar ultimo tema aunque no quede pendiente una aclaracion.
    if "tasa" in pregunta_norm and (
        "propiedad" in pregunta_norm or "inmueble" in pregunta_norm
    ):
        state["last_topic"] = "tasa_propiedad"

    if (
        "en qué zona está tu propiedad" in respuesta_norm
        or "en que zona esta tu propiedad" in respuesta_norm
    ):
        state["pending_topic"] = "tasa_propiedad"
        anios = extraer_anios_de_texto(pregunta_usuario)
        state["year"] = anios[0] if anios else "2025"
        return

    # Si la respuesta ya resolvió una consulta de zona, limpiar pendiente.
    if (
        state.get("pending_topic") == "tasa_propiedad"
        and detectar_zona_propiedad(pregunta_usuario) is not None
    ):
        state.pop("pending_topic", None)


NO_RESULT_MESSAGE = (
    "No encontré información relevante en el Digesto Digital para tu consulta. "
    "Intentá reformular la pregunta con términos más específicos, "
    "o indicar el número de ordenanza si lo conocés."
)
MIN_COSINE_FOR_GPT = 0.45


def _resultados_son_relevantes(resultados: list) -> bool:
    """Verifica si los resultados tienen calidad suficiente para enviar a GPT."""
    if not resultados:
        return False
    for r in resultados:
        if r.get("score_semantico", 0) >= MIN_COSINE_FOR_GPT:
            return True
        if r.get("coincidencias_textuales", 0) >= 2:
            return True
    return False


def _intentar_resolver_deterministico(pregunta: str, resultados: list) -> dict | None:
    """
    Ejecuta todos los resolvers determinísticos en orden de prioridad.
    Retorna dict {respuesta, ordenanzas_citadas} o None.
    """
    for resolver in [
        resolver_tasa_propiedad,
        resolver_modalidad_pago_propiedad,
        resolver_tarifaria_intenciones,
        construir_pregunta_aclaratoria,
        resolver_pregunta_presupuesto,
    ]:
        try:
            resultado = resolver(pregunta, resultados)
            if resultado:
                # construir_pregunta_aclaratoria retorna str, no dict
                if isinstance(resultado, str):
                    return {"respuesta": resultado, "ordenanzas_citadas": []}
                return resultado
        except Exception as e:
            print(f"Error en resolver {resolver.__name__}: {e}")
    return None


@app.get("/metadatos")
async def get_metadatos():
    """
    Devuelve metadatos ordenados por numero de ordenanza (descendente).
    """
    metadatos_path = DATA_DIR / "metadatos.json"
    if not metadatos_path.exists():
        raise HTTPException(
            status_code=404, detail="Archivo metadatos.json no encontrado"
        )

    with open(metadatos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def safe_ord_num(x):
        try:
            return int(x.get("numero_ordenanza", "0") or "0")
        except (ValueError, TypeError):
            return 0

    data.sort(key=safe_ord_num, reverse=True)

    return JSONResponse(
        content=data,
        headers={
            "Cache-Control": "public, max-age=3600",
            "ETag": f"metadatos-{os.path.getmtime(metadatos_path)}",
        },
    )


@app.post("/ask", response_model=Answer)
async def ask_question(q: Question):
    if not q.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    conversation_id = _get_or_create_conversation_id(q.conversation_id)
    state = _get_memory_state(conversation_id)
    pregunta_efectiva = _aplicar_memoria_pregunta(q.pregunta, state)

    loop = asyncio.get_event_loop()
    resultados = await loop.run_in_executor(None, buscar_similares, pregunta_efectiva)

    if not _resultados_son_relevantes(resultados):
        _actualizar_memoria(conversation_id, q.pregunta, NO_RESULT_MESSAGE)
        return Answer(respuesta=NO_RESULT_MESSAGE, documentos=[])

    # Intentar resolvers determinísticos primero (Consistencia con streaming)
    resultado_resolver = await loop.run_in_executor(
        None, _intentar_resolver_deterministico, pregunta_efectiva, resultados
    )

    if resultado_resolver:
        respuesta_texto = resultado_resolver["respuesta"]
        ordenanzas_citadas = set(resultado_resolver.get("ordenanzas_citadas", []))
        documentos_info = construir_documentos_info(resultados, ordenanzas_citadas)
        _actualizar_memoria(conversation_id, q.pregunta, respuesta_texto)
        return Answer(respuesta=respuesta_texto, documentos=documentos_info)

    contexto = armar_contexto(resultados)
    resultado_gpt = await loop.run_in_executor(
        None, preguntar_a_gpt, pregunta_efectiva, contexto, resultados
    )
    respuesta_texto = resultado_gpt["respuesta"]
    ordenanzas_citadas = set(resultado_gpt.get("ordenanzas_citadas", []))
    documentos_info = construir_documentos_info(resultados, ordenanzas_citadas)
    _actualizar_memoria(conversation_id, q.pregunta, respuesta_texto)

    return Answer(respuesta=respuesta_texto, documentos=documentos_info)


@app.post("/ask-stream")
async def ask_question_stream(q: Question):
    """
    Endpoint streaming real:
    1) Busca documentos y los envía
    2) Ejecuta resolvers determinísticos
    3a) Si un resolver responde → streaming simulado char-a-char (rápido)
    3b) Si no → streaming real de OpenAI token a token
    """
    if not q.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    async def generate():
        conversation_id = _get_or_create_conversation_id(q.conversation_id)
        state = _get_memory_state(conversation_id)
        pregunta_efectiva = _aplicar_memoria_pregunta(q.pregunta, state)

        loop = asyncio.get_event_loop()
        resultados = await loop.run_in_executor(
            None, buscar_similares, pregunta_efectiva
        )

        if not _resultados_son_relevantes(resultados):
            yield f"data: {json.dumps({'tipo': 'documentos', 'documentos': []})}\n\n"
            msg = NO_RESULT_MESSAGE
            paso = 8
            for i in range(0, len(msg), paso):
                yield f"data: {json.dumps({'tipo': 'chunk', 'texto': msg[i:i+paso]})}\n\n"
                await asyncio.sleep(0.02)
            _actualizar_memoria(conversation_id, q.pregunta, msg)
            yield "data: [DONE]\n\n"
            return

        documentos_info = construir_documentos_info(resultados)

        # 1. Enviar documentos al frontend
        yield f"data: {json.dumps({'tipo': 'documentos', 'documentos': documentos_info})}\n\n"

        # 2. Intentar resolvers determinísticos primero
        resultado_resolver = await loop.run_in_executor(
            None, _intentar_resolver_deterministico, pregunta_efectiva, resultados
        )

        if resultado_resolver:
            # ⚡ Resolver respondió — streaming simulado char-a-char
            respuesta_texto = resultado_resolver.get("respuesta", "")
            ordenanzas_citadas = set(resultado_resolver.get("ordenanzas_citadas", []))
            _actualizar_memoria(conversation_id, q.pregunta, respuesta_texto)

            # Reenviar documentos filtrados
            if ordenanzas_citadas:
                docs_filtrados = construir_documentos_info(
                    resultados, ordenanzas_citadas
                )
                yield f"data: {json.dumps({'tipo': 'documentos', 'documentos': docs_filtrados})}\n\n"

            # Simular efecto de escritura: enviar en segmentos pequeños
            paso = 8  # ~8 chars por tick para efecto de escritura fluido
            for i in range(0, len(respuesta_texto), paso):
                parte = respuesta_texto[i : i + paso]
                yield f"data: {json.dumps({'tipo': 'chunk', 'texto': parte})}\n\n"
                await asyncio.sleep(0.02)  # 20ms entre ticks para efecto visual

            yield "data: [DONE]\n\n"
            return

        # 3. Sin resolver → Streaming real de OpenAI
        contexto = armar_contexto(resultados)

        # Calcular números disponibles para el prompt
        numeros_disponibles = []
        if resultados:
            vistos = set()
            for r in resultados:
                n = r.get("numero_ordenanza", "")
                if n and n not in vistos and n != "desconocido":
                    vistos.add(n)
                    numeros_disponibles.append(n)

        lista_nums = (
            ", ".join(numeros_disponibles) if numeros_disponibles else "ninguna"
        )
        tipo_pregunta = detectar_tipo_pregunta(pregunta_efectiva)

        if tipo_pregunta == "numero_especifico":
            prompt = f"""Eres el Digesto Digital de Villa María. El usuario quiere saber de qué trata la ordenanza N° {pregunta_efectiva}.

Usando SOLO la información del contexto, escribe un resumen claro y conciso.
Comienza tu respuesta con "La Ordenanza N° {pregunta_efectiva}" y describe su contenido principal.

Contexto:
{contexto}

Responde directamente en Markdown limpio. NO uses formato JSON."""
        else:
            prompt = f"""(Dando formato como para poner en una pagina web) Eres el Digesto Digital de Villa María. Responde de forma completa y clara usando SOLO la información del contexto. Si la pregunta pide enumerar elementos (acuerdos, artículos, partes, etc.), listarlos todos con viñetas Markdown. Si es una pregunta simple, responde en 1-2 oraciones.

Contexto (ordenanzas disponibles: {lista_nums}):
{contexto}

Pregunta: {pregunta_efectiva}

Responde directamente en Markdown limpio. NO uses formato JSON."""

        respuesta_acumulada = ""
        citas_extraidas = []

        try:
            stream = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    respuesta_acumulada += delta
                    yield f"data: {json.dumps({'tipo': 'chunk', 'texto': delta})}\n\n"

        except Exception as e:
            print(f"Error en OpenAI Async Stream: {e}")
            error_msg = (
                "Hubo un error al generar la respuesta. Por favor, intenta de nuevo."
            )
            yield f"data: {json.dumps({'tipo': 'chunk', 'texto': error_msg})}\n\n"
            respuesta_acumulada = error_msg

        # Actualizar memoria y reenviar documentos con citas
        _actualizar_memoria(conversation_id, q.pregunta, respuesta_acumulada)

        # Intentar extraer citas del texto
        for n in numeros_disponibles:
            if n in respuesta_acumulada:
                citas_extraidas.append(n)

        if citas_extraidas:
            docs_filtrados = construir_documentos_info(resultados, set(citas_extraidas))
            yield f"data: {json.dumps({'tipo': 'documentos', 'documentos': docs_filtrados})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"status": "ok", "index_loaded": True}
