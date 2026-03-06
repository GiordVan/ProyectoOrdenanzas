import asyncio
import json
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from chat_engine import (
    MAX_DOCS_MOSTRADOS,
    SIMILARITY_THRESHOLD,
    aclient,
    armar_contexto,
    buscar_similares,
    chunks as engine_chunks,
    construir_pregunta_aclaratoria,
    detectar_tipo_pregunta,
    ensure_index_loaded,
    generar_embedding_local,
    metadatos as engine_metadatos,
    obtener_chunk_y_meta_seguro,
    preguntar_a_gpt,
    rerank_con_llm,
    resolver_modalidad_pago_propiedad,
    resolver_pregunta_presupuesto,
    resolver_tarifaria_intenciones,
    resolver_tasa_propiedad,
)

app = FastAPI(title="Chat Legal IA API")

_DEFAULT_ORIGINS = [
    "https://demo-digestodigital.netlify.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]
CORS_ORIGINS = (
    [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if os.getenv("CORS_ORIGINS")
    else _DEFAULT_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CONVERSATION_MEMORY = {}
CONVERSATION_LOCK = threading.Lock()
CONVERSATION_TTL = 3600  # 1 hora
MAX_HISTORY = 10  # 5 intercambios (usuario + asistente)

_FOLLOW_UP_INDICATORS = {
    "eso", "esa", "ese", "esto", "esta", "este",
    "lo mismo", "también", "tambien", "además", "ademas",
    "y si", "pero", "entonces", "o sea",
    "la misma", "el mismo", "de esa", "de ese", "sobre eso",
    "me dijiste", "mencionaste", "hablamos", "dijiste",
}


def _cleanup_stale_conversations():
    """Elimina conversaciones inactivas por mas de CONVERSATION_TTL segundos."""
    now = time.time()
    with CONVERSATION_LOCK:
        stale = [
            k
            for k, v in CONVERSATION_MEMORY.items()
            if now - v.get("_last_access", 0) > CONVERSATION_TTL
        ]
        for k in stale:
            del CONVERSATION_MEMORY[k]


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
                "resumen": r.get("resumen", ""),
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
        state = CONVERSATION_MEMORY.setdefault(conversation_id, {})
        state["_last_access"] = time.time()
        return state


def _parece_follow_up(pregunta: str) -> bool:
    """Heurística rápida: ¿parece continuación de conversación anterior?"""
    words = pregunta.strip().split()
    if len(words) <= 3:
        return True  # Muy corta, casi seguro follow-up
    p_lower = pregunta.lower()
    # Preguntas de 4-6 palabras: solo follow-up si tienen pronombres/referencias
    if len(words) <= 6:
        pronombres = {
            "esa", "ese", "esta", "este", "lo", "la", "las", "los",
            "su", "sus", "el mismo", "la misma", "ahí", "eso", "esto",
            "dicha", "dicho", "mencionada", "mencionado",
        }
        if any(p in p_lower for p in pronombres):
            return True
        # También si tiene indicadores explícitos de follow-up
        return any(ind in p_lower for ind in _FOLLOW_UP_INDICATORS)
    return any(ind in p_lower for ind in _FOLLOW_UP_INDICATORS)


async def _reformular_si_follow_up(
    pregunta: str, historial: list, last_ordinances: list | None = None
) -> str:
    """
    Si hay historial y la pregunta parece follow-up, usa GPT para reformularla
    como pregunta completa e independiente para retrieval.
    """
    if not historial or not _parece_follow_up(pregunta):
        return pregunta

    hist_text = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:400]}"
        for m in historial
    )

    ctx_ords = ""
    if last_ordinances:
        ctx_ords = f"\nOrdenanzas discutidas previamente: {', '.join(last_ordinances)}"

    try:
        response = await aclient.responses.create(
            model="gpt-5-mini",
            input=f"""Historial de conversación:
{hist_text}
{ctx_ords}

Pregunta actual del usuario: "{pregunta}"

Si la pregunta actual es una continuación o se refiere al historial, reformulala como una pregunta completa e independiente que incluya todo el contexto necesario para buscar en una base de datos. Incluí números de ordenanza si aplica.
Si la pregunta actual es sobre un tema NUEVO y no tiene relación con el historial, devolvela tal cual.

Respondé SOLO con la pregunta reformulada, sin explicaciones ni comillas.""",
            max_output_tokens=150,
            timeout=8,
        )
        reformulada = response.output_text.strip()
        if reformulada and len(reformulada) > 3:
            if reformulada != pregunta:
                print(f"  Reformulada: '{pregunta}' -> '{reformulada}'")
            return reformulada
    except Exception as e:
        print(f"Error en reformulacion: {e}")

    return pregunta


def _formatear_historial(historial: list) -> str:
    """Formatea el historial para inyectar en prompts de GPT."""
    if not historial:
        return ""
    lines = []
    for m in historial:
        rol = "Usuario" if m["role"] == "user" else "Asistente"
        lines.append(f"{rol}: {m['content'][:600]}")
    return "\n".join(lines)


def _actualizar_memoria(
    conversation_id: str,
    pregunta_usuario: str,
    respuesta_texto: str,
    ordenanzas_citadas=None,
):
    state = _get_memory_state(conversation_id)
    history = state.setdefault("history", [])
    history.append({"role": "user", "content": pregunta_usuario})
    history.append({"role": "assistant", "content": (respuesta_texto or "")[:1500]})
    if len(history) > MAX_HISTORY:
        state["history"] = history[-MAX_HISTORY:]
    # Guardar últimas ordenanzas citadas para boost en follow-ups
    if ordenanzas_citadas:
        state["last_ordinances"] = list(ordenanzas_citadas)[:10]


def _boost_ordenanzas_previas(resultados: list, last_ords: list) -> list:
    """Si es follow-up, asegura que chunks de ordenanzas previas estén en resultados."""
    ords_en_resultados = {r.get("numero_ordenanza") for r in resultados}
    faltantes = [o for o in last_ords if o not in ords_en_resultados]
    if not faltantes:
        return resultados
    from chat_engine import normalizar_numero
    for num_ord in faltantes[:3]:
        num_norm = normalizar_numero(num_ord)
        chunks_agregados = 0
        for i, meta in enumerate(engine_metadatos):
            if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
                chunk, _ = obtener_chunk_y_meta_seguro(i)
                if chunk:
                    resultados.append({"chunk_texto": chunk, **meta})
                    chunks_agregados += 1
                    if chunks_agregados >= 3:
                        break
    return resultados


NO_RESULT_MESSAGE = (
    "No encontré información relevante en el Digesto Digital para tu consulta. "
    "Intentá reformular la pregunta con términos más específicos, "
    "o indicar el número de ordenanza si lo conocés."
)
def _resultados_son_relevantes(resultados: list) -> bool:
    """Verifica si los resultados tienen calidad suficiente para enviar a GPT."""
    if not resultados:
        return False
    for r in resultados:
        if r.get("score_semantico", 0) >= SIMILARITY_THRESHOLD:
            return True
        if r.get("coincidencias_textuales", 0) >= 2:
            return True
        # Resultados de CASO 1 (búsqueda directa por número de ordenanza)
        # no tienen scores pero son coincidencias exactas → siempre relevantes
        if (
            r.get("numero_ordenanza")
            and "score_semantico" not in r
            and "coincidencias_textuales" not in r
        ):
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
    _cleanup_stale_conversations()
    if not q.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    conversation_id = _get_or_create_conversation_id(q.conversation_id)
    state = _get_memory_state(conversation_id)
    historial = state.get("history", [])
    last_ords = state.get("last_ordinances", [])
    pregunta_efectiva = await _reformular_si_follow_up(
        q.pregunta, historial, last_ords
    )

    loop = asyncio.get_event_loop()
    resultados = await loop.run_in_executor(None, buscar_similares, pregunta_efectiva)

    if not _resultados_son_relevantes(resultados):
        _actualizar_memoria(conversation_id, q.pregunta, NO_RESULT_MESSAGE)
        return Answer(respuesta=NO_RESULT_MESSAGE, documentos=[])

    # Intentar resolvers determinísticos primero (antes del rerank)
    resultado_resolver = await loop.run_in_executor(
        None, _intentar_resolver_deterministico, pregunta_efectiva, resultados
    )

    if resultado_resolver:
        respuesta_texto = resultado_resolver["respuesta"]
        ordenanzas_citadas = set(resultado_resolver.get("ordenanzas_citadas", []))
        documentos_info = construir_documentos_info(resultados, ordenanzas_citadas)
        _actualizar_memoria(
            conversation_id, q.pregunta, respuesta_texto, ordenanzas_citadas
        )
        return Answer(respuesta=respuesta_texto, documentos=documentos_info)

    # LLM Reranking: GPT filtra los chunks más relevantes antes de generar
    resultados = await rerank_con_llm(pregunta_efectiva, resultados, top_n=7)

    # Boost DESPUÉS del rerank: así no se eliminan los chunks inyectados
    if last_ords and _parece_follow_up(q.pregunta):
        resultados = _boost_ordenanzas_previas(resultados, last_ords)

    contexto = armar_contexto(resultados)
    historial_texto = _formatear_historial(historial)
    resultado_gpt = await preguntar_a_gpt(
        pregunta_efectiva, contexto, resultados, historial_texto
    )
    respuesta_texto = resultado_gpt["respuesta"]
    ordenanzas_citadas = set(resultado_gpt.get("ordenanzas_citadas", []))
    documentos_info = construir_documentos_info(resultados, ordenanzas_citadas)
    _actualizar_memoria(
        conversation_id, q.pregunta, respuesta_texto, ordenanzas_citadas
    )

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
    _cleanup_stale_conversations()
    if not q.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacia.")

    async def generate():
        conversation_id = _get_or_create_conversation_id(q.conversation_id)
        state = _get_memory_state(conversation_id)
        historial = state.get("history", [])
        last_ords = state.get("last_ordinances", [])
        pregunta_efectiva = await _reformular_si_follow_up(
            q.pregunta, historial, last_ords
        )

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
            # Resolver respondió — streaming simulado char-a-char
            respuesta_texto = resultado_resolver.get("respuesta", "")
            ordenanzas_citadas = set(resultado_resolver.get("ordenanzas_citadas", []))
            _actualizar_memoria(
                conversation_id, q.pregunta, respuesta_texto, ordenanzas_citadas
            )

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

        # 3. Sin resolver → LLM Reranking + Streaming real de OpenAI
        resultados = await rerank_con_llm(pregunta_efectiva, resultados, top_n=7)

        # Boost DESPUÉS del rerank: así no se eliminan los chunks inyectados
        if last_ords and _parece_follow_up(q.pregunta):
            resultados = _boost_ordenanzas_previas(resultados, last_ords)

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
        historial_texto = _formatear_historial(historial)
        seccion_historial = f"\nHISTORIAL DE CONVERSACIÓN (para contexto de follow-ups):\n{historial_texto}\n" if historial_texto else ""

        if tipo_pregunta == "numero_especifico":
            prompt = f"""Eres el Digesto Digital de Villa María. El usuario quiere saber de qué trata la ordenanza N° {pregunta_efectiva}.

Usando SOLO la información del contexto, escribe un resumen claro y conciso.
Comienza tu respuesta con "La Ordenanza N° {pregunta_efectiva}" y describe su contenido principal.
{seccion_historial}
Contexto:
{contexto}

Responde directamente en Markdown limpio. NO uses formato JSON."""
        else:
            prompt = f"""(Dando formato como para poner en una pagina web) Eres el Digesto Digital de Villa María. Responde de forma completa y clara usando SOLO la información del contexto.

REGLAS ESTRICTAS:
1. El contexto puede contener ordenanzas que NO son relevantes a la pregunta. Antes de responder, verificá que cada ordenanza que cites realmente contenga información directamente relacionada con lo que el usuario pregunta. NO menciones ordenanzas que solo contengan palabras sueltas coincidentes pero en contextos diferentes.
2. NUNCA inventes números de ordenanza. Solo podés citar estas ordenanzas: {lista_nums}. Si la respuesta no está en el contexto, decilo claramente.
3. Si la pregunta pide enumerar elementos (acuerdos, artículos, partes, etc.), listarlos todos con viñetas Markdown. Si es una pregunta simple, responde en 1-2 oraciones.
4. El Intendente Municipal actual (2025) es Eduardo Luis Accastello. No menciones intendentes de gestiones anteriores como actuales.
{seccion_historial}
Contexto (ordenanzas disponibles: {lista_nums}):
{contexto}

Pregunta: {pregunta_efectiva}

Responde directamente en Markdown limpio. NO uses formato JSON."""

        respuesta_acumulada = ""

        try:
            async with aclient.responses.stream(
                model="gpt-5-mini",
                input=prompt,
                max_output_tokens=1200,
            ) as stream:
                async for text in stream.text_stream:
                    respuesta_acumulada += text
                    yield f"data: {json.dumps({'tipo': 'chunk', 'texto': text})}\n\n"

        except Exception as e:
            print(f"Error en OpenAI Async Stream: {e}")
            error_msg = (
                "Hubo un error al generar la respuesta. Por favor, intenta de nuevo."
            )
            yield f"data: {json.dumps({'tipo': 'chunk', 'texto': error_msg})}\n\n"
            respuesta_acumulada = error_msg

        # Actualizar memoria y reenviar documentos con citas
        citas_extraidas = []
        for n in numeros_disponibles:
            if n in respuesta_acumulada:
                citas_extraidas.append(n)
        _actualizar_memoria(
            conversation_id,
            q.pregunta,
            respuesta_acumulada,
            set(citas_extraidas) if citas_extraidas else None,
        )

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
