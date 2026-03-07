import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from chat_engine import (
    CHAT_MODEL_COMPLEX,
    CHAT_MODEL_DEFAULT,
    MAX_DOCS_MOSTRADOS,
    SIMILARITY_THRESHOLD,
    aclient,
    armar_contexto,
    buscar_similares,
    chunks as engine_chunks,
    construir_respuesta_extractiva_local,
    construir_pregunta_aclaratoria,
    detectar_tipo_pregunta,
    ensure_index_loaded,
    extraer_articulo_de_pregunta,
    extraer_numero_ordenanza_de_pregunta,
    extraer_texto_respuesta_modelo,
    generar_embedding_local,
    metadatos as engine_metadatos,
    normalizar_numero,
    normalizar_texto_para_busqueda,
    obtener_chunk_y_meta_seguro,
    priorizar_resultados_para_respuesta,
    preguntar_a_gpt,
    rerank_con_llm,
    resolver_ordenanza_extrema,
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
_CLARIFICATION_PREFIXES = (
    "es ",
    "es un",
    "es una",
    "zona ",
    "de ",
    "del ",
    "para ",
    "la que",
    "el que",
    "quiero",
    "busco",
)


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
    try:
        generar_embedding_local("test de calentamiento")
    except Exception as e:
        print(f"No se pudo pre-calentar embeddings: {e}")
    print("Sistema listo para recibir consultas")


class Question(BaseModel):
    pregunta: str
    conversation_id: str | None = None


class Answer(BaseModel):
    respuesta: str
    documentos: list
    escalado: bool = False
    modelo_usado: str | None = None


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


def _extraer_foco_consulta(pregunta: str) -> str | None:
    texto = (pregunta or "").strip().strip("¿?.,;: ")
    if not texto:
        return None

    if re.fullmatch(r"(?:ordenanza\s*[n°º]?\s*)?\d{4,5}", texto, re.IGNORECASE):
        return texto

    if re.fullmatch(
        r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}",
        texto,
    ):
        return texto

    if len(texto.split()) == 1 and not any(ch.isdigit() for ch in texto):
        return texto

    return None


def _parece_respuesta_a_aclaracion(pregunta: str) -> bool:
    texto = (pregunta or "").lower().strip(" ¿?.,;:")
    if not texto:
        return False
    if len(texto.split()) <= 10:
        return True
    return any(texto.startswith(pref) for pref in _CLARIFICATION_PREFIXES)


def _combinar_con_aclaracion_pendiente(pregunta: str, state: dict) -> tuple[str, bool]:
    pendiente = state.get("pending_clarification")
    if not pendiente or not _parece_respuesta_a_aclaracion(pregunta):
        return pregunta, False

    original = (pendiente.get("original_question") or "").strip()
    if not original:
        return pregunta, False

    state.pop("pending_clarification", None)
    return f"{original}. Aclaración del usuario: {pregunta.strip()}", True


def _reformular_follow_up_deterministico(
    pregunta: str,
    last_focus: str | None = None,
    last_ordinances: list | None = None,
) -> str:
    if not _parece_follow_up(pregunta):
        return pregunta

    p_lower = pregunta.lower()
    if last_focus and any(
        token in p_lower
        for token in (
            "se lo",
            "se la",
            "lo menciona",
            "la menciona",
            "en qué ordenanzas",
            "en que ordenanzas",
            "sobre eso",
            "sobre él",
            "sobre el",
            "sobre ella",
            "sobre la",
        )
    ):
        return f"{pregunta.strip()} sobre {last_focus}"

    if last_focus and len(pregunta.split()) <= 5:
        return f"{last_focus}. {pregunta.strip()}"

    if (
        last_ordinances
        and not re.search(r"\d{4,5}", pregunta)
        and any(
            token in p_lower
            for token in (
                "art",
                "artículo",
                "articulo",
                "vigencia",
                "directorio",
                "sindicatura",
                "incompatibil",
                "plazo",
                "mayoría",
                "mayoria",
                "funcionamiento",
            )
        )
    ):
        return f"{pregunta.strip()} sobre la Ordenanza {last_ordinances[0]}"

    return pregunta


async def _reformular_si_follow_up(
    pregunta: str,
    historial: list,
    last_ordinances: list | None = None,
    last_focus: str | None = None,
) -> str:
    """
    Si hay historial y la pregunta parece follow-up, usa GPT para reformularla
    como pregunta completa e independiente para retrieval.
    """
    if not historial or not _parece_follow_up(pregunta):
        return pregunta

    pregunta_base = _reformular_follow_up_deterministico(
        pregunta, last_focus, last_ordinances
    )
    if pregunta_base != pregunta:
        return pregunta_base

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
    focus = _extraer_foco_consulta(pregunta_usuario)
    if focus:
        state["last_focus"] = focus
    # Guardar últimas ordenanzas citadas para boost en follow-ups
    if ordenanzas_citadas:
        state["last_ordinances"] = list(ordenanzas_citadas)[:10]
    elif state.get("last_ordinances") and state.get("pending_clarification"):
        state["last_ordinances"] = state["last_ordinances"][:10]


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
        if r.get("coincidencias_textuales", 0) >= 1.5:
            return True
        if r.get("score_textual_local", 0) >= 2:
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


def _score_resultado_confianza(resultado: dict) -> float:
    score = float(resultado.get("_score_contexto", 0) or 0)
    score = max(score, float(resultado.get("score_combinado", 0) or 0))
    score = max(score, float(resultado.get("score_textual_local", 0) or 0))
    score = max(score, float(resultado.get("coincidencias_textuales", 0) or 0))
    score = max(score, float(resultado.get("score_semantico", 0) or 0) * 4.0)
    return score


def _seleccionar_modelo_respuesta(
    pregunta: str, resultados: list, es_follow_up: bool = False
) -> dict:
    top = priorizar_resultados_para_respuesta(pregunta, resultados, 5)
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    tokens = [t for t in texto_norm.split() if t]
    num_ordenanza = extraer_numero_ordenanza_de_pregunta(pregunta or "")
    articulo = extraer_articulo_de_pregunta(pregunta or "")
    scores = [_score_resultado_confianza(r) for r in top]
    mejor_score = scores[0] if scores else 0.0
    segundo_score = scores[1] if len(scores) > 1 else 0.0
    diferencia = mejor_score - segundo_score
    ordenanzas_top = [
        str(r.get("numero_ordenanza", ""))
        for r in top
        if r.get("numero_ordenanza") and r.get("numero_ordenanza") != "desconocido"
    ]
    ordenanzas_unicas = list(dict.fromkeys(ordenanzas_top))
    dificultad = 0.0
    razones = []

    if mejor_score < 2.6:
        dificultad += 1.0
        razones.append("retrieval_debil")
    elif len(scores) > 1 and diferencia < 0.45:
        dificultad += 0.7
        razones.append("retrieval_ambiguo")

    if len(ordenanzas_unicas) >= 3:
        dificultad += 0.8
        razones.append("varias_ordenanzas_compiten")

    compleja = any(
        termino in texto_norm
        for termino in (
            "articulo",
            "deroga",
            "derogacion",
            "modifica",
            "vigencia",
            "incompatibil",
            "compar",
            "diferencia",
            "presupuesto",
            "plazo",
            "requisito",
            "excepto",
            "salvo",
            "condicion",
            "condiciones",
            "quien",
        )
    )
    if compleja:
        dificultad += 0.5
        razones.append("consulta_normativa_compleja")

    if len(tokens) >= 18:
        dificultad += 0.4
        razones.append("consulta_larga")

    if es_follow_up and len(tokens) <= 6 and not num_ordenanza:
        dificultad += 0.6
        razones.append("followup_corto")

    if num_ordenanza:
        num_norm = normalizar_numero(num_ordenanza)
        mismos = [
            r
            for r in top
            if normalizar_numero(str(r.get("numero_ordenanza", ""))) == num_norm
        ]
        if not mismos:
            dificultad += 1.2
            razones.append("ordenanza_no_anclada")
        elif len(ordenanzas_unicas) >= 2 and len(mismos) < 2:
            dificultad += 0.4
            razones.append("ordenanza_compite")

    if articulo:
        exactos = [r for r in top if str(r.get("_articulo_match") or "") == articulo]
        if not exactos:
            dificultad += 1.0
            razones.append("articulo_no_anclado")
        elif top and str(top[0].get("_articulo_match") or "") != articulo:
            dificultad += 0.4
            razones.append("articulo_no_lidera")

    dificil = dificultad >= 1.8
    return {
        "dificil": dificil,
        "modelo": CHAT_MODEL_COMPLEX if dificil else CHAT_MODEL_DEFAULT,
        "score": round(dificultad, 2),
        "razones": razones,
    }


def _asegurar_referencia_normativa(resultado: dict | None) -> dict | None:
    if not resultado:
        return resultado

    respuesta = (resultado.get("respuesta") or "").strip()
    ordenanzas = [str(x) for x in resultado.get("ordenanzas_citadas", []) if str(x)]
    if len(ordenanzas) != 1 or not respuesta:
        return resultado

    numero = ordenanzas[0]
    if (
        f"Ordenanza N° {numero}" in respuesta
        or f"Ordenanza Nº {numero}" in respuesta
        or f"Ordenanza {numero}" in respuesta
    ):
        return resultado

    copia = dict(resultado)
    copia["respuesta"] = f"Según la Ordenanza N° {numero}: {respuesta}"
    return copia


def _intentar_resolver_deterministico(pregunta: str, resultados: list) -> dict | None:
    """
    Ejecuta todos los resolvers determinísticos en orden de prioridad.
    Retorna dict {respuesta, ordenanzas_citadas} o None.
    """
    for resolver in [
        resolver_ordenanza_extrema,
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
                return _asegurar_referencia_normativa(resultado)
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
    last_focus = state.get("last_focus")
    es_follow_up = _parece_follow_up(q.pregunta)
    pregunta_preparada, _ = _combinar_con_aclaracion_pendiente(q.pregunta, state)
    pregunta_efectiva = await _reformular_si_follow_up(
        pregunta_preparada, historial, last_ords, last_focus
    )

    loop = asyncio.get_event_loop()
    resultados = await loop.run_in_executor(None, buscar_similares, pregunta_efectiva)

    if not _resultados_son_relevantes(resultados):
        state.pop("pending_clarification", None)
        _actualizar_memoria(conversation_id, q.pregunta, NO_RESULT_MESSAGE)
        return Answer(
            respuesta=NO_RESULT_MESSAGE,
            documentos=[],
            escalado=False,
            modelo_usado=None,
        )

    # Intentar resolvers determinísticos primero (antes del rerank)
    resultado_resolver = await loop.run_in_executor(
        None, _intentar_resolver_deterministico, pregunta_efectiva, resultados
    )

    if resultado_resolver:
        respuesta_texto = resultado_resolver["respuesta"]
        ordenanzas_citadas = set(resultado_resolver.get("ordenanzas_citadas", []))
        if not ordenanzas_citadas and "?" in respuesta_texto:
            state["pending_clarification"] = {
                "original_question": pregunta_preparada,
                "created_at": time.time(),
            }
        else:
            state.pop("pending_clarification", None)
        documentos_info = construir_documentos_info(resultados, ordenanzas_citadas)
        _actualizar_memoria(
            conversation_id, q.pregunta, respuesta_texto, ordenanzas_citadas
        )
        return Answer(
            respuesta=respuesta_texto,
            documentos=documentos_info,
            escalado=False,
            modelo_usado=None,
        )

    # LLM Reranking: GPT filtra los chunks más relevantes antes de generar
    resultados = await rerank_con_llm(pregunta_efectiva, resultados, top_n=7)

    # Boost DESPUÉS del rerank: así no se eliminan los chunks inyectados
    if last_ords and es_follow_up:
        resultados = _boost_ordenanzas_previas(resultados, last_ords)

    resultados = priorizar_resultados_para_respuesta(pregunta_efectiva, resultados, 7)
    decision_modelo = _seleccionar_modelo_respuesta(
        pregunta_efectiva, resultados, es_follow_up
    )
    contexto = armar_contexto(resultados)
    historial_texto = _formatear_historial(historial)
    resultado_gpt = await preguntar_a_gpt(
        pregunta_efectiva,
        contexto,
        resultados,
        historial_texto,
        modelo=decision_modelo["modelo"],
    )
    respuesta_texto = resultado_gpt["respuesta"]
    ordenanzas_citadas = set(resultado_gpt.get("ordenanzas_citadas", []))
    state.pop("pending_clarification", None)
    documentos_info = construir_documentos_info(resultados, ordenanzas_citadas)
    _actualizar_memoria(
        conversation_id, q.pregunta, respuesta_texto, ordenanzas_citadas
    )

    return Answer(
        respuesta=respuesta_texto,
        documentos=documentos_info,
        escalado=decision_modelo["dificil"],
        modelo_usado=decision_modelo["modelo"],
    )


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
        last_focus = state.get("last_focus")
        es_follow_up = _parece_follow_up(q.pregunta)
        pregunta_preparada, _ = _combinar_con_aclaracion_pendiente(
            q.pregunta, state
        )
        pregunta_efectiva = await _reformular_si_follow_up(
            pregunta_preparada, historial, last_ords, last_focus
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
            state.pop("pending_clarification", None)
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
            if not ordenanzas_citadas and "?" in respuesta_texto:
                state["pending_clarification"] = {
                    "original_question": pregunta_preparada,
                    "created_at": time.time(),
                }
            else:
                state.pop("pending_clarification", None)
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
        state.pop("pending_clarification", None)

        # Boost DESPUÉS del rerank: así no se eliminan los chunks inyectados
        if last_ords and es_follow_up:
            resultados = _boost_ordenanzas_previas(resultados, last_ords)

        resultados = priorizar_resultados_para_respuesta(pregunta_efectiva, resultados, 7)
        decision_modelo = _seleccionar_modelo_respuesta(
            pregunta_efectiva, resultados, es_follow_up
        )
        if decision_modelo["dificil"]:
            yield f"data: {json.dumps({'tipo': 'estado', 'estado': 'pensando', 'modelo': decision_modelo['modelo']})}\n\n"

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
        stream_final_response = None
        stream_error = None

        try:
            async with aclient.responses.stream(
                model=decision_modelo["modelo"],
                input=prompt,
                max_output_tokens=1200,
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        text = event.delta
                        respuesta_acumulada += text
                        yield f"data: {json.dumps({'tipo': 'chunk', 'texto': text})}\n\n"
                    elif event.type == "response.output_text.done" and not respuesta_acumulada:
                        text = getattr(event, "text", "") or ""
                        if text:
                            respuesta_acumulada = text
                            yield f"data: {json.dumps({'tipo': 'chunk', 'texto': text})}\n\n"
                    elif event.type == "response.completed":
                        stream_final_response = getattr(event, "response", None)

                if stream_final_response is None:
                    try:
                        stream_final_response = await stream.get_final_response()
                    except Exception as final_error:
                        stream_error = final_error

            if not respuesta_acumulada and stream_final_response is not None:
                texto_final = extraer_texto_respuesta_modelo(stream_final_response)
                if texto_final:
                    respuesta_acumulada = texto_final
                    yield f"data: {json.dumps({'tipo': 'chunk', 'texto': texto_final})}\n\n"

        except Exception as e:
            print(f"Error en OpenAI Async Stream: {e}")
            stream_error = e

        if stream_error and not respuesta_acumulada.strip():
            print(f"OpenAI Async Stream sin respuesta util, aplicando fallback: {stream_error}")

        if not respuesta_acumulada.strip():
            respuesta_extractiva = construir_respuesta_extractiva_local(
                pregunta_efectiva, resultados
            )
            if respuesta_extractiva:
                respuesta_acumulada = respuesta_extractiva.get("respuesta", "").strip()
                if respuesta_acumulada:
                    yield f"data: {json.dumps({'tipo': 'chunk', 'texto': respuesta_acumulada})}\n\n"

        if not respuesta_acumulada.strip():
            respuesta_acumulada = (
                "Hubo un error al generar la respuesta. Por favor, intenta de nuevo."
            )
            yield f"data: {json.dumps({'tipo': 'chunk', 'texto': respuesta_acumulada})}\n\n"

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
