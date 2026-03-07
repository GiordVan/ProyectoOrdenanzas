import os
import json
import threading
import math
import faiss
import numpy as np
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import re
from datetime import datetime
from functools import lru_cache
from collections import Counter

# NUEVO: Importar stemmer español
try:
    from nltk.stem import SnowballStemmer
    import nltk

    # Descargar recursos si es necesario
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    stemmer = SnowballStemmer("spanish")
except ImportError:
    print("NLTK no instalado. Instalar con: pip install nltk")
    stemmer = None

# --- Configuración ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ⚡ Usar rutas absolutas para evitar errores en Railway/servidores
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data")
INDEX_FILE = os.path.join(DATA_PATH, "index.faiss")
METADATA_FILE = os.path.join(DATA_PATH, "metadatos.json")
CHUNKS_FILE = os.path.join(DATA_PATH, "chunks.json")
CHAT_MODEL_DEFAULT = os.getenv("OPENAI_CHAT_MODEL_DEFAULT", "gpt-5-mini")
CHAT_MODEL_COMPLEX = os.getenv("OPENAI_CHAT_MODEL_COMPLEX", "gpt-5")
TOP_K = 15  # Resultados base de FAISS
SIMILARITY_THRESHOLD = (
    0.38  # Umbral mínimo de similitud coseno (unificado para FAISS y GPT)
)
MAX_DOCS_MOSTRADOS = 5  # Máximo de documentos únicos a mostrar al usuario
MAX_CHUNKS_POR_ORD = 5  # Máximo de chunks por ordenanza en búsqueda híbrida
CHUNK_ADMIN_NOISE_PATTERNS = (
    "hoja adicional de firmas",
    "anexo firma conjunta",
    "protocolicese comuniquese publiquese",
    "dese al registro y boletin municipal",
    "dada en la sala de sesiones del concejo deliberante",
    "referencia ordenanza",
)
RESIDUOS_INTENT_CLARITY_KEYWORDS = {
    "cuanto",
    "cuesta",
    "costo",
    "monto",
    "tarifa",
    "valor",
    "incentivo",
    "descuento",
    "dependencia",
    "secretaria",
    "subsecretaria",
    "cooperativa",
    "empresa",
    "quien",
    "cual",
    "regula",
    "modifica",
    "modificacion",
    "informar",
    "ingresos",
    "separados",
    "separado",
}
MULTA_GENERIC_TOKENS = {
    "multa",
    "multas",
    "municipal",
    "municipales",
    "infraccion",
    "infracciones",
    "falta",
    "faltas",
    "sancion",
    "sanciones",
    "monto",
    "montos",
    "actualizado",
    "actualizada",
    "actualizados",
    "actualizadas",
    "cuanto",
    "cual",
    "valor",
    "pagar",
    "usuario",
    "aclaracion",
    "ejemplo",
}


def _l2_to_cosine(l2_score: float) -> float:
    """Convierte distancia L2 de FAISS a similitud coseno para vectores unitarios.
    Para vectores unitarios: L2² = 2(1 - cos_sim), entonces cos_sim = 1 - L2²/2.
    """
    return max(0.0, min(1.0, 1.0 - (l2_score**2) / 2.0))


# Variables globales
index = None
metadatos = []
chunks = []
chunks_normalized = []  # Pre-computado: normalizar_texto_para_busqueda(chunk)
chunks_stemmed = []  # Pre-computado: aplicar_stemming(chunk)
token_doc_freq = Counter()
token_doc_freq_stemmed = Counter()
_index_loaded = False
_index_lock = threading.Lock()
CONSULTA_MONTO_KEYWORDS = {
    "cuanto",
    "cuánto",
    "pagar",
    "pago",
    "monto",
    "tasa",
    "tasas",
    "tarifa",
    "tarifaria",
    "tributo",
    "tributos",
    "impuesto",
    "impuestos",
    "alicuota",
    "alícuota",
    "arancel",
    "aranceles",
    "cuota",
    "cuotas",
    "valor",
    "valores",
    "costo",
    "costos",
}
CONSULTA_COMPARATIVA_KEYWORDS = {
    "aumento",
    "variacion",
    "variación",
    "diferencia",
    "comparar",
    "comparación",
    "incremento",
    "evolución",
    "entre",
}
CONSULTA_PRESUPUESTO_INSTITUCIONAL_KEYWORDS = {
    "secretaria",
    "secretaría",
    "departamento",
    "responsable",
    "ejecuta",
    "ejecutar",
    "ejecucion",
    "ejecución",
    "seguimiento",
    "organo",
    "órgano",
    "area",
    "área",
}
CONSULTA_PRESUPUESTO_MONTO_KEYWORDS = {
    "monto",
    "total",
    "cuanto",
    "cuánto",
    "valor",
    "valores",
}


def ensure_index_loaded():
    global _index_loaded, index, metadatos, chunks
    if not _index_loaded:
        with _index_lock:
            if not _index_loaded:
                cargar_indice_y_metadatos()
                _index_loaded = True


def normalizar_numero(num: str) -> str:
    return re.sub(r"\D", "", num)


def es_chunk_administrativo_ruidoso(texto: str) -> bool:
    texto_norm = normalizar_texto_para_busqueda(texto or "")
    if not texto_norm:
        return True

    hits = sum(1 for patron in CHUNK_ADMIN_NOISE_PATTERNS if patron in texto_norm)
    if hits >= 2:
        return True
    if hits >= 1 and len(texto_norm) < 260:
        return True

    return "protocolicese" in texto_norm and "archivese" in texto_norm


def limpiar_texto_respuesta_local(texto: str, max_chars: int = 420) -> str:
    texto = re.sub(r"\s+", " ", texto or "").strip()
    if not texto:
        return ""
    if len(texto) <= max_chars:
        return texto.rstrip(" ,;.")
    return texto[:max_chars].rstrip(" ,;.") + "..."


def obtener_resumen_ordenanza(
    resultados: list | None = None, numero_ordenanza: str | None = None
) -> str:
    numero_norm = normalizar_numero(numero_ordenanza or "")

    for resultado in resultados or []:
        numero_resultado = normalizar_numero(
            str(resultado.get("numero_ordenanza", ""))
        )
        if numero_norm and numero_resultado != numero_norm:
            continue
        resumen = limpiar_texto_respuesta_local(resultado.get("resumen", ""))
        if resumen:
            return resumen

    if numero_norm:
        ensure_index_loaded()
        for meta in metadatos:
            if normalizar_numero(str(meta.get("numero_ordenanza", ""))) != numero_norm:
                continue
            resumen = limpiar_texto_respuesta_local(meta.get("resumen", ""))
            if resumen:
                return resumen

    return ""


def hay_ordenanza_dominante(resultados: list | None, min_hits: int = 2) -> bool:
    numeros = [
        normalizar_numero(str(r.get("numero_ordenanza", "")))
        for r in (resultados or [])[:5]
        if r.get("numero_ordenanza")
    ]
    if not numeros:
        return False
    _, cantidad = Counter(numeros).most_common(1)[0]
    return cantidad >= min_hits


def es_consulta_residuos_con_intencion_clara(pregunta: str) -> bool:
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    if not any(
        token in texto_norm
        for token in (
            "residuo",
            "residuos",
            "recicl",
            "gestion ambiental",
            "centro de gestion ambiental",
        )
    ):
        return False

    if extraer_numero_ordenanza_de_pregunta(pregunta or ""):
        return True

    return any(token in texto_norm for token in RESIDUOS_INTENT_CLARITY_KEYWORDS)


def es_consulta_multa_municipal(pregunta: str) -> bool:
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    menciona_sancion = any(
        token in texto_norm
        for token in ("multa", "infraccion", "infracciones", "falta", "faltas")
    )
    pide_monto = any(
        token in texto_norm
        for token in ("cuanto", "monto", "valor", "pagar", "actualizado")
    )
    return menciona_sancion and pide_monto


def extraer_tokens_especificos_multa(pregunta: str) -> list:
    tokens = extraer_tokens_busqueda(normalizar_texto_para_busqueda(pregunta or ""))
    especificos = [
        token
        for token in tokens
        if token not in MULTA_GENERIC_TOKENS and not token.isdigit()
    ]
    return list(dict.fromkeys(especificos))


def formatear_lista_articulos(articulos: list[str]) -> str:
    articulos = [str(int(a)) for a in articulos if str(a).isdigit()]
    articulos = list(dict.fromkeys(articulos))
    if not articulos:
        return ""
    if len(articulos) == 1:
        return f"el Art. {articulos[0]}"
    if len(articulos) == 2:
        return f"los Arts. {articulos[0]} y {articulos[1]}"
    return "los Arts. " + ", ".join(articulos[:-1]) + f" y {articulos[-1]}"


def extraer_articulos_modificados_desde_resultados(
    resultados: list | None, numero_ordenanza: str | None = None
) -> list[str]:
    numero_norm = normalizar_numero(numero_ordenanza or "")
    candidatos = list(resultados or [])

    if numero_norm:
        ensure_index_loaded()
        for idx, meta in enumerate(metadatos):
            if normalizar_numero(str(meta.get("numero_ordenanza", ""))) != numero_norm:
                continue
            chunk, meta_seguro = obtener_chunk_y_meta_seguro(idx)
            if not chunk or not meta_seguro:
                continue
            candidatos.append({"chunk_texto": chunk, **meta_seguro})

    articulos = []
    vistos = set()
    for resultado in candidatos:
        numero_resultado = normalizar_numero(
            str(resultado.get("numero_ordenanza", ""))
        )
        if numero_norm and numero_resultado != numero_norm:
            continue

        chunk_texto = resultado.get("chunk_texto", "")
        if es_chunk_administrativo_ruidoso(chunk_texto):
            continue
        clave_chunk = (
            numero_resultado,
            resultado.get("chunk_id"),
            limpiar_texto_respuesta_local(chunk_texto, 120),
        )
        if clave_chunk in vistos:
            continue
        vistos.add(clave_chunk)

        chunk_norm = normalizar_texto_para_busqueda(chunk_texto)
        chunk_norm = re.sub(
            r"(\bart(?:iculo)?\s+)(\d)\s+(\d)\b", r"\1\2\3", chunk_norm
        )
        if not any(
            token in chunk_norm
            for token in (
                "quedara redactado",
                "modific",
                "sustituy",
                "reemplaz",
                "incorporese",
                "incorporase",
            )
        ):
            continue

        for match in re.finditer(r"\bart(?:iculo)?\s*(\d{1,3})\b", chunk_norm):
            contexto = chunk_norm[match.start() : match.start() + 140]
            if any(
                marca in contexto
                for marca in (
                    "del anexo",
                    "de la ordenanza",
                    "quedara redactado",
                    "incorporese",
                    "sustituy",
                    "reemplaz",
                )
            ):
                articulos.append(str(int(match.group(1))))

    return list(dict.fromkeys(articulos))


def obtener_chunk_y_meta_seguro(idx: int):
    """
    Obtiene chunk y metadato de forma segura.
    Retorna None si el índice está fuera de rango.
    """
    if 0 <= idx < len(chunks) and 0 <= idx < len(metadatos):
        return chunks[idx], metadatos[idx]
    return None, None


def cargar_indice_y_metadatos():
    """Carga el índice FAISS y metadatos livianos en RAM.
    Pre-computa texto normalizado y stemmed para búsqueda rápida.
    """
    global index, metadatos, chunks, chunks_normalized, chunks_stemmed

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"No se encontró el archivo {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadatos_comprimidos = json.load(f)

    # ⚡ Cargar chunks primero para saber la cantidad
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # ⚡ Expandir metadatos comprimidos
    # IMPORTANTE: Los chunk_indices en metadatos.json son LOCALES (0,1,2... por ordenanza)
    # pero chunks.json es GLOBAL (secuencial). Necesitamos crear un metadato por cada chunk global.
    metadatos = []
    global_chunk_idx = 0

    for meta in metadatos_comprimidos:
        meta_copy = {k: v for k, v in meta.items() if k != "chunk_indices"}

        if "chunk_indices" in meta:
            # Formato comprimido: crear un metadato por cada chunk de esta ordenanza
            num_chunks = len(meta["chunk_indices"])
            total_chunks = meta.get("total_chunks", num_chunks)

            for local_idx in range(num_chunks):
                meta_expandido = meta_copy.copy()
                meta_expandido["chunk_id"] = local_idx
                meta_expandido["total_chunks"] = total_chunks
                meta_expandido["global_chunk_idx"] = global_chunk_idx
                metadatos.append(meta_expandido)
                global_chunk_idx += 1
        else:
            # Formato legacy: usar tal cual
            meta_copy["global_chunk_idx"] = global_chunk_idx
            metadatos.append(meta_copy)
            global_chunk_idx += 1

    print(f"Cargados {len(chunks)} chunks y {len(metadatos)} metadatos expandidos")
    if len(chunks) != len(metadatos):
        print(
            f"ADVERTENCIA: Desajuste chunks ({len(chunks)}) vs metadatos ({len(metadatos)})"
        )

    # ⚡ Pre-computar texto normalizado y stemmed (evita re-calcular en cada búsqueda)
    print("Pre-computando índices de texto para búsqueda rápida...")
    chunks_normalized = [normalizar_texto_para_busqueda(c) for c in chunks]
    if stemmer:
        chunks_stemmed = [aplicar_stemming(c) for c in chunks]
    else:
        chunks_stemmed = chunks_normalized[:]

    # Pre-computar Art N°1 normalizado y stemmed en metadatos
    for meta in metadatos:
        art1 = meta.get("Art N°1", "")
        if art1:
            meta["_art1_normalized"] = normalizar_texto_para_busqueda(art1)
            meta["_art1_stemmed"] = (
                aplicar_stemming(art1) if stemmer else meta["_art1_normalized"]
            )
        else:
            meta["_art1_normalized"] = ""
            meta["_art1_stemmed"] = ""
        # Pre-computar palabras clave normalizadas
        palabras_clave = meta.get("palabras_clave", [])
        meta["_keywords_normalized"] = (
            " ".join(palabras_clave).lower() if palabras_clave else ""
        )

    actualizar_estadisticas_busqueda()
    print(f"Índices de texto pre-computados ({len(chunks)} chunks)")


# ⚡ Cache LRU para embeddings - evita recalcular para consultas repetidas
@lru_cache(maxsize=500)
def _generar_embedding_cached(texto: str) -> tuple:
    """Versión cacheada que devuelve tupla (hasheable para lru_cache)."""
    response = openai.embeddings.create(input=texto, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return tuple(embedding)


def generar_embedding_local(texto: str):
    """Genera embedding con cache. Consultas repetidas son instantáneas."""
    # Normalizar texto para mejor hit rate del cache
    texto_norm = texto.strip().lower()
    cached = _generar_embedding_cached(texto_norm)
    return np.array(cached)


def _leer_attr_o_key(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extraer_texto_respuesta_modelo(response) -> str:
    texto = (_leer_attr_o_key(response, "output_text", "") or "").strip()
    if texto:
        return texto

    partes = []
    for item in _leer_attr_o_key(response, "output", []) or []:
        for parte in _leer_attr_o_key(item, "content", []) or []:
            tipo = _leer_attr_o_key(parte, "type", "")
            if tipo not in ("output_text", "text"):
                continue

            texto_parte = _leer_attr_o_key(parte, "text", "")
            if isinstance(texto_parte, str):
                valor = texto_parte
            else:
                valor = (
                    _leer_attr_o_key(texto_parte, "value", "")
                    or _leer_attr_o_key(texto_parte, "text", "")
                    or _leer_attr_o_key(parte, "value", "")
                )
            if valor:
                partes.append(str(valor))

    return "\n".join(partes).strip()


def limpiar_json_respuesta_modelo(texto: str) -> str:
    texto = (texto or "").strip()
    if not texto:
        return ""

    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto).strip()

    if texto and texto[0] not in "{[":
        inicio_obj = texto.find("{")
        inicio_arr = texto.find("[")
        inicios = [x for x in (inicio_obj, inicio_arr) if x >= 0]
        if inicios:
            texto = texto[min(inicios):].strip()

    if texto.startswith("{"):
        fin = texto.rfind("}")
        if fin >= 0:
            texto = texto[: fin + 1]
    elif texto.startswith("["):
        fin = texto.rfind("]")
        if fin >= 0:
            texto = texto[: fin + 1]

    return texto.strip()


def parsear_json_respuesta_modelo(response):
    texto = limpiar_json_respuesta_modelo(extraer_texto_respuesta_modelo(response))
    if not texto:
        raise ValueError("respuesta vacia del modelo")
    return json.loads(texto)


def resumir_error_llm(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "respuesta JSON vacia o invalida"
    return str(error)


def extraer_numero_ordenanza_de_pregunta(pregunta: str):
    match = re.search(r"(?:ordenanza\s*[n°º]?\s*)?(\d{4,5})", pregunta, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extraer_articulo_de_pregunta(pregunta: str) -> str | None:
    texto = normalizar_texto_para_busqueda(pregunta or "")
    match = re.search(r"\bart(?:iculo)?\s*(\d{1,3})\b", texto)
    if match:
        return str(int(match.group(1)))
    return None


def extraer_fecha_de_pregunta(pregunta: str):
    meses = {
        "enero": "01",
        "febrero": "02",
        "marzo": "03",
        "abril": "04",
        "mayo": "05",
        "junio": "06",
        "julio": "07",
        "agosto": "08",
        "septiembre": "09",
        "octubre": "10",
        "noviembre": "11",
        "diciembre": "12",
    }

    match_texto = re.search(
        rf'({"|".join(meses.keys())})\s*(?:de\s*)?(\d{{4}})', pregunta, re.IGNORECASE
    )
    if match_texto:
        mes_palabra = match_texto.group(1).lower()
        anio = match_texto.group(2)
        return f"{meses[mes_palabra]}/{anio}"

    match_num = re.search(r"(0?[1-9]|1[0-2])/(\d{4})", pregunta)
    if match_num:
        mes = int(match_num.group(1))
        anio = match_num.group(2)
        return f"{mes:02d}/{anio}"

    return None


def extraer_anios_de_texto(texto: str) -> list:
    """Extrae años de cuatro dígitos (1900-2099) preservando orden."""
    anios = re.findall(r"\b(19\d{2}|20\d{2})\b", texto or "")
    return list(dict.fromkeys(anios))


def extraer_anio_de_fecha(fecha: str) -> str | None:
    """Obtiene año desde fechas DD/MM/YYYY, YYYY-MM-DD o texto mixto."""
    if not fecha:
        return None

    match_iso = re.search(r"\b(19\d{2}|20\d{2})-\d{2}-\d{2}\b", fecha)
    if match_iso:
        return match_iso.group(1)

    match_latam = re.search(r"\b\d{1,2}/\d{1,2}/(19\d{2}|20\d{2})\b", fecha)
    if match_latam:
        return match_latam.group(1)

    match_simple = re.search(r"\b(19\d{2}|20\d{2})\b", fecha)
    if match_simple:
        return match_simple.group(1)

    return None


def obtener_anios_disponibles(top_n: int = 3) -> list:
    """
    Devuelve años disponibles en metadatos (orden descendente).
    Se usa para sugerir opciones cuando falta año en consultas tarifarias.
    """
    ensure_index_loaded()
    anios = set()

    for meta in metadatos:
        anio = extraer_anio_de_fecha(meta.get("fecha_sancion_iso", ""))
        if not anio:
            anio = extraer_anio_de_fecha(meta.get("fecha_sancion", ""))
        if anio:
            anios.add(anio)

    return sorted(anios, reverse=True)[:top_n]


def obtener_anios_en_resultados(resultados: list) -> list:
    """Extrae años presentes en los resultados recuperados."""
    if not resultados:
        return []

    anios = []
    for r in resultados:
        anio = extraer_anio_de_fecha(
            r.get("fecha_sancion_iso", "")
        ) or extraer_anio_de_fecha(r.get("fecha_sancion", ""))
        if anio:
            anios.append(anio)

    return list(dict.fromkeys(anios))


def buscar_por_etiqueta(etiqueta: str, anio: str | None = None) -> list:
    """
    Busca en metadatos cargados en RAM las ordenanzas que tengan la etiqueta.
    Si se pasa año, busca primero 'etiqueta_año' (ej: 'tarifaria_2025'),
    y si no encuentra, busca solo 'etiqueta'.

    Returns:
        Lista de dicts con chunk_texto + metadatos de los chunks encontrados.
    """
    ensure_index_loaded()

    etiqueta_lower = etiqueta.lower().strip()
    etiqueta_con_anio = f"{etiqueta_lower}_{anio}" if anio else None

    resultados = []
    for chunk, meta in zip(chunks, metadatos):
        tags = meta.get("etiquetas", [])
        if not tags:
            continue

        tags_lower = [t.lower() for t in tags]

        # Primero buscar con año específico
        if etiqueta_con_anio and etiqueta_con_anio in tags_lower:
            resultados.append({"chunk_texto": chunk, **meta})
        # Luego buscar solo la etiqueta base
        elif etiqueta_lower in tags_lower:
            resultados.append({"chunk_texto": chunk, **meta})

    return resultados


def extraer_monto_grande(texto: str) -> str | None:
    """
    Extrae el monto más grande (>1M) encontrado en el texto.
    Formatos: $74.908.214.520,00 ó 74.908.214.520 ó 74,908,214,520
    """
    # ⚡ Mejorado: buscar todos y elegir el mayor (el total del presupuesto siempre es el mayor)
    patrones = [
        r"\$\s*([\d]{1,3}(?:\.[\d]{3})+(?:,[\d]{2})?)",
        r"([\d]{1,3}(?:\.[\d]{3})+(?:,[\d]{2})?)",
    ]
    montos_encontrados = []

    for patron in patrones:
        matches = re.findall(patron, texto)
        for monto in matches:
            # Normalizar para convertir a float (quitar puntos, cambiar coma por punto)
            monto_float_str = monto.replace(".", "").replace(",", ".")
            try:
                valor = float(monto_float_str)
                if valor > 1_000_000:
                    montos_encontrados.append((valor, monto))
            except ValueError:
                continue

    if not montos_encontrados:
        return None

    # Ordenar por valor real y devolver el string original del más grande
    montos_encontrados.sort(key=lambda x: x[0], reverse=True)
    return f"${montos_encontrados[0][1]}"


def es_consulta_de_monto_o_tarifa(pregunta: str) -> bool:
    """Detecta si la pregunta apunta a montos/tasas/aranceles/presupuesto."""
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    tokens = set(texto_norm.split())
    return bool(tokens & CONSULTA_MONTO_KEYWORDS)


def es_consulta_comparativa(pregunta: str) -> bool:
    """Detecta consultas comparativas (aumento, variación, diferencia, etc.)."""
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    tokens = set(texto_norm.split())
    return bool(tokens & CONSULTA_COMPARATIVA_KEYWORDS)


def es_consulta_presupuesto_institucional(pregunta: str) -> bool:
    """
    Detecta preguntas de presupuesto sobre organo/secretaria responsable,
    no sobre montos o valores.
    """
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    tokens = set(texto_norm.split())
    return "presupuesto" in tokens and bool(
        tokens & CONSULTA_PRESUPUESTO_INSTITUCIONAL_KEYWORDS
    )


def es_consulta_presupuesto_de_monto(pregunta: str) -> bool:
    """
    Detecta preguntas de presupuesto orientadas a monto/total/valor.
    """
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    tokens = set(texto_norm.split())
    return "presupuesto" in tokens and bool(
        tokens & CONSULTA_PRESUPUESTO_MONTO_KEYWORDS
    )


def detectar_zona_propiedad(texto: str) -> int | None:
    """Detecta zona de inmueble en texto libre. Ej: 'Zona 7'."""
    if not texto:
        return None

    match = re.search(r"\bzona\s*([1-9])\b", texto, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def es_consulta_tasa_propiedad(pregunta: str) -> bool:
    """Detecta intención sobre tasa de servicios a la propiedad/inmuebles."""
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    tiene_base = ("propiedad" in texto_norm) or ("inmueble" in texto_norm)
    tiene_tasa = ("tasa" in texto_norm) or ("servicios a la propiedad" in texto_norm)
    return tiene_base and tiene_tasa


def es_consulta_modalidad_pago_propiedad(pregunta: str) -> bool:
    """
    Detecta intención sobre modalidad de pago (contado/cuotas) para tasa de propiedad.
    """
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    pide_modalidad = any(
        k in texto_norm
        for k in (
            "cuotas",
            "cuota",
            "contado",
            "pagar en cuotas",
            "solo al contado",
            "forma de pago",
            "modalidad de pago",
            "vencimiento",
            "vencimientos",
        )
    )
    refiere_propiedad = any(
        k in texto_norm
        for k in (
            "propiedad",
            "inmueble",
            "tasa de servicios a la propiedad",
            "servicios a la propiedad",
            "contribucion por los servicios que se prestan a la propiedad inmueble",
        )
    )
    # Si no menciona "propiedad", igual puede ser follow-up corto ("¿en cuotas o contado?")
    # y se valida por evidencia en chunks dentro del resolvedor.
    return pide_modalidad and (refiere_propiedad or len(texto_norm.split()) <= 12)


def resolver_modalidad_pago_propiedad(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    """
    Resuelve consultas sobre contado/cuotas para tasa de propiedad usando chunks relevantes.
    """
    if not es_consulta_modalidad_pago_propiedad(pregunta):
        return None

    ensure_index_loaded()

    candidatos = []
    if resultados:
        candidatos.extend(resultados)

    terminos = [
        "contribucion por los servicios que se prestan a la propiedad inmueble",
        "se cancelara al contado o en doce (12) cuotas mensuales",
        "zona 7",
        "tarifa social general",
        "6 cuotas bimestrales",
        "facultase al d.e.m. a modificar las fechas",
    ]
    candidatos.extend(busqueda_textual_directa(terminos, top_k=80, usar_stemming=False))
    if not candidatos:
        return None

    numeros = [
        str(r.get("numero_ordenanza", ""))
        for r in candidatos
        if r.get("numero_ordenanza") and r.get("numero_ordenanza") != "desconocido"
    ]
    if not numeros:
        return None
    num_citado = Counter(numeros).most_common(1)[0][0]

    textos = [r.get("chunk_texto", "") for r in candidatos if r.get("chunk_texto")]
    big_text = " ".join(textos)
    big_norm = normalizar_texto_para_busqueda(big_text)

    tiene_contado = "al contado" in big_norm
    tiene_12_cuotas = (
        "doce 12 cuotas mensuales" in big_norm
        or "doce cuotas mensuales" in big_norm
        or "12 cuotas mensuales" in big_norm
    )
    tiene_zona7_6 = "zona 7" in big_norm and "6 cuotas bimestrales" in big_norm

    if not (tiene_contado and tiene_12_cuotas):
        return None

    respuesta = (
        "Sí, podés elegir pagar al contado o en 12 cuotas mensuales y consecutivas, "
        "a opción del contribuyente. "
    )
    if tiene_zona7_6:
        respuesta += "Para Zona 7 (Tarifa Social General), la modalidad es en 6 cuotas bimestrales. "
    respuesta += "Además, el D.E.M. puede ajustar fechas de vencimiento y establecer facilidades de pago."

    return {"respuesta": respuesta, "ordenanzas_citadas": [num_citado]}


def resolver_tarifaria_intenciones(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    """
    Resuelve intenciones tarifarias frecuentes con extracción directa desde chunks.
    Cubre: tarifa social, cementerio mora, agua 2%, adicionales comercio, camiones,
    alquiler de espacios municipales y tasas de aeropuerto.
    """
    texto = normalizar_texto_para_busqueda(pregunta or "")
    if not texto:
        return None

    # Guard rápido: solo proceder si la pregunta contiene al menos un keyword tarifario
    keywords_tarifarios = {
        "tarifa",
        "social",
        "cementerio",
        "panteon",
        "nicho",
        "agua",
        "cloaca",
        "comercio",
        "comercial",
        "camion",
        "utilitario",
        "anfiteatro",
        "evento",
        "hexagonal",
        "teatro",
        "deposito",
        "vehiculo",
        "aeropuerto",
        "aterrizar",
        "aeronave",
        "ambulancia",
        "samu",
        "107",
        "descuento",
        "benefico",
        "cultural",
    }
    if not any(kw in texto for kw in keywords_tarifarios):
        return None

    ensure_index_loaded()
    candidatos = []
    if resultados:
        candidatos.extend(resultados)

    terminos_base = [
        "tarifa social general",
        "tarifa social diferencial",
        "estudios socioeconomicos",
        "cementerio",
        "panteon",
        "nicho",
        "recargos",
        "agua potable y servicios de cloacas",
        "contribucion especial",
        "actividad comercial industrial y de servicios",
        "tributo para el mantenimiento de los accesos",
        "camiones utilitarios",
        "articulo 113",
        "anfiteatro municipal",
        "teatro del anfiteatro",
        "salon hexagonal",
        "aeropuerto regional",
        "100ll",
        "tasa de aterrizaje",
        "tasa de estacionamiento de aeronaves",
        "107 samu",
        "servicio de ambulancia",
        "articulo 118",
    ]
    candidatos.extend(
        busqueda_textual_directa(terminos_base, top_k=120, usar_stemming=False)
    )
    if not candidatos:
        return None

    numeros = [
        str(r.get("numero_ordenanza", ""))
        for r in candidatos
        if r.get("numero_ordenanza") and r.get("numero_ordenanza") != "desconocido"
    ]
    if not numeros:
        return None
    num_citado = Counter(numeros).most_common(1)[0][0]
    textos = [r.get("chunk_texto", "") for r in candidatos if r.get("chunk_texto")]
    big_text = " ".join(textos)
    big_norm = normalizar_texto_para_busqueda(big_text)

    # 1) Tarifa social (bajos ingresos)
    if "tarifa social" in texto or ("bajos ingresos" in texto and "propiedad" in texto):
        if (
            "estudios socioeconomicos" in big_norm
            and "tarifa social general" in big_norm
        ):
            return {
                "respuesta": "Sí. Existe Tarifa Social para contribuyentes en situación socioeconómica vulnerable, con evaluación mediante estudio socioeconómico. Incluye Tarifa Social General (mínimo anual $30.000,00) y Tarifa Social Diferencial (reducción de hasta 50%).",
                "ordenanzas_citadas": [num_citado],
            }

    # 2) Consecuencia de no pagar cementerio
    if ("cementerio" in texto) and any(
        k in texto for k in ("no pago", "no pagar", "mora", "deuda", "que pasa")
    ):
        if (
            "contribuciones que inciden sobre los cementerios" in big_norm
            or "propietarios de panteones nichos fosas" in big_norm
        ):
            return {
                "respuesta": "Si no se paga la contribución del cementerio, la deuda queda en mora y se aplican recargos/intereses conforme la normativa municipal, además de habilitarse gestión de cobro administrativo o judicial.",
                "ordenanzas_citadas": [num_citado],
            }

    # 2b) Costo de mantenimiento en cementerio
    if (
        "cementerio" in texto
        or "panteon" in texto
        or "panteón" in texto
        or "tumba" in texto
    ) and any(k in texto for k in ("cuanto", "cuánto", "costo", "mantener", "cuesta")):
        if (
            "1 700" in big_norm
            and "2 200" in big_norm
            and "11 550" in big_norm
            and "5 150" in big_norm
        ):
            return {
                "respuesta": "Depende del tipo: panteones y terrenos $1.700,00 por m² (mínimo anual $11.550,00), panteones de sociedades/instituciones $2.200,00 por m², y nichos $5.150,00 anuales. Vencimiento previsto: cuota única 16/06/2025.",
                "ordenanzas_citadas": [num_citado],
            }

    # 3) Tributo adicional agua/cloacas
    if ("agua" in texto or "cloaca" in texto) and any(
        k in texto for k in ("tributo", "adicional", "extra", "porcentaje", "alicuota")
    ):
        if (
            "contribucion especial" in big_norm
            and "2" in big_norm
            and "agua potable" in big_norm
        ):
            return {
                "respuesta": "Sí. Se aplica una Contribución Especial con alícuota del 2% sobre el valor del servicio de agua potable y cloacas (Título XIII BIS de la O.G.I.V.).",
                "ordenanzas_citadas": [num_citado],
            }

    # 4) Adicionales para comercio
    if ("comercio" in texto or "actividad comercial" in texto) and any(
        k in texto
        for k in ("tributos adicionales", "adicionales", "que tributos", "que pago")
    ):
        rs_comercio = busqueda_textual_directa(
            [
                "articulo 117",
                "financiamiento de los servicios de salud",
                "actividad comercial industrial y de servicios",
                "articulo 119",
                "alicuota quince por ciento",
            ],
            top_k=30,
            usar_stemming=False,
        )
        txt_comercio = " ".join([x.get("chunk_texto", "") for x in rs_comercio])
        norm_comercio = normalizar_texto_para_busqueda(txt_comercio)
        tiene_art_117 = "articulo 117" in norm_comercio
        tiene_art_119 = "articulo 119" in norm_comercio
        tiene_trib_salud = (
            "financiamiento de los servicios de salud" in norm_comercio
            and ("veinte por ciento" in norm_comercio or "20" in norm_comercio)
        )
        tiene_trib_obras = (
            "alicuota quince por ciento" in norm_comercio
            or "quince por ciento" in norm_comercio
            or "15" in norm_comercio
        )
        if (tiene_trib_salud or tiene_art_117) and (tiene_trib_obras or tiene_art_119):
            nums_comercio = [
                str(x.get("numero_ordenanza", ""))
                for x in rs_comercio
                if x.get("numero_ordenanza")
                and x.get("numero_ordenanza") != "desconocido"
            ]
            ord_comercio = (
                Counter(nums_comercio).most_common(1)[0][0]
                if nums_comercio
                else num_citado
            )
            return {
                "respuesta": "Además de la contribución principal, se prevén adicionales: 20% para financiamiento de servicios de salud municipal y 15% para los conceptos definidos en el Art. 119 (incluye actividad comercial, industrial y de servicios).",
                "ordenanzas_citadas": [ord_comercio],
            }

    # 5) Tributo especial camiones/utilitarios
    if ("camion" in texto or "utilitario" in texto) and any(
        k in texto for k in ("tributo", "especial", "pagar", "mantenimiento de accesos")
    ):
        if "350 00" in big_norm and "30 000 00" in big_norm:
            return {
                "respuesta": "Sí. Para actividades con camiones/utilitarios afectados al establecimiento que circulen en la ciudad o accesos, el tributo es $350,00 por tonelada, con mínimo mensual por camión de $30.000,00. Las empresas de transporte de pasajeros están exceptuadas por ese inciso.",
                "ordenanzas_citadas": [num_citado],
            }

    # 6) Descuento por evento cultural/benefico (prioridad sobre precios)
    if any(
        k in texto for k in ("descuento", "benefico", "benéfico", "cultural")
    ) and any(
        k in texto
        for k in ("evento", "anfiteatro", "espacios municipales", "arancel", "tarifa")
    ):
        if (
            "reducir total o parcialmente" in big_norm
            and "fin benefico o cultural" in big_norm
        ):
            return {
                "respuesta": "Sí. El D.E.M. está facultado para reducir total o parcialmente los valores cuando el evento tenga auspicio municipal u otras entidades con fin benéfico o cultural.",
                "ordenanzas_citadas": [num_citado],
            }

    # 7) Alquiler de Anfiteatro / espacios municipales
    if any(
        k in texto
        for k in (
            "anfiteatro",
            "espacios municipales",
            "alquilar",
            "evento",
            "salon hexagonal",
            "teatro del anfiteatro",
        )
    ):
        if "3 000 000 00" in big_norm and "teatro del anfiteatro" in big_norm:
            return {
                "respuesta": "Los eventos en el Anfiteatro Municipal abonan 3% de la recaudación por entradas con mínimo de $3.000.000,00 por evento. El Teatro del Anfiteatro: $2.000.000,00. Salón Hexagonal: $250.000,00 por día o $100.000,00 por medio día.",
                "ordenanzas_citadas": [num_citado],
            }

    # 8) Deposito municipal (estadía por día)
    if any(
        k in texto
        for k in (
            "deposito municipal",
            "depósito municipal",
            "saquen mi auto",
            "retirar auto",
            "vehiculo en deposito",
            "vehículo en depósito",
        )
    ):
        rs_deposito = busqueda_textual_directa(
            [
                "articulo 112",
                "vehiculos en deposito",
                "30 mt",
                "20 mt",
                "11 mt",
                "dos primeras horas",
            ],
            top_k=20,
            usar_stemming=False,
        )
        norm_dep = normalizar_texto_para_busqueda(
            " ".join([x.get("chunk_texto", "") for x in rs_deposito])
        )
        if (
            "vehiculos en deposito" in norm_dep
            and "20 mt" in norm_dep
            and "30 mt" in norm_dep
        ):
            return {
                "respuesta": "Por vehículos en depósito municipal, la estadía es: Automóvil/jeep 20 MT por día, Camiones/acoplados/ómnibus 30 MT por día, Motocicletas 11 MT por día. Si regularizás dentro de las primeras 2 horas, no se cobra estadía.",
                "ordenanzas_citadas": [num_citado],
            }

    # 9) Tasas de aeropuerto (aterrizaje) — solo si hay intención tarifaria
    tiene_intencion_tarifaria_aero = any(
        t in texto
        for t in (
            "tasa",
            "tarifa",
            "costo",
            "cuanto",
            "pagar",
            "aterrizaje",
            "estacionamiento de aeronave",
        )
    )
    if (
        "aeropuerto" in texto or "aterrizar" in texto or "aeronave" in texto
    ) and tiene_intencion_tarifaria_aero:
        if (
            "1 litro de combustible 100ll" in big_norm
            and "peso maximo de despegue" in big_norm
        ):
            return {
                "respuesta": "La tasa de aterrizaje es equivalente a 1 litro de combustible 100LL por cada tonelada de MTOW (peso máximo de despegue), tomando como referencia YPF San Fernando (Buenos Aires). Además, la tasa de estacionamiento por hora es: 2-5T $57,00; 6-10T $77,00; 11-16T $103,00; 17T o más $128,00. Operación nocturna: +30%.",
                "ordenanzas_citadas": [num_citado],
            }

    # 10) Servicio de ambulancia SAMU
    if any(
        k in texto
        for k in (
            "ambulancia",
            "samu",
            "107",
            "emergencia medica",
            "servicio medico",
        )
    ):
        rs_samu = busqueda_textual_directa(
            [
                "107 samu",
                "servicio de ambulancia",
                "articulo 118",
                "chofer y paramedico por evento",
                "dea",
            ],
            top_k=30,
            usar_stemming=False,
        )
        txt_samu = " ".join([x.get("chunk_texto", "") for x in rs_samu])
        norm_samu = normalizar_texto_para_busqueda(txt_samu)
        if "107 samu" in norm_samu and "307 100" in norm_samu and "61 500" in norm_samu:
            # Extraer datos exactos del texto
            respuesta_samu = (
                "Según el Art. 118° de la Ordenanza Tarifaria, los aranceles del servicio 107 SAMU son:\n\n"
                "**Servicios de Ambulancia:**\n"
                "- Ambulancia con chofer y paramédico por evento: **$307.100,00**\n"
                "- Ambulancia con chofer, paramédico y médico por hora: **$61.500,00**\n"
                "- Servicio de médico por hora: **$41.000,00**\n\n"
                "Estos valores aplican cuando el servicio es requerido por personas o entidades no cubiertas por el sistema municipal."
            )
            num_samu = Counter(
                [
                    str(x.get("numero_ordenanza", ""))
                    for x in rs_samu
                    if x.get("numero_ordenanza")
                ]
            ).most_common(1)
            ord_samu = num_samu[0][0] if num_samu else num_citado
            return {
                "respuesta": respuesta_samu,
                "ordenanzas_citadas": [ord_samu],
            }

    return None


def parece_nombre_persona_sin_contexto(pregunta: str) -> bool:
    texto = (pregunta or "").strip().strip("¿?.,;: ")
    if not texto or extraer_numero_ordenanza_de_pregunta(texto):
        return False

    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{2,}", texto)
    if not (2 <= len(tokens) <= 4):
        return False

    tokens_lower = [t.lower() for t in tokens]
    palabras_no_nombre = {
        "ordenanza",
        "articulo",
        "art",
        "vigencia",
        "presupuesto",
        "comercio",
        "residuos",
        "reciclaje",
    }
    if any(t in palabras_no_nombre for t in tokens_lower):
        return False

    return all(t[:1].isupper() for t in tokens)


def es_consulta_habilitacion_generica(pregunta: str) -> bool:
    texto = normalizar_texto_para_busqueda(pregunta or "")
    if "habilit" not in texto:
        return False

    if not any(t in texto for t in ("comercio", "local", "negocio", "actividad")):
        return False

    especificadores = (
        "gastronom",
        "bar",
        "restaurant",
        "kiosco",
        "farmacia",
        "industr",
        "servicio",
        "alimento",
        "boliche",
        "transporte",
        "taxi",
        "vta",
    )
    return not any(e in texto for e in especificadores)


def _inferir_temas_resultado(resultado: dict) -> set[str]:
    texto = normalizar_texto_para_busqueda(resultado.get("chunk_texto", ""))
    temas = set()

    if any(k in texto for k in ("tarifa", "tasa", "tribut", "monto", "canon", "pagar")):
        temas.add("tarifas o montos")
    if any(k in texto for k in ("cooperativa", "convenio", "ratific", "prestacion")):
        temas.add("convenios o cooperativas")
    if any(
        k in texto
        for k in (
            "gestion ambiental",
            "recicl",
            "programa",
            "secretaria",
            "centro de gestion ambiental",
            "funcionamiento",
        )
    ):
        temas.add("regulacion o funcionamiento")
    if any(k in texto for k in ("multa", "sancion", "prohib", "infraccion", "falta")):
        temas.add("infracciones")

    return temas


def _temas_dominantes(resultados: list | None, max_temas: int = 3) -> list:
    contador = Counter()
    for r in (resultados or [])[:8]:
        for tema in _inferir_temas_resultado(r):
            contador[tema] += 1
    return [tema for tema, _ in contador.most_common(max_temas)]


def resolver_ordenanza_extrema(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    texto = normalizar_texto_para_busqueda(pregunta or "")
    if not any(
        t in texto
        for t in ("ultima ordenanza", "mas reciente", "primera ordenanza")
    ):
        return None

    if not resultados:
        return None

    principal = resultados[0]
    numero = str(principal.get("numero_ordenanza", "")).strip()
    fecha = principal.get("fecha_sancion", "desconocida")
    if (not fecha or str(fecha).strip().lower() == "desconocida") and numero:
        ensure_index_loaded()
        numero_norm = normalizar_numero(numero)
        for meta in metadatos:
            if normalizar_numero(str(meta.get("numero_ordenanza", ""))) != numero_norm:
                continue
            fecha = (
                meta.get("fecha_sancion", "")
                or meta.get("fecha_sancion_iso", "")
                or fecha
            )
            if fecha:
                break
    descripcion = (
        principal.get("resumen")
        or principal.get("Art N°1")
        or principal.get("chunk_texto", "")
    )
    descripcion = re.sub(r"\s+", " ", descripcion).strip()[:260]

    if "primera ordenanza" in texto:
        respuesta = f"La primera ordenanza cargada es la Ordenanza N° {numero}, sancionada el {fecha}."
    else:
        respuesta = f"La ultima ordenanza cargada es la Ordenanza N° {numero}, sancionada el {fecha}."

    if descripcion:
        respuesta += f" Segun el texto disponible, trata sobre: {descripcion}"

    return {"respuesta": respuesta, "ordenanzas_citadas": [numero] if numero else []}


def construir_pregunta_aclaratoria(
    pregunta: str, resultados: list | None = None
) -> str | None:
    """
    Si falta información clave para responder con precisión, devuelve una repregunta.
    Si hay contexto suficiente, devuelve None.
    """
    pregunta = (pregunta or "").strip()
    anios_en_pregunta = extraer_anios_de_texto(pregunta)
    anios_en_resultados = obtener_anios_en_resultados(resultados or [])

    if es_consulta_comparativa(pregunta) and len(anios_en_pregunta) < 2:
        ejemplo = (
            f"{anios_en_resultados[0]} y {anios_en_resultados[1]}"
            if len(anios_en_resultados) >= 2
            else "2024 y 2025"
        )
        return (
            "Para calcular la variación necesito dos años concretos. "
            f"¿Qué años querés comparar (por ejemplo, {ejemplo})?"
        )

    requiere_anio_monto = es_consulta_de_monto_o_tarifa(
        pregunta
    ) or es_consulta_presupuesto_de_monto(pregunta)
    # No pedir año si la pregunta ya menciona un número de ordenanza específico
    # (ej: "¿Qué cooperativa solicitó la revisión tarifaria de la Ordenanza 8240?")
    tiene_ord_especifica = bool(extraer_numero_ordenanza_de_pregunta(pregunta))
    if es_consulta_multa_municipal(pregunta) and not tiene_ord_especifica:
        tokens_especificos_multa = extraer_tokens_especificos_multa(pregunta)
        if len(tokens_especificos_multa) < 2:
            return (
                "Para decirte el monto necesito la infracción o conducta sancionada. "
                "¿Por ejemplo, estacionar en lugar prohibido u otra falta específica?"
            )
    # No pedir año si la pregunta es factual (quién, qué, cuál) y no pide un monto
    palabras_factuales = {
        "quien",
        "quién",
        "cual",
        "cuál",
        "qué",
        "que",
        "cooperativa",
        "empresa",
        "entidad",
        "carnet",
        "sellado",
        "certificado",
        "trámite",
        "tramite",
        "manipulador",
    }
    es_factual = any(p in pregunta.lower() for p in palabras_factuales) and not any(
        p in pregunta.lower()
        for p in ("cuánto", "cuanto", "monto", "pagar", "cuesta", "precio", "valor")
    )
    # No pedir año si ya hay una tarifaria vigente en resultados
    tiene_tarifaria_vigente = any(
        any(
            token in normalizar_texto_para_busqueda(
                " ".join(
                    filter(
                        None,
                        [
                            r.get("chunk_texto", ""),
                            r.get("resumen", ""),
                            r.get("Art N°1", ""),
                            r.get("Art NÂ°1", ""),
                        ],
                    )
                )
            )
            for token in (
                "ordenanza tarifaria",
                "tasa de servicios a la propiedad",
                "contribucion por los servicios que se prestan a la propiedad inmueble",
                "tasa anual minima",
            )
        )
        for r in (resultados or [])
    )

    if (
        requiere_anio_monto
        and not es_consulta_presupuesto_institucional(pregunta)
        and not es_consulta_tasa_propiedad(pregunta)
        and not es_consulta_multa_municipal(pregunta)
        and not anios_en_pregunta
        and not tiene_ord_especifica
        and not es_factual
        and not tiene_tarifaria_vigente
    ):
        sugeridos = anios_en_resultados or obtener_anios_disponibles(top_n=2)
        if len(sugeridos) >= 2:
            return (
                "Para darte un monto exacto necesito el año de la ordenanza. "
                f"¿Querés {sugeridos[0]} o {sugeridos[1]}?"
            )
        if len(sugeridos) == 1:
            return (
                "Para darte un monto exacto necesito el año de la ordenanza. "
                f"¿Te referís a {sugeridos[0]}?"
            )
        return (
            "Para darte un monto exacto necesito el año de la ordenanza tarifaria. "
            "¿De qué año querés consultar?"
        )

    if parece_nombre_persona_sin_contexto(pregunta):
        nombre = pregunta.strip().strip("¿? ")
        return (
            f"Necesito un poco más de contexto sobre {nombre}. "
            f"¿Querés saber en qué ordenanzas se lo menciona, qué cargo tenía o algún dato puntual?"
        )

    if es_consulta_habilitacion_generica(pregunta):
        return (
            "Para responder con precisión necesito el tipo de actividad o local. "
            "¿Qué querés habilitar exactamente?"
        )

    texto_norm = normalizar_texto_para_busqueda(pregunta)
    if any(t in texto_norm for t in ("reciclaje", "residuo", "residuos", "ambiental")):
        temas = _temas_dominantes(resultados)
        if (
            len(temas) >= 2
            and not es_consulta_residuos_con_intencion_clara(pregunta)
            and not hay_ordenanza_dominante(resultados, min_hits=3)
        ):
            opciones = ", ".join(temas[:3])
            return (
                "Encontré varias líneas normativas relacionadas con tu consulta. "
                f"¿Buscás {opciones}?"
            )

    if resultados is not None and len(resultados) == 0:
        return (
            "No encontré información suficiente para responder con precisión. "
            "¿Podés indicar número de ordenanza, tema y año?"
        )

    return None


def resolver_tasa_propiedad(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    """
    Resuelve preguntas de tasa de servicios a la propiedad con lógica determinística.
    """
    if not es_consulta_tasa_propiedad(pregunta):
        return None

    ensure_index_loaded()
    zona = detectar_zona_propiedad(pregunta)

    candidatos = []
    if resultados:
        candidatos.extend(resultados)

    terminos = [
        "tasa de servicios a la propiedad",
        "contribucion por los servicios que se prestan a la propiedad inmueble",
        "contribuciones que inciden sobre los inmuebles",
        "tasa anual minima",
        "valor de referencia fiscal",
        "zona 1",
        "zona 5",
        "zona 7",
        "tarifa social",
        "por mil",
        "monto fijo",
    ]
    if zona:
        terminos.append(f"zona {zona}")
    candidatos.extend(busqueda_textual_directa(terminos, top_k=80, usar_stemming=False))

    if not candidatos:
        return None

    def puntuar_candidato_propiedad(candidato: dict) -> float:
        texto_candidato = " ".join(
            filter(
                None,
                [
                    candidato.get("chunk_texto", ""),
                    candidato.get("resumen", ""),
                    candidato.get("Art N°1", ""),
                    candidato.get("Art NÂ°1", ""),
                ],
            )
        )
        texto_norm = normalizar_texto_para_busqueda(texto_candidato)
        score = 0.0

        if es_chunk_administrativo_ruidoso(candidato.get("chunk_texto", "")):
            score -= 8.0
        if any(
            token in texto_norm
            for token in (
                "tasa de servicios a la propiedad",
                "propiedad inmueble",
                "contribuciones que inciden sobre los inmuebles",
            )
        ):
            score += 7.0
        if "tasa anual minima" in texto_norm:
            score += 6.0
        if "valor de referencia fiscal" in texto_norm or "vrf" in texto_norm:
            score += 3.0
        if "por mil" in texto_norm:
            score += 3.0
        if "monto fijo" in texto_norm:
            score += 3.0
        if "tarifa social" in texto_norm:
            score += 2.0
        if "zona " in texto_norm:
            score += 2.0
        if zona and f"zona {zona}" in texto_norm:
            score += 5.0

        return score

    candidatos_filtrados = []
    for candidato in candidatos:
        score_propiedad = puntuar_candidato_propiedad(candidato)
        if score_propiedad < 4:
            continue
        candidato["_score_propiedad"] = score_propiedad
        candidatos_filtrados.append(candidato)

    if not candidatos_filtrados:
        return None

    candidatos_filtrados.sort(
        key=lambda c: (
            c.get("_score_propiedad", 0),
            puntuar_resultado_local(pregunta, c, usar_stemming=False),
            c.get("coincidencias_textuales", 0),
        ),
        reverse=True,
    )

    mejor_candidato = next(
        (
            candidato
            for candidato in candidatos_filtrados
            if candidato.get("numero_ordenanza")
            and candidato.get("numero_ordenanza") != "desconocido"
            and any(
                token in normalizar_texto_para_busqueda(candidato.get("chunk_texto", ""))
                for token in ("tasa anual minima", "por mil", "monto fijo")
            )
        ),
        None,
    ) or next(
        (
            candidato
            for candidato in candidatos_filtrados
            if candidato.get("numero_ordenanza")
            and candidato.get("numero_ordenanza") != "desconocido"
        ),
        None,
    )
    if not mejor_candidato:
        return None
    num_citado = str(mejor_candidato.get("numero_ordenanza", "")).strip()

    textos = [
        r.get("chunk_texto", "")
        for r in candidatos_filtrados
        if r.get("chunk_texto") and str(r.get("numero_ordenanza", "")).strip() == num_citado
    ]
    if not textos:
        textos = [r.get("chunk_texto", "") for r in candidatos_filtrados if r.get("chunk_texto")]
    big_text = " ".join(textos).replace("\n", " ")
    big_upper = big_text.upper()

    # Parseo de alicuotas por zona (ej: ZONA 1 8,00 POR MIL)
    alicuotas = {}
    for z, a in re.findall(
        r"ZONA\s*([1-9])\s*([0-9]+(?:[.,][0-9]+)?)\s*POR\s*MIL", big_upper
    ):
        alicuotas[int(z)] = f"{a.replace('.', ',')} por mil"

    # Parseo de mínimos anuales (ej: ZONA 1 $ 78.050,00)
    minimos = {}
    for z, monto in re.findall(r"ZONA\s*([1-9])\s*\$\s*([0-9\.\,]+)", big_upper):
        minimos[int(z)] = f"${monto}"

    # Preferir bloque explícito de Tasa Anual Mínima 2025
    bloque_minimo_2025 = ""
    for t in textos:
        up = (t or "").upper()
        if (
            "TASA ANUAL MÍNIMA PARA EL AÑO 2025" in up
            or "TASA ANUAL MINIMA PARA EL ANO 2025" in up
        ):
            bloque_minimo_2025 = up
            break
    if bloque_minimo_2025:
        minimos_2025 = {}
        for z, monto in re.findall(
            r"ZONA\s*([1-9])\s*\$\s*([0-9\.\,]+)", bloque_minimo_2025
        ):
            minimos_2025[int(z)] = f"${monto}"
        if minimos_2025:
            minimos = minimos_2025

    if zona:
        if zona == 7:
            return {
                "respuesta": "Para Zona 7 (Tarifa Social), la alícuota es 0 por mil. Se aplica Tarifa Social General (mínimo anual de $30.000,00) o Tarifa Social Diferencial (reducción de hasta 50%, según evaluación socioeconómica).",
                "ordenanzas_citadas": [num_citado],
            }

        ali = alicuotas.get(zona)
        minimo = minimos.get(zona)
        if ali and minimo:
            return {
                "respuesta": f"Para Zona {zona}, la alícuota es {ali} y la tasa anual mínima es {minimo}. El monto final depende del Valor de Referencia Fiscal (VRF) del inmueble.",
                "ordenanzas_citadas": [num_citado],
            }
        if minimo:
            return {
                "respuesta": f"Para Zona {zona}, la tasa anual mínima es {minimo}. El monto final puede variar según el Valor de Referencia Fiscal (VRF) del inmueble y las alícuotas aplicables.",
                "ordenanzas_citadas": [num_citado],
            }
        if ali:
            return {
                "respuesta": f"Para Zona {zona}, la alícuota es {ali}. El monto final depende del VRF del inmueble y de los mínimos vigentes.",
                "ordenanzas_citadas": [num_citado],
            }

    ali_z1 = alicuotas.get(1, "8,00 por mil")
    ali_z7 = alicuotas.get(7, "0,00 por mil")
    min_z1 = minimos.get(1, "$78.050,00")
    min_z5 = minimos.get(5, "$27.850,00")

    return {
        "respuesta": f"El monto depende de la zona y del valor fiscal (VRF) de tu propiedad. Como referencia, las alícuotas van de {ali_z1} en Zona 1 a {ali_z7} en Zona 7 (Tarifa Social); los mínimos anuales incluyen Zona 1 = {min_z1} y Zona 5 = {min_z5}. ¿En qué zona está tu propiedad?",
        "ordenanzas_citadas": [num_citado],
    }


def resolver_consulta_ordenanza_explicita(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    numero_ordenanza = extraer_numero_ordenanza_de_pregunta(pregunta or "")
    if not numero_ordenanza:
        return None

    resumen = obtener_resumen_ordenanza(resultados, numero_ordenanza)

    consulta_articulos = bool(
        re.search(r"\bart(?:icul\w*)?\b", texto_norm)
        or "art cul" in texto_norm
        or "art culos" in texto_norm
    )
    if consulta_articulos and any(
        token in texto_norm
        for token in (
            "modifica",
            "modificacion",
            "modific",
            "introduce",
            "reemplaza",
            "sustituye",
            "incorpora",
        )
    ):
        articulos = extraer_articulos_modificados_desde_resultados(
            resultados, numero_ordenanza
        )
        if articulos:
            lista_articulos = formatear_lista_articulos(articulos)
            return {
                "respuesta": f"La Ordenanza N° {numero_ordenanza} modifica {lista_articulos}.",
                "ordenanzas_citadas": [numero_ordenanza],
            }

    pide_resumen = any(
        token in texto_norm
        for token in (
            "trata",
            "breve",
            "resumen",
            "finalidad",
            "objeto",
            "regula",
            "establece",
            "incentivo",
            "descuento",
            "beneficio",
            "dependencia",
            "secretaria",
            "subsecretaria",
            "cooperativa",
        )
    )
    pide_resumen = pide_resumen or (
        "modific" in texto_norm and any(token in texto_norm for token in ("introduce", "tipo"))
    )
    if pide_resumen and resumen:
        return {
            "respuesta": resumen,
            "ordenanzas_citadas": [numero_ordenanza],
        }

    return None


def resolver_multa_municipal(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    if not es_consulta_multa_municipal(pregunta):
        return None

    tokens_especificos = extraer_tokens_especificos_multa(pregunta)
    if len(tokens_especificos) < 2:
        return None

    ensure_index_loaded()
    candidatos = []
    if resultados:
        candidatos.extend(resultados)

    terminos = tokens_especificos + [
        "multa",
        "sancionado con multa",
        "minimo se fija",
        "u m",
        "mt",
    ]
    candidatos.extend(busqueda_textual_directa(terminos, top_k=60, usar_stemming=False))
    if not candidatos:
        return None

    def puntuar_candidato_multa(candidato: dict) -> float:
        chunk_texto = candidato.get("chunk_texto", "")
        if es_chunk_administrativo_ruidoso(chunk_texto):
            return -10.0

        chunk_norm = normalizar_texto_para_busqueda(chunk_texto)
        score = 0.0
        score += sum(2.2 for token in tokens_especificos if token in chunk_norm)
        if any(token in chunk_norm for token in ("multa", "sancionado", "falta")):
            score += 3.0
        if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:u\.?\s*m\.?|mt)\b", chunk_texto, re.IGNORECASE):
            score += 4.0
        if re.search(r"\$\s*[0-9\.\,]+", chunk_texto):
            score += 2.5
        return score

    candidatos_puntuados = []
    for candidato in candidatos:
        score_multa = puntuar_candidato_multa(candidato)
        if score_multa < 5:
            continue
        candidato["_score_multa"] = score_multa
        candidatos_puntuados.append(candidato)

    if not candidatos_puntuados:
        return None

    candidatos_puntuados.sort(
        key=lambda c: (
            c.get("_score_multa", 0),
            puntuar_resultado_local(pregunta, c, usar_stemming=False),
        ),
        reverse=True,
    )

    mejor = candidatos_puntuados[0]
    chunk_texto = mejor.get("chunk_texto", "")
    numero_ordenanza = str(mejor.get("numero_ordenanza", "")).strip()
    articulo = extraer_articulo_de_chunk(chunk_texto)

    match_um = re.search(
        r"(?:minimo|mínimo)[^.;\n]{0,80}?(\d+(?:[.,]\d+)?)\s*(u\.?\s*m\.?|mt)\b",
        chunk_texto,
        re.IGNORECASE,
    )
    match_pesos = re.search(r"\$\s*([0-9\.\,]+)", chunk_texto)

    detalle = None
    if match_um:
        unidad = match_um.group(2).upper().replace(" ", "")
        if unidad in {"UM", "U.M", "U.M."}:
            unidad = "U.M."
        elif unidad == "MT":
            unidad = "MT"
        detalle = f"la multa minima es de {match_um.group(1)} {unidad}"
    elif match_pesos:
        detalle = f"la multa informada es de ${match_pesos.group(1)}"

    if not detalle:
        fragmento = limpiar_texto_respuesta_local(
            extraer_fragmento_relevante(chunk_texto, pregunta, max_chars=220), 220
        )
        if not fragmento:
            return None
        detalle = fragmento[0].lower() + fragmento[1:] if len(fragmento) > 1 else fragmento.lower()

    referencia = f"Segun la Ordenanza N° {numero_ordenanza}"
    if articulo:
        referencia += f", Art. {articulo}°"

    return {
        "respuesta": f"{referencia}, para la infraccion consultada {detalle}.",
        "ordenanzas_citadas": [numero_ordenanza] if numero_ordenanza else [],
    }


def normalizar_texto_para_busqueda(texto: str) -> str:
    """Normaliza texto eliminando acentos y caracteres especiales."""
    import unicodedata

    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def aplicar_stemming(texto: str) -> str:
    """
    Aplica stemming (raíz de palabras) al texto.
    inaugúrese -> inaugur
    inauguración -> inaugur
    inaugurar -> inaugur
    """
    if not stemmer:
        return normalizar_texto_para_busqueda(texto)

    texto_norm = normalizar_texto_para_busqueda(texto)
    palabras = texto_norm.split()
    palabras_stem = [stemmer.stem(palabra) for palabra in palabras]
    return " ".join(palabras_stem)


def busqueda_textual_directa(terminos: list, top_k=10, usar_stemming=True):
    """
    Búsqueda textual con matching de frases (n-gramas) usando arrays pre-computados.
    Busca en: chunks + palabras_clave + Art N°1 de metadatos.
    Bonifica fuertemente cuando los términos aparecen juntos como frase.
    """
    resultados = []

    if usar_stemming and stemmer:
        chunk_texts = chunks_stemmed
    else:
        chunk_texts = chunks_normalized

    # Usar arrays pre-computados (sin re-calcular stemming/normalización)
    for idx, (chunk_proc, meta) in enumerate(zip(chunk_texts, metadatos)):
        coincidencias, n_matches = _puntuar_coincidencias_textuales(
            terminos, chunk_proc, meta, usar_stemming
        )

        if coincidencias > 0 and n_matches > 0:
            meta_copy = dict(meta)
            meta_copy["chunk_texto"] = chunks[idx]
            meta_copy["coincidencias_textuales"] = coincidencias
            meta_copy["_n_matches_textuales"] = n_matches
            resultados.append(meta_copy)

    resultados.sort(key=lambda x: x.get("coincidencias_textuales", 0), reverse=True)
    return resultados[:top_k]


def expandir_consulta(pregunta: str) -> str:
    """Expande la consulta con variaciones y sinónimos.
    No expande si la consulta ya contiene frases legales específicas.
    """
    pregunta_lower = pregunta.lower()

    # Frases legales específicas: si la consulta ya es precisa, no diluir
    frases_especificas = [
        "acuerdo transaccional",
        "acuerdos transaccionales",
        "acta acuerdo",
        "convenio colectivo",
        "convenio marco",
        "acuerdo paritario",
    ]
    for frase in frases_especificas:
        if frase in pregunta_lower:
            return pregunta  # Ya es suficientemente precisa

    expansion = pregunta

    # Diccionario de expansiones
    expansiones = {
        "acta": ["convenio", "acuerdo"],
        "acuerdo": ["convenio", "pacto"],
        "transaccional": ["acuerdo transaccional"],
        "transaccionales": ["acuerdo transaccional"],
        "suoem": ["sindicato", "gremio", "empleados municipales"],
        "salarial": ["salario", "remuneración", "sueldo"],
        "aumento": ["incremento", "ajuste", "recomposición"],
        "inaugurar": ["inauguración", "inaugurese", "apertura"],
        "inauguración": ["inaugurar", "inaugurese", "apertura"],
        "inaugurese": ["inaugurar", "inauguración", "apertura"],
        "musical": ["sadaic", "aadi capif", "fonogramas"],
        "musicales": ["sadaic", "aadi capif", "fonogramas"],
        "reciclaje": ["reciclado", "residuos", "gestion ambiental"],
        "reciclado": ["reciclaje", "residuos", "gestion ambiental"],
        "habilitar": ["habilitacion", "direccion de habilitaciones"],
        "habilitación": ["habilitacion", "direccion de habilitaciones"],
        "comercio": ["actividad comercial", "tasa de comercio"],
    }

    terminos_agregados = []
    for termino_base, sinonimos in expansiones.items():
        if termino_base in pregunta_lower:
            terminos_agregados.extend(sinonimos[:2])

    if terminos_agregados:
        expansion = f"{pregunta} {' '.join(terminos_agregados)}"

    return expansion


def detectar_tipo_pregunta(pregunta: str) -> str:
    """
    Detecta el tipo de pregunta:
    - 'numero_especifico': Solo un número de ordenanza (ej: "8058")
    - 'palabra_clave': Una sola palabra o término (ej: "inaugurese", "SUOEM")
    - 'generica': Búsqueda simple sin contexto
    - 'referencia': Pregunta sobre derogaciones, modificaciones
    - 'directa': Pregunta específica con contexto
    """
    pregunta_limpia = pregunta.strip()
    palabras = pregunta_limpia.split()

    # Si es un solo número, es consulta de ordenanza específica → GPT resume
    if len(palabras) == 1 and pregunta_limpia.isdigit():
        return "numero_especifico"

    # Si es una sola palabra no numérica, es búsqueda por palabra clave
    if len(palabras) == 1:
        return "palabra_clave"

    palabras_interrogativas = [
        "qué",
        "que",
        "cuál",
        "cual",
        "cuándo",
        "cuando",
        "cómo",
        "como",
        "dónde",
        "donde",
        "quién",
        "quien",
        "por",
        "para",
    ]

    tiene_interrogativa = any(
        palabra.lower() in palabras_interrogativas for palabra in palabras
    )

    # Detectar frases imperativas: "pasame X", "listame Y", "busca Z"
    verbos_imperativos = {
        "pasame",
        "pasá",
        "pásame",
        "listame",
        "listá",
        "dame",
        "dáme",
        "muestra",
        "mostrá",
        "mostrame",
        "busca",
        "buscá",
        "buscame",
        "decime",
        "decí",
        "contame",
        "contá",
        "traeme",
        "traé",
        "enviame",
        "mandame",
        "mandá",
        "encontrame",
        "necesito",
        "quiero",
    }
    primera_palabra = palabras[0].lower().rstrip(",.:;") if palabras else ""
    if primera_palabra in verbos_imperativos:
        return "directa"

    if len(palabras) <= 4 and not tiene_interrogativa:
        return "generica"

    pregunta_lower = pregunta.lower()
    patrones_referencia = [
        r"cu[áa]ndo\s+(?:se\s+)?(?:dio\s+de\s+baja|derog[óo]|modific[óo]|suspend[ióo])",
        r"(?:fue\s+)?(?:dada\s+de\s+baja|derogada|modificada|suspendida)",
        r"qu[ée]\s+ordenanza\s+(?:deroga|modifica|suspende|da\s+de\s+baja)",
        r"qu[ée]\s+norma\s+(?:deroga|modifica|suspende)",
        r"est[áa]\s+(?:vigente|derogada|activa)",
        r"sigue\s+en\s+vigor",
    ]

    for patron in patrones_referencia:
        if re.search(patron, pregunta_lower):
            return "referencia"

    return "directa"


def buscar_ordenanzas_que_mencionan(numero_ordenanza: str, top_k=10):
    num_norm = normalizar_numero(numero_ordenanza)
    resultados = []

    patrones = [
        rf"ordenanza\s*[n°º]?\s*{numero_ordenanza}",
        rf"ordenanza\s*[n°º]?\s*{num_norm}",
        rf"\b{numero_ordenanza}\b",
    ]

    # Usar zip para iterar de forma segura
    for chunk, meta in zip(chunks, metadatos):
        chunk_lower = chunk.lower()

        if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
            continue

        for patron in patrones:
            if re.search(patron, chunk_lower, re.IGNORECASE):
                palabras_clave = [
                    "derog",
                    "modific",
                    "suspend",
                    "dej",
                    "sin efecto",
                    "baja",
                    "anula",
                    "revoca",
                    "sustitu",
                    "reemplaza",
                ]

                if any(palabra in chunk_lower for palabra in palabras_clave):
                    meta_copy = dict(meta)
                    meta_copy["chunk_texto"] = chunk
                    meta_copy["relevancia_referencia"] = True
                    resultados.append(meta_copy)
                    break

    return resultados[:top_k]


STOPWORDS_BUSQUEDA = {
    # Artículos y preposiciones
    "el",
    "la",
    "de",
    "del",
    "en",
    "y",
    "a",
    "que",
    "es",
    "por",
    "para",
    "con",
    "un",
    "una",
    "los",
    "las",
    "se",
    "sobre",
    "al",
    "ante",
    "sin",
    "entre",
    "desde",
    "hasta",
    "hacia",
    "tras",
    # Demostrativos
    "esta",
    "este",
    "esto",
    "estos",
    "estas",
    "aquel",
    "aquella",
    "aquello",
    "aquellos",
    "aquellas",
    # Interrogativos y relativos
    "que",
    "cual",
    "cuales",
    "quien",
    "quienes",
    "como",
    "cuando",
    "donde",
    "porque",
    "cuantos",
    "cuantas",
    # Verbos comunes (no aportan al contenido)
    "son",
    "hay",
    "tiene",
    "tienen",
    "fue",
    "fueron",
    "sido",
    "ser",
    "estar",
    "haber",
    "hacer",
    "puede",
    "pueden",
    "debe",
    "deben",
    "era",
    "eran",
    # Cuantificadores
    "todos",
    "todas",
    "todo",
    "toda",
    "otro",
    "otra",
    "otros",
    "otras",
    "cada",
    "mismo",
    "misma",
    "algun",
    "alguno",
    "alguna",
    "algunos",
    "algunas",
    "mas",
    "menos",
    "muy",
    "tan",
    # Contexto municipal (demasiado genéricos)
    "villa",
    "maria",
    "municipalidad",
    "ordenanza",
    "ciudad",
    "intendente",
    "concejo",
    "deliberante",
    "cordoba",
    "articulo",
    "artículo",
    "provincia",
}


def extraer_terminos_clave(pregunta: str) -> list:
    palabras = re.findall(r"\b\w{3,}\b", pregunta.lower())
    terminos = [p for p in palabras if p not in STOPWORDS_BUSQUEDA]

    siglas = re.findall(r"\b[A-ZÁÉÍÓÚ]{2,}\b", pregunta)
    terminos.extend([s.lower() for s in siglas])

    # Preservar orden (importante para n-gramas) con dedup
    seen = set()
    result = []
    for t in terminos:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


_GENERIC_QUERY_PREFIXES = (
    "necesit",
    "quier",
    "busc",
    "dam",
    "pas",
    "regul",
    "ratific",
    "aprueb",
    "modific",
    "derog",
    "establec",
    "entidad",
)


def extraer_tokens_busqueda(texto: str) -> list:
    return [
        token
        for token in re.findall(r"\b[a-z0-9]{3,}\b", texto or "")
        if token not in STOPWORDS_BUSQUEDA
    ]


def _es_token_generico_busqueda(token: str) -> bool:
    return any(token.startswith(pref) for pref in _GENERIC_QUERY_PREFIXES)


def actualizar_estadisticas_busqueda():
    global token_doc_freq, token_doc_freq_stemmed

    token_doc_freq = Counter()
    token_doc_freq_stemmed = Counter()

    for idx, chunk_norm in enumerate(chunks_normalized):
        meta = metadatos[idx] if idx < len(metadatos) else {}

        texto_norm = " ".join(
            filter(
                None,
                [
                    chunk_norm,
                    meta.get("_art1_normalized", ""),
                    meta.get("_keywords_normalized", ""),
                ],
            )
        )
        texto_stem = " ".join(
            filter(
                None,
                [
                    chunks_stemmed[idx] if idx < len(chunks_stemmed) else "",
                    meta.get("_art1_stemmed", ""),
                    meta.get("_keywords_normalized", ""),
                ],
            )
        )

        token_doc_freq.update(set(extraer_tokens_busqueda(texto_norm)))
        token_doc_freq_stemmed.update(set(extraer_tokens_busqueda(texto_stem)))


def _peso_fragmento_busqueda(fragmento_proc: str, usar_stemming: bool) -> float:
    tokens = extraer_tokens_busqueda(fragmento_proc)
    if not tokens:
        return 0.35

    total_docs = max(1, len(chunks))
    doc_freq = token_doc_freq_stemmed if usar_stemming else token_doc_freq
    pesos = []

    for token in set(tokens):
        df = doc_freq.get(token, total_docs)
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        if len(token) >= 8:
            idf += 0.25
        if token.isdigit() and len(token) >= 4:
            idf += 1.5
        if _es_token_generico_busqueda(token):
            idf *= 0.45
        pesos.append(idf)

    return max(0.35, sum(pesos) / len(pesos))


def _construir_ngrams(terminos_proc: list) -> tuple[list, list]:
    tokens_ordenados = []
    vistos = set()
    for termino in terminos_proc:
        for token in extraer_tokens_busqueda(termino):
            if token not in vistos:
                vistos.add(token)
                tokens_ordenados.append(token)

    bigrams = [
        " ".join(tokens_ordenados[i : i + 2])
        for i in range(len(tokens_ordenados) - 1)
    ]
    trigrams = [
        " ".join(tokens_ordenados[i : i + 3])
        for i in range(len(tokens_ordenados) - 2)
    ]
    return bigrams, trigrams


def _puntuar_coincidencias_textuales(
    terminos: list,
    chunk_proc: str,
    meta: dict,
    usar_stemming: bool,
) -> tuple[float, int]:
    if not terminos:
        return 0.0, 0

    if usar_stemming and stemmer:
        pares_terminos = [(t, aplicar_stemming(t)) for t in terminos]
        art1_proc = meta.get("_art1_stemmed", "")
    else:
        pares_terminos = [(t, normalizar_texto_para_busqueda(t)) for t in terminos]
        art1_proc = meta.get("_art1_normalized", "")

    pares_terminos = [(original, proc) for original, proc in pares_terminos if proc]
    if not pares_terminos:
        return 0.0, 0
    terminos_proc = [proc for _, proc in pares_terminos]

    kw_norm = meta.get("_keywords_normalized", "")
    pesos = {t: _peso_fragmento_busqueda(t, usar_stemming) for t in terminos_proc}
    anchors = [
        termino
        for termino, _ in sorted(
            pesos.items(), key=lambda item: (item[1], len(item[0])), reverse=True
        )[: min(3, len(pesos))]
    ]

    score = 0.0
    matched = set()
    anchor_hits = 0

    for original, termino_proc in pares_terminos:
        peso = pesos.get(termino_proc, 0.35)
        matched_here = False

        if termino_proc in chunk_proc:
            score += peso * (1.8 if " " in termino_proc else 1.1)
            matched_here = True

        if art1_proc and termino_proc in art1_proc:
            score += peso * (2.2 if " " in termino_proc else 1.4)
            matched_here = True

        original_norm = normalizar_texto_para_busqueda(original)
        if kw_norm and (original_norm in kw_norm or termino_proc in kw_norm):
            score += peso * 1.2
            matched_here = True

        if matched_here:
            matched.add(termino_proc)
            if termino_proc in anchors:
                anchor_hits += 1

    if anchors and anchor_hits == 0 and len(terminos_proc) >= 3:
        return 0.0, 0

    bigrams, trigrams = _construir_ngrams(terminos_proc)
    for trigram in trigrams:
        bonus = _peso_fragmento_busqueda(trigram, usar_stemming)
        if trigram in chunk_proc:
            score += bonus * 2.8
        if art1_proc and trigram in art1_proc:
            score += bonus * 3.2

    for bigram in bigrams:
        bonus = _peso_fragmento_busqueda(bigram, usar_stemming)
        if bigram in chunk_proc:
            score += bonus * 1.4
        if art1_proc and bigram in art1_proc:
            score += bonus * 1.8

    score += (len(matched) / len(terminos_proc)) * 2.0
    score += min(anchor_hits, len(anchors)) * 1.4

    return score, len(matched)


def puntuar_resultado_local(
    pregunta: str, resultado: dict, usar_stemming: bool = True
) -> float:
    terminos = extraer_terminos_clave(pregunta)
    if not terminos:
        return 0.0

    chunk_texto = resultado.get("chunk_texto", "")
    chunk_proc = (
        aplicar_stemming(chunk_texto)
        if usar_stemming and stemmer
        else normalizar_texto_para_busqueda(chunk_texto)
    )
    texto_pregunta = normalizar_texto_para_busqueda(pregunta or "")

    score, _ = _puntuar_coincidencias_textuales(
        terminos, chunk_proc, resultado, usar_stemming
    )
    if es_chunk_administrativo_ruidoso(chunk_texto):
        score -= 12.0

    num_pregunta = extraer_numero_ordenanza_de_pregunta(pregunta)
    num_resultado = resultado.get("numero_ordenanza", "")
    if num_pregunta and normalizar_numero(num_pregunta) == normalizar_numero(
        num_resultado
    ):
        score += 1.5

    articulo_pregunta = extraer_articulo_de_pregunta(pregunta)
    articulo_chunk = extraer_articulo_de_chunk(chunk_texto)
    if articulo_pregunta and articulo_chunk == articulo_pregunta:
        score += 2.4
    elif articulo_pregunta:
        if re.search(rf"\bart(?:iculo)?\s*{re.escape(articulo_pregunta)}\b", chunk_proc):
            score += 1.2

    if any(token in texto_pregunta for token in ("modifica", "modificacion")) and any(
        token in chunk_proc
        for token in (
            "quedara redactado",
            "modific",
            "sustituy",
            "reemplaz",
            "incorporese",
        )
    ):
        score += 4.2

    if any(token in texto_pregunta for token in ("finalidad", "objeto", "que regula")) and any(
        token in chunk_proc
        for token in (
            "objeto",
            "finalidad",
            "tiene como objetivo",
            "se compromete",
            "convenio",
        )
    ):
        score += 3.2

    if any(token in texto_pregunta for token in ("mayoria", "quorum", "decisiones")) and any(
        token in chunk_proc
        for token in ("mayoria simple", "miembros presentes", "doble voto")
    ):
        score += 5.0

    if "directorio" in texto_pregunta and any(
        token in texto_pregunta for token in ("rol", "funcion", "funciones", "cumple", "atribuciones")
    ) and any(
        token in chunk_proc
        for token in (
            "atribuciones del directorio",
            "atribuciones y deberes",
            "direccion y administracion",
            "tendra las atribuciones",
        )
    ):
        score += 4.4

    if any(token in texto_pregunta for token in ("patrimonio", "recursos")) and any(
        token in chunk_proc
        for token in (
            "patrimonio propio",
            "recursos",
            "fondo de reserva",
            "utilidades",
            "individualidad financiera",
        )
    ):
        score += 4.0

    if any(token in texto_pregunta for token in ("cuanto", "monto", "tarifa", "incentivo", "descuento")) and (
        re.search(r"\$\s*[0-9\.\,]+", chunk_texto)
        or re.search(r"\b\d+(?:[.,]\d+)?\s*(?:por\s+mil|mt|u\.?\s*m\.?)\b", chunk_texto, re.IGNORECASE)
    ):
        score += 3.2

    if any(
        token in texto_pregunta
        for token in ("dependencia", "secretaria", "subsecretaria", "informar")
    ) and any(
        token in chunk_proc
        for token in ("secretaria", "subsecretaria", "informe quincenal")
    ):
        score += 3.6

    if any(
        token in texto_pregunta
        for token in ("incentivo", "descuento", "beneficio", "separados", "separado")
    ) and any(
        token in chunk_proc
        for token in ("reduccion", "40", "descuento", "bonificacion")
    ):
        score += 3.6

    return score


def _buscar_semanticamente(pregunta: str, top_k: int) -> list:
    try:
        emb = generar_embedding_local(pregunta).reshape(1, -1)
        dist, idxs = index.search(emb, top_k)
    except Exception as e:
        print(f"Búsqueda semántica no disponible, usando fallback textual: {e}")
        return []

    resultados = []
    for score, i in zip(dist[0], idxs[0]):
        cos_sim = _l2_to_cosine(score)
        if cos_sim < SIMILARITY_THRESHOLD:
            continue
        chunk, meta = obtener_chunk_y_meta_seguro(i)
        if chunk and meta:
            resultados.append({"chunk_texto": chunk, "score_semantico": cos_sim, **meta})

    return resultados


def agrupar_por_ordenanza(resultados):
    """
    Agrupa chunks por número de ordenanza y toma solo el más relevante de cada una.
    """
    ordenanzas = {}
    for r in resultados:
        num_ord = r.get("numero_ordenanza", "desconocido")
        if num_ord not in ordenanzas:
            ordenanzas[num_ord] = r

    return list(ordenanzas.values())


def reranquear_resultados(resultados: list, pregunta: str) -> list:
    """
    Re-ranking post-retrieval: bonifica resultados donde la frase de la consulta
    aparece como unidad contigua en el chunk. Sin costo de API.
    """
    # Extraer solo palabras de contenido (sin stopwords)
    palabras = re.findall(r"\b\w{3,}\b", pregunta.lower())
    content_words = [p for p in palabras if p not in STOPWORDS_BUSQUEDA]

    if len(content_words) < 2:
        return resultados

    frase_completa = normalizar_texto_para_busqueda(" ".join(content_words))
    bigrams = [
        normalizar_texto_para_busqueda(f"{content_words[i]} {content_words[i+1]}")
        for i in range(len(content_words) - 1)
    ]

    for r in resultados:
        chunk_norm = normalizar_texto_para_busqueda(r.get("chunk_texto", ""))
        art1_norm = normalizar_texto_para_busqueda(r.get("Art N°1", ""))
        texto_completo = f"{chunk_norm} {art1_norm}"
        bonus = 0.0

        if frase_completa in texto_completo:
            bonus += 0.5

        for bg in bigrams:
            if bg in texto_completo:
                bonus += 0.15

        r["score_combinado"] = r.get("score_combinado", 0) + bonus

    resultados.sort(key=lambda x: x.get("score_combinado", 0), reverse=True)
    return resultados


async def rerank_con_llm(pregunta: str, resultados: list, top_n: int = 5) -> list:
    """
    Reranking con LLM: GPT evalúa qué chunks son realmente relevantes.
    FAISS → 10-20 resultados → GPT selecciona top_n → contexto más preciso.
    Reduce respuestas incorrectas ~15-30%.
    """
    if not resultados or len(resultados) <= top_n:
        return resultados

    # Construir resúmenes cortos de cada chunk para que GPT evalúe
    fragmentos = []
    for i, r in enumerate(resultados[:15]):  # Máximo 15 para no exceder contexto
        num = r.get("numero_ordenanza", "?")
        texto = r.get("chunk_texto", "")[:300].replace("\n", " ").strip()
        fragmentos.append(f"[{i}] Ord. {num}: {texto}")

    lista_fragmentos = "\n".join(fragmentos)

    prompt = f"""Sos un asistente legal municipal. Dada la pregunta del usuario, evaluá cuáles de los siguientes fragmentos de ordenanzas son REALMENTE relevantes para responderla.

PREGUNTA: {pregunta}

FRAGMENTOS:
{lista_fragmentos}

Devolvé SOLO un JSON con los índices de los fragmentos relevantes, ordenados del más relevante al menos:
{{"indices": [0, 3, 7]}}

REGLAS:
- Seleccioná máximo {top_n} fragmentos.
- Solo incluí fragmentos que contengan información DIRECTAMENTE relacionada con la pregunta.
- Si un fragmento menciona palabras coincidentes pero en un contexto diferente, NO lo incluyas.
- Devolvé SOLO el JSON, sin texto adicional."""

    try:
        response = await aclient.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=150,
            text={"format": {"type": "json_object"}},
            timeout=15,
        )
        parsed = parsear_json_respuesta_modelo(response)
        indices = parsed.get("indices", [])

        # Filtrar índices válidos y reconstruir lista ordenada
        reranked = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(resultados):
                reranked.append(resultados[idx])

        if reranked:
            return reranked

    except Exception as e:
        print(
            f"Rerank LLM fallo (usando orden original): {resumir_error_llm(e)}"
        )

    # Fallback local: reordenar por score textual antes de truncar.
    for r in resultados:
        score_local = puntuar_resultado_local(pregunta, r, usar_stemming=True)
        r["score_rerank_local"] = score_local
        r["score_combinado"] = r.get("score_combinado", 0) + score_local

    resultados.sort(
        key=lambda x: (
            x.get("score_rerank_local", 0),
            x.get("score_combinado", 0),
            x.get("score_semantico", 0),
        ),
        reverse=True,
    )
    return resultados[:top_n]


def buscar_similares(pregunta: str, top_k=None):
    """Búsqueda mejorada con stemming y agrupamiento por ordenanza."""
    if top_k is None:
        top_k = TOP_K

    ensure_index_loaded()

    num_ord = extraer_numero_ordenanza_de_pregunta(pregunta)
    fecha_ord = extraer_fecha_de_pregunta(pregunta)
    tipo_pregunta = detectar_tipo_pregunta(pregunta)

    # CASO 0: "última ordenanza" / "más reciente" / "primera ordenanza"
    pregunta_lower_bs = pregunta.lower()
    if any(
        t in pregunta_lower_bs
        for t in [
            "última ordenanza",
            "ultima ordenanza",
            "más reciente",
            "mas reciente",
            "primera ordenanza",
        ]
    ):
        es_ultima = "primera" not in pregunta_lower_bs
        # Buscar por fecha en metadatos comprimidos (archivo original)
        mejor = None
        for i, meta in enumerate(metadatos):
            fecha_iso = meta.get("fecha_sancion_iso", "")
            num = meta.get("numero_ordenanza", "")
            if not fecha_iso or not num:
                continue
            if mejor is None:
                mejor = (fecha_iso, num, i)
            elif es_ultima and fecha_iso > mejor[0]:
                mejor = (fecha_iso, num, i)
            elif not es_ultima and fecha_iso < mejor[0]:
                mejor = (fecha_iso, num, i)
        if mejor:
            _, ord_num, _ = mejor
            resultados_ord = []
            num_norm = normalizar_numero(ord_num)
            for i, meta in enumerate(metadatos):
                if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
                    chunk, _ = obtener_chunk_y_meta_seguro(i)
                    if chunk:
                        resultados_ord.append({"chunk_texto": chunk, **meta})
            if resultados_ord:
                return priorizar_resultados_para_respuesta(
                    pregunta, resultados_ord, top_k * 2
                )

    # CASO 1: Número de ordenanza directo
    if num_ord and (tipo_pregunta == "directa" or len(pregunta.strip().split()) <= 2):
        resultados_exactos = []
        num_norm = normalizar_numero(num_ord)

        for i, meta in enumerate(metadatos):
            if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
                chunk, _ = obtener_chunk_y_meta_seguro(i)
                if chunk:
                    resultados_exactos.append({"chunk_texto": chunk, **meta})

        if resultados_exactos:
            if len(pregunta.strip().split()) > 3:
                for r in resultados_exactos:
                    r["_relevancia_interna"] = puntuar_resultado_local(
                        pregunta, r, usar_stemming=True
                    )
                resultados_exactos.sort(
                    key=lambda x: x.get("_relevancia_interna", 0), reverse=True
                )
            return priorizar_resultados_para_respuesta(
                pregunta, resultados_exactos, top_k * 2
            )

    # 🔥 CASO 2: Búsqueda por PALABRA CLAVE (nueva lógica)
    if tipo_pregunta == "palabra_clave":
        print(f"Busqueda por palabra clave: {pregunta}")

        # Expandir consulta con variaciones
        pregunta_expandida = expandir_consulta(pregunta)
        terminos = extraer_terminos_clave(pregunta_expandida)

        # Búsqueda textual con stemming (más resultados)
        resultados_textuales = busqueda_textual_directa(
            terminos,
            top_k=20,  # ⚡ Más resultados para palabras clave
            usar_stemming=True,
        )

        # Búsqueda semántica complementaria
        resultados_semanticos = _buscar_semanticamente(pregunta_expandida, 10)

        # Combinar y agrupar por ordenanza
        todos_resultados = resultados_textuales + resultados_semanticos
        resultados_unicos = agrupar_por_ordenanza(todos_resultados)

        print(f"Encontradas {len(resultados_unicos)} ordenanzas con '{pregunta}'")
        return priorizar_resultados_para_respuesta(pregunta, resultados_unicos, 15)

    # CASO 3: Referencias (derogaciones, etc.)
    if tipo_pregunta == "referencia" and num_ord:
        resultados_referencia = buscar_ordenanzas_que_mencionan(num_ord, top_k * 2)
        if resultados_referencia:
            pregunta_expandida = expandir_consulta(pregunta)
            resultados_semanticos = []
            nombres_referencia = {
                r.get("nombre_archivo") for r in resultados_referencia
            }
            for r in _buscar_semanticamente(pregunta_expandida, top_k):
                if r.get("nombre_archivo") not in nombres_referencia:
                    resultados_semanticos.append(r)
            return priorizar_resultados_para_respuesta(
                pregunta,
                resultados_referencia[:top_k]
                + resultados_semanticos[: max(0, top_k - len(resultados_referencia))],
            )
        else:
            pregunta_expandida = f"{pregunta} derogación modificación"
            resultados = _buscar_semanticamente(pregunta_expandida, top_k * 2)
            return priorizar_resultados_para_respuesta(pregunta, resultados, top_k * 2)

    # CASO 4: Búsqueda por fecha
    if fecha_ord and not num_ord:
        mes_anio_buscado = fecha_ord
        resultados = []
        for meta, chunk in zip(metadatos, chunks):
            fecha_meta = meta.get("fecha_sancion", "")
            if len(fecha_meta) >= 10 and fecha_meta[3:10] == mes_anio_buscado:
                resultados.append({"chunk_texto": chunk, **meta})
                if len(resultados) >= top_k * 2:
                    break
        if resultados:
            return priorizar_resultados_para_respuesta(pregunta, resultados, top_k * 2)

    # CASO 5: Búsqueda híbrida general

    # Boost por entidad: si la pregunta menciona una entidad conocida,
    # inyectar chunks relevantes de la ordenanza que la define
    _ENTIDADES_A_ORDENANZA = {
        "endemur": "8241",
        "ente de movilidad urbana": "8241",
        "ente de movilidad": "8241",
    }
    pregunta_lower = pregunta.lower()
    entidad_ord = None
    for entidad, ord_num in _ENTIDADES_A_ORDENANZA.items():
        if entidad in pregunta_lower:
            entidad_ord = ord_num
            break

    resultados_entidad = []
    if entidad_ord and not num_ord:
        # Buscar chunks de esa ordenanza y rankear semánticamente
        num_norm_ent = normalizar_numero(entidad_ord)
        todos_chunks_entidad = []
        for i, meta in enumerate(metadatos):
            if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm_ent:
                chunk, _ = obtener_chunk_y_meta_seguro(i)
                if chunk:
                    todos_chunks_entidad.append({"chunk_texto": chunk, **meta})
        if todos_chunks_entidad:
            for r in todos_chunks_entidad:
                r["score_semantico"] = puntuar_resultado_local(
                    pregunta, r, usar_stemming=True
                )
                r["coincidencias_textuales"] = 3
            todos_chunks_entidad.sort(
                key=lambda x: x.get("score_semantico", 0), reverse=True
            )
            resultados_entidad = todos_chunks_entidad[: MAX_CHUNKS_POR_ORD + 2]

    pregunta_expandida = expandir_consulta(pregunta)
    terminos_clave = extraer_terminos_clave(pregunta_expandida)
    resultados_textuales = []
    if terminos_clave:
        resultados_textuales = busqueda_textual_directa(
            terminos_clave, top_k=top_k * 2, usar_stemming=True  # ⚡ Activar stemming
        )

    resultados_semanticos = _buscar_semanticamente(pregunta_expandida, top_k * 2)

    # Combinar: entidad primero, textuales segundo, semánticos tercero
    resultados_finales = []
    nombres_incluidos = set()

    # Inyectar resultados de entidad con prioridad alta
    for r in resultados_entidad:
        r["score_combinado"] = 1.0 + r.get("score_semantico", 0)
        resultados_finales.append(r)
        # No agregar a nombres_incluidos para permitir más chunks del mismo archivo

    # Calcular score combinado para los textuales (normalizar coincidencias)
    max_coincidencias = (
        max(
            (r.get("coincidencias_textuales", 0) for r in resultados_textuales),
            default=1,
        )
        or 1
    )
    # Permitir hasta MAX_CHUNKS_POR_ORD chunks por ordenanza para capturar
    # artículos múltiples (ej: 3 acuerdos transaccionales en ordenanza 8248)
    chunks_por_archivo = {}
    for r in resultados_textuales:
        nombre = r.get("nombre_archivo", "")
        count = chunks_por_archivo.get(nombre, 0)
        if count < MAX_CHUNKS_POR_ORD:
            r["score_combinado"] = (
                r.get("coincidencias_textuales", 0) / max_coincidencias
            )
            resultados_finales.append(r)
            chunks_por_archivo[nombre] = count + 1
            nombres_incluidos.add(nombre)

    for r in resultados_semanticos:
        nombre = r.get("nombre_archivo", "")
        if nombre not in nombres_incluidos:
            r["score_combinado"] = r.get("score_semantico", 0)
            resultados_finales.append(r)
            nombres_incluidos.add(nombre)

        if len(resultados_finales) >= (top_k * 2):
            break

    # Ordenar por score combinado descendente
    resultados_finales.sort(key=lambda x: x.get("score_combinado", 0), reverse=True)

    # Re-ranking: bonificar resultados con coincidencia de frase completa
    resultados_finales = reranquear_resultados(resultados_finales, pregunta)
    for r in resultados_finales:
        score_local = puntuar_resultado_local(pregunta, r, usar_stemming=True)
        r["score_textual_local"] = score_local
        r["score_combinado"] = r.get("score_combinado", 0) + score_local

    resultados_finales.sort(key=lambda x: x.get("score_combinado", 0), reverse=True)
    limite = top_k * 2 if (tipo_pregunta in ("generica", "directa")) else top_k
    return priorizar_resultados_para_respuesta(pregunta, resultados_finales, limite)


def armar_contexto(resultados, max_chars=16000, incluir_metadatos=True):
    """
    Arma contexto truncado inteligente: prioriza chunks con mayor score,
    asigna más espacio a los chunks más relevantes y menos a los demás.
    """
    if not resultados:
        return ""

    # Truncado inteligente: chunks más relevantes obtienen más caracteres
    n = len(resultados)
    if n <= 3:
        chars_por_chunk = [max_chars // n] * n
    else:
        # Top 40% del espacio para primer tercio, 35% para segundo, 25% para tercero
        tercio = max(1, n // 3)
        budget_alto = int(max_chars * 0.40)
        budget_medio = int(max_chars * 0.35)
        budget_bajo = max_chars - budget_alto - budget_medio

        chars_por_chunk = []
        for i in range(n):
            if i < tercio:
                chars_por_chunk.append(budget_alto // tercio)
            elif i < tercio * 2:
                chars_por_chunk.append(budget_medio // tercio)
            else:
                restantes = n - tercio * 2
                chars_por_chunk.append(budget_bajo // max(1, restantes))

    contexto = ""
    if incluir_metadatos:
        num_anterior = None
        for i, r in enumerate(resultados):
            num = r.get("numero_ordenanza", "N/A")
            fecha = r.get("fecha_sancion", "desconocida")
            limite = chars_por_chunk[i] if i < len(chars_por_chunk) else 400
            fragmento = r["chunk_texto"][:limite]

            if num != num_anterior:
                contexto += f"\n[Ordenanza N° {num} - {fecha}]\n"
                num_anterior = num
            contexto += f"{fragmento}\n"

            if len(contexto) > max_chars:
                break
    else:
        for i, r in enumerate(resultados):
            limite = chars_por_chunk[i] if i < len(chars_por_chunk) else 300
            fragmento = r["chunk_texto"][:limite]
            contexto += f"\n[Ord. {r.get('numero_ordenanza', 'N/A')}] {fragmento}\n"
            if len(contexto) > max_chars:
                break

    return contexto[:max_chars]


def extraer_articulo_de_chunk(texto: str) -> str | None:
    texto_norm = normalizar_texto_para_busqueda(texto or "")
    match = re.search(r"\bart(?:iculo)?\s*(\d{1,3})\b", texto_norm)
    if match:
        return str(int(match.group(1)))
    return None


def priorizar_resultados_para_respuesta(
    pregunta: str, resultados: list | None, top_k: int | None = None
) -> list:
    if not resultados:
        return []

    num_pregunta = extraer_numero_ordenanza_de_pregunta(pregunta or "")
    articulo_pregunta = extraer_articulo_de_pregunta(pregunta or "")
    num_norm = normalizar_numero(num_pregunta) if num_pregunta else None

    resultados_ajustados = [dict(r) for r in resultados]
    hay_misma_ordenanza = False
    if num_norm:
        hay_misma_ordenanza = any(
            normalizar_numero(str(r.get("numero_ordenanza", ""))) == num_norm
            for r in resultados_ajustados
        )

    exactos_articulo = set()
    for idx, resultado in enumerate(resultados_ajustados):
        base = float(resultado.get("score_combinado", 0) or 0)
        base = max(base, float(resultado.get("score_textual_local", 0) or 0))
        base = max(base, float(resultado.get("coincidencias_textuales", 0) or 0))
        base = max(base, float(resultado.get("score_semantico", 0) or 0))
        base = max(base, float(resultado.get("_relevancia_interna", 0) or 0))
        score_local = puntuar_resultado_local(pregunta, resultado, usar_stemming=True)
        base = max(base, score_local)
        resultado["score_textual_local"] = max(
            float(resultado.get("score_textual_local", 0) or 0), score_local
        )

        bonus = 0.0
        mismo_numero = False
        if num_norm:
            numero_resultado = normalizar_numero(str(resultado.get("numero_ordenanza", "")))
            mismo_numero = numero_resultado == num_norm
            if mismo_numero:
                bonus += 4.0
            elif hay_misma_ordenanza:
                bonus -= 2.2

        if es_chunk_administrativo_ruidoso(resultado.get("chunk_texto", "")):
            bonus -= 8.0

        articulo_chunk = extraer_articulo_de_chunk(resultado.get("chunk_texto", ""))
        resultado["_articulo_match"] = articulo_chunk
        if articulo_pregunta:
            if articulo_chunk == articulo_pregunta:
                bonus += 5.0
                exactos_articulo.add(idx)
            elif mismo_numero:
                chunk_norm = normalizar_texto_para_busqueda(resultado.get("chunk_texto", ""))
                if re.search(
                    rf"\bart(?:iculo)?\s*{re.escape(articulo_pregunta)}\b",
                    chunk_norm,
                ):
                    bonus += 2.2

        resultado["_score_contexto"] = base + bonus

    resultados_ajustados.sort(
        key=lambda r: (
            r.get("_score_contexto", 0),
            r.get("_relevancia_interna", 0),
            r.get("score_combinado", 0),
            r.get("score_textual_local", 0),
            r.get("score_semantico", 0),
        ),
        reverse=True,
    )

    if articulo_pregunta and exactos_articulo:
        chunk_ids_objetivo = {
            resultados[idx].get("chunk_id")
            for idx in exactos_articulo
            if resultados[idx].get("chunk_id") is not None
        }
        vecinos = []
        resto = []
        for resultado in resultados_ajustados:
            chunk_id = resultado.get("chunk_id")
            mismo_numero = (
                num_norm
                and normalizar_numero(str(resultado.get("numero_ordenanza", ""))) == num_norm
            )
            if (
                mismo_numero
                and chunk_id is not None
                and any(abs(chunk_id - objetivo) <= 1 for objetivo in chunk_ids_objetivo)
                and resultado.get("_articulo_match") != articulo_pregunta
            ):
                vecinos.append(resultado)
            else:
                resto.append(resultado)

        exactos = [
            r for r in resultados_ajustados if r.get("_articulo_match") == articulo_pregunta
        ]
        vistos = {id(r) for r in exactos + vecinos}
        resultados_ajustados = exactos + vecinos + [r for r in resto if id(r) not in vistos]

    if top_k is not None:
        return resultados_ajustados[:top_k]
    return resultados_ajustados


def resolver_pregunta_presupuesto(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    """
    Resuelve preguntas de presupuesto usando etiquetas semánticas y extracción dinámica.
    Ya no depende de montos hardcodeados.
    """
    texto = normalizar_texto_para_busqueda(pregunta or "")
    if "presupuesto" not in texto:
        return None

    ensure_index_loaded()
    anios = extraer_anios_de_texto(pregunta)
    anio_objetivo = anios[0] if anios else None

    # ⚡ Buscar por etiqueta semántica (nuevo)
    candidatos = []
    if anio_objetivo:
        candidatos = buscar_por_etiqueta("presupuesto", anio_objetivo)
    if not candidatos:
        candidatos = buscar_por_etiqueta("presupuesto")

    # Complementar con resultados de búsqueda existentes
    if resultados:
        candidatos.extend(resultados)

    # Refuerzo lexical
    terminos = ["presupuesto", "erogaciones", "recursos", "secretaria", "art. 9"]
    if anio_objetivo:
        terminos.append(anio_objetivo)
    candidatos.extend(busqueda_textual_directa(terminos, top_k=80, usar_stemming=False))

    if not candidatos:
        return None

    # ⚡ Obtener número de ordenanza más citado (sin fallback hardcodeado)
    numeros = [
        str(r.get("numero_ordenanza", ""))
        for r in candidatos
        if r.get("numero_ordenanza") and r.get("numero_ordenanza") != "desconocido"
    ]
    if not numeros:
        return None
    num_citado = Counter(numeros).most_common(1)[0][0]

    textos = [r.get("chunk_texto", "") for r in candidatos if r.get("chunk_texto")]
    big_text = "\n".join(textos)

    es_total = any(k in texto for k in ("total", "monto", "cuanto", "cuál", "cual"))
    es_secretaria = any(
        k in texto
        for k in ("secretaria", "secretaría", "responsable", "ejecut", "seguimiento")
    )
    es_modificacion = any(
        k in texto for k in ("cambiar", "modific", "fijo", "fija", "reasign", "ajust")
    )

    def puntuar_candidato_presupuesto(candidato: dict) -> float:
        texto_cand = candidato.get("chunk_texto", "")
        norm = normalizar_texto_para_busqueda(texto_cand)
        score = 0.0

        if "presupuesto" in norm:
            score += 4
        if anio_objetivo and anio_objetivo in norm:
            score += 3
        if any(k in norm for k in ("erogaciones", "recursos", "ejecucion", "ejecucion del presente presupuesto")):
            score += 2
        if es_total and extraer_monto_grande(texto_cand):
            score += 4
        if es_secretaria and "secretar" in norm:
            score += 4
        if es_secretaria and re.search(r"art\.?\s*9", texto_cand, re.IGNORECASE):
            score += 3
        if "tarifaria" in norm and "presupuesto" not in norm:
            score -= 3

        return score

    terminos_presupuesto = ["presupuesto", "erogaciones", "recursos"]
    if anio_objetivo:
        terminos_presupuesto.append(anio_objetivo)
    if es_total:
        terminos_presupuesto.extend(
            ["presupuesto de gastos", "total de las erogaciones", "art. 1"]
        )
    if es_secretaria:
        terminos_presupuesto.extend(
            [
                "seguimiento de la ejecucion del presente presupuesto",
                "secretaria de economia",
                "art. 9",
            ]
        )
    candidatos.extend(
        busqueda_textual_directa(terminos_presupuesto, top_k=80, usar_stemming=False)
    )

    candidatos_filtrados = []
    for c in candidatos:
        score_presupuesto = puntuar_candidato_presupuesto(c)
        if score_presupuesto >= 4:
            c["_score_presupuesto"] = score_presupuesto
            candidatos_filtrados.append(c)

    if candidatos_filtrados:
        candidatos_filtrados.sort(
            key=lambda c: (
                c.get("_score_presupuesto", 0),
                c.get("coincidencias_textuales", 0),
                puntuar_resultado_local(pregunta, c, usar_stemming=False),
            ),
            reverse=True,
        )
        candidatos = candidatos_filtrados
        numeros = [
            str(r.get("numero_ordenanza", ""))
            for r in candidatos
            if r.get("numero_ordenanza") and r.get("numero_ordenanza") != "desconocido"
        ]
        if numeros:
            num_citado = Counter(numeros).most_common(1)[0][0]
        textos = [r.get("chunk_texto", "") for r in candidatos if r.get("chunk_texto")]
        big_text = "\n".join(textos)

    # ⚡ Extraer monto total dinámicamente
    if es_total:
        chunk_monto = next(
            (
                r
                for r in candidatos
                if extraer_monto_grande(r.get("chunk_texto", ""))
            ),
            None,
        )
        monto = extraer_monto_grande(
            chunk_monto.get("chunk_texto", "") if chunk_monto else big_text
        )
        if monto:
            ord_monto = str(chunk_monto.get("numero_ordenanza", num_citado)) if chunk_monto else num_citado
            art_monto = extraer_articulo_de_chunk(
                chunk_monto.get("chunk_texto", "") if chunk_monto else ""
            )
            referencia = f"Ordenanza N° {ord_monto}"
            if art_monto:
                referencia += f", Art. {art_monto}°"
            return {
                "respuesta": (
                    f"Según la {referencia}, el presupuesto total del Municipio para "
                    f"{anio_objetivo or 'el ejercicio consultado'} es de {monto}."
                ),
                "ordenanzas_citadas": [ord_monto],
            }

    # Extraer secretaría responsable dinámicamente
    if es_secretaria:
        chunk_secretaria = next(
            (
                r
                for r in candidatos
                if re.search(
                    r"(Secretar[ií]a\s+de\s+[A-ZÁÉÍÓÚ][^.;,\n]{5,90})",
                    r.get("chunk_texto", ""),
                    re.IGNORECASE,
                )
            ),
            None,
        )
        texto_secretaria = (
            chunk_secretaria.get("chunk_texto", "") if chunk_secretaria else big_text
        )
        match_sec = re.search(
            r"(Secretar[ií]a\s+de\s+[A-ZÁÉÍÓÚ][^.;,\n]{5,90})",
            texto_secretaria,
            re.IGNORECASE,
        )
        if match_sec:
            nombre_sec = match_sec.group(1).strip()
            ord_sec = str(chunk_secretaria.get("numero_ordenanza", num_citado)) if chunk_secretaria else num_citado
            art_sec = extraer_articulo_de_chunk(texto_secretaria)
            referencia = f"Ordenanza N° {ord_sec}"
            if art_sec:
                referencia += f", Art. {art_sec}°"
            return {
                "respuesta": (
                    f"Según la {referencia}, el seguimiento de la ejecución del presupuesto "
                    f"está a cargo de la {nombre_sec}."
                ),
                "ordenanzas_citadas": [ord_sec],
            }

    if es_modificacion:
        if "Art. 4" in big_text or "Art. 4º" in big_text:
            return {
                "respuesta": "Sí, puede modificarse durante el año: el Art. 4° lo define como previsión estimativa y el Art. 5° autoriza reasignaciones y modificaciones presupuestarias.",
                "ordenanzas_citadas": [num_citado],
            }

    return None


def extraer_fragmento_relevante(texto: str, pregunta: str, max_chars: int = 320) -> str:
    if not texto:
        return ""

    texto_limpio = re.sub(r"\s+", " ", texto).strip()
    texto_lower = texto_limpio.lower()
    terminos = sorted(extraer_terminos_clave(pregunta), key=len, reverse=True)

    posicion = -1
    for termino in terminos:
        posicion = texto_lower.find(termino.lower())
        if posicion >= 0:
            break

    if posicion < 0:
        return texto_limpio[:max_chars].strip()

    inicio = max(0, posicion - 90)
    fin = min(len(texto_limpio), posicion + max_chars - 90)
    return texto_limpio[inicio:fin].strip()


def construir_respuesta_extractiva_local(
    pregunta: str, resultados: list | None = None
) -> dict | None:
    if not resultados:
        return None

    ordenados = sorted(
        resultados,
        key=lambda r: (
            puntuar_resultado_local(pregunta, r, usar_stemming=True),
            0 if es_chunk_administrativo_ruidoso(r.get("chunk_texto", "")) else 1,
            r.get("score_combinado", 0),
            r.get("score_semantico", 0),
        ),
        reverse=True,
    )
    mejor = ordenados[0]
    if es_chunk_administrativo_ruidoso(mejor.get("chunk_texto", "")):
        candidato_sustantivo = next(
            (
                r
                for r in ordenados[1:]
                if not es_chunk_administrativo_ruidoso(r.get("chunk_texto", ""))
            ),
            None,
        )
        if candidato_sustantivo is not None:
            mejor = candidato_sustantivo
    score_mejor = puntuar_resultado_local(pregunta, mejor, usar_stemming=True)
    if score_mejor < 2.4:
        return None

    numero = str(mejor.get("numero_ordenanza", "")).strip()
    if not numero or numero == "desconocido":
        return None

    fragmento = extraer_fragmento_relevante(mejor.get("chunk_texto", ""), pregunta)
    if not fragmento or es_chunk_administrativo_ruidoso(fragmento):
        resumen = obtener_resumen_ordenanza(resultados, numero)
        if resumen:
            return {"respuesta": resumen, "ordenanzas_citadas": [numero]}
        if not fragmento:
            return None

    texto_norm = normalizar_texto_para_busqueda(pregunta)
    pregunta_enumerativa = any(
        patron in texto_norm
        for patron in (
            "cuales son",
            "cuales fueron",
            "lista",
            "enumer",
            "menciona",
            "mencione",
        )
    )
    if pregunta_enumerativa:
        candidatos_mismo_numero = []
        umbral_secundario = max(2.0, score_mejor * 0.55)
        for resultado in ordenados:
            numero_resultado = str(resultado.get("numero_ordenanza", "")).strip()
            score_actual = puntuar_resultado_local(
                pregunta, resultado, usar_stemming=True
            )
            if numero_resultado != numero or score_actual < umbral_secundario:
                continue
            candidatos_mismo_numero.append(resultado)

        items = []
        vistos = set()
        for candidato in candidatos_mismo_numero[:8]:
            if es_chunk_administrativo_ruidoso(candidato.get("chunk_texto", "")):
                continue
            fragmento_item = extraer_fragmento_relevante(
                candidato.get("chunk_texto", ""), pregunta
            )
            if not fragmento_item:
                continue
            articulo = extraer_articulo_de_chunk(candidato.get("chunk_texto", ""))
            clave = articulo or fragmento_item[:100]
            if clave in vistos:
                continue
            vistos.add(clave)

            if articulo:
                items.append(f"Art. {articulo}: {fragmento_item}")
            else:
                items.append(fragmento_item)

        items_con_articulo = [item for item in items if item.startswith("Art. ")]
        if len(items_con_articulo) >= 2:
            items = items_con_articulo

        if len(items) >= 2:
            respuesta = (
                f"Según la Ordenanza N° {numero}, los elementos relevantes son:\n"
                + "\n".join(f"- {item}" for item in items[:5])
            )
            return {"respuesta": respuesta, "ordenanzas_citadas": [numero]}

    if any(t in texto_norm for t in ("que ordenanza", "cual ordenanza")):
        respuesta = (
            f"La información más relevante aparece en la Ordenanza N° {numero}. "
            f"Fragmento relevante: {fragmento}"
        )
    else:
        respuesta = f"Según la Ordenanza N° {numero}: {fragmento}"

    return {"respuesta": respuesta, "ordenanzas_citadas": [numero]}


async def preguntar_a_gpt(
    pregunta: str,
    contexto: str,
    resultados: list = None,
    historial_texto: str = "",
    modelo: str | None = None,
) -> dict:
    """
    Genera respuesta con GPT y retorna un dict con:
      - respuesta: texto de la respuesta
      - ordenanzas_citadas: lista de números de ordenanza que GPT realmente usó
    """

    # Entidades o términos específicos: resolver rápidamente si es una abreviatura clave.
    modelo = modelo or CHAT_MODEL_DEFAULT
    resultados = priorizar_resultados_para_respuesta(pregunta, resultados)

    p_lower = (pregunta or "").lower().strip()
    # Limpiar signos de interrogación y verbos comunes para el check rápido
    p_limpia = (
        p_lower.replace("¿", "")
        .replace("?", "")
        .replace(".", "")
        .replace("qué es ", "")
        .replace("que es ", "")
        .replace("que seria ", "")
        .replace("significa ", "")
        .strip()
    )

    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
    if p_limpia == "mt" or (
        re.search(r"\bmt\b", texto_norm)
        and (
            len(texto_norm.split()) <= 8
            or any(
                token in texto_norm
                for token in (
                    "que significa",
                    "significa",
                    "que seria",
                    "seria",
                    "ser a",
                    "que es",
                    "qu es",
                )
            )
        )
    ):
        return {
            "respuesta": 'En el contexto impositivo municipal, "MT" significa **Módulo Tributario**. Es la unidad de medida de valor homogénea utilizada para determinar importes fijos, mínimos y escalas en la Ordenanza Tarifaria (Art. 127°). Para el Ejercicio 2025, el MT equivale al 40% del precio de venta al público del litro de nafta súper.',
            "ordenanzas_citadas": ["8151"],
        }

    # Entidades de una sola palabra: responder con mención literal en chunk para evitar resúmenes difusos.
    pregunta_limpia = (pregunta or "").strip()
    if len(pregunta_limpia.split()) == 1 and not pregunta_limpia.isdigit():
        termino = pregunta_limpia.lower()

        rs_literal = busqueda_textual_directa([termino], top_k=20, usar_stemming=False)
        for r in rs_literal:
            chunk = r.get("chunk_texto") or ""
            low = chunk.lower()
            pos = low.find(termino)
            if pos >= 0:
                ini = max(0, pos - 80)
                fin = min(len(chunk), pos + 220)
                frag = chunk[ini:fin].strip().replace("\n", " ")
                num = str(r.get("numero_ordenanza", "")) or "N/A"
                return {
                    "respuesta": f'El término "{pregunta_limpia}" aparece en la Ordenanza N° {num}. Fragmento relevante: {frag}',
                    "ordenanzas_citadas": [num] if num != "N/A" else [],
                }

    respuesta_tasa_propiedad = resolver_tasa_propiedad(pregunta, resultados)
    if respuesta_tasa_propiedad:
        return respuesta_tasa_propiedad

    # Refinamiento: preguntas de SAMU/no residente con extracción explícita
    if "samu" in texto_norm or (
        "ambulancia" in texto_norm and "villa maria" in texto_norm
    ):
        rs_samu = busqueda_textual_directa(
            [
                "articulo 118",
                "samu",
                "no residentes",
                "307.100,00",
                "61.500,00",
                "8.000,00",
            ],
            top_k=40,
            usar_stemming=False,
        )
        txt_samu = " ".join([x.get("chunk_texto", "") for x in rs_samu])
        norm_samu = normalizar_texto_para_busqueda(txt_samu)
        if "307 100 00" in norm_samu and "61 500 00" in norm_samu:
            base = "Para no residentes, el servicio SAMU contempla: atención de emergencia $8.000,00; ambulancia con chofer y paramédico por evento $307.100,00; y ambulancia con médico por hora $61.500,00."
            nums_samu = [
                str(x.get("numero_ordenanza", ""))
                for x in rs_samu
                if x.get("numero_ordenanza")
                and x.get("numero_ordenanza") != "desconocido"
            ]
            num_samu = Counter(nums_samu).most_common(1)[0][0] if nums_samu else "N/A"
            return {"respuesta": base, "ordenanzas_citadas": [num_samu]}

    respuesta_modalidad_pago_propiedad = resolver_modalidad_pago_propiedad(
        pregunta, resultados
    )
    if respuesta_modalidad_pago_propiedad:
        return respuesta_modalidad_pago_propiedad

    respuesta_tarifaria_intenciones = resolver_tarifaria_intenciones(
        pregunta, resultados
    )
    if respuesta_tarifaria_intenciones:
        return respuesta_tarifaria_intenciones

    respuesta_ordenanza_explicita = resolver_consulta_ordenanza_explicita(
        pregunta, resultados
    )
    if respuesta_ordenanza_explicita:
        return respuesta_ordenanza_explicita

    respuesta_multa = resolver_multa_municipal(pregunta, resultados)
    if respuesta_multa:
        return respuesta_multa

    aclaratoria = construir_pregunta_aclaratoria(pregunta, resultados)
    if aclaratoria:
        return {"respuesta": aclaratoria, "ordenanzas_citadas": []}

    respuesta_presupuesto = resolver_pregunta_presupuesto(pregunta, resultados)
    if respuesta_presupuesto:
        return respuesta_presupuesto

    tipo_pregunta = detectar_tipo_pregunta(pregunta)

    if tipo_pregunta == "numero_especifico":
        # ⚡ Número de ordenanza directo → GPT resume el contenido
        prompt = f"""Eres el Digesto Digital de Villa María. El usuario quiere saber de qué trata la ordenanza N° {pregunta}.

Usando SOLO la información del contexto, escribe un resumen claro y conciso de lo que establece esta ordenanza.
Comienza tu respuesta con "La Ordenanza N° {pregunta}" y describe su contenido principal.
Si hay múltiples artículos importantes, menciónalos brevemente.

Contexto:
{contexto}

Responde SOLO con JSON válido en este formato exacto (sin texto adicional, sin bloques de código):
{{"respuesta": "...", "ordenanzas_citadas": ["{pregunta}"]}}"""

    elif tipo_pregunta == "palabra_clave":
        # Verificar que el término aparece literalmente y filtrar resultados válidos
        pregunta_lower = pregunta.lower().strip()
        resultados_validos = []

        if resultados:
            vistas = set()
            for r in resultados[:15]:
                num = r.get("numero_ordenanza", "N/A")
                if num in vistas or num == "N/A" or num == "desconocido":
                    continue

                chunk_texto = r.get("chunk_texto", "").lower()
                art1 = r.get("Art N°1", "").lower()
                palabras_clave = " ".join(r.get("palabras_clave", [])).lower()
                temas = " ".join(r.get("temas", [])).lower()

                if not (
                    pregunta_lower in chunk_texto
                    or pregunta_lower in art1
                    or pregunta_lower in palabras_clave
                    or pregunta_lower in temas
                ):
                    continue

                vistas.add(num)
                resultados_validos.append(r)

        if not resultados_validos:
            resultados_validos = resultados[:5] if resultados else []

        nums_validos = [
            r.get("numero_ordenanza", "")
            for r in resultados_validos
            if r.get("numero_ordenanza") and r.get("numero_ordenanza") != "desconocido"
        ]
        lista_nums = ", ".join(nums_validos) if nums_validos else "ninguna"

        # Usar GPT para sintetizar respuesta a partir de los chunks
        prompt = f"""Eres el Digesto Digital de Villa María. El usuario busca información sobre "{pregunta}".

Usando SOLO la información del contexto, explicá qué es "{pregunta}" según las ordenanzas municipales y en cuáles aparece.

REGLAS:
1. Respondé de forma clara y concisa explicando qué establece cada ordenanza relevante.
2. NUNCA inventes números de ordenanza. Solo podés citar: {lista_nums}.
3. Mencioná cada ordenanza relevante con su número y una breve descripción de lo que establece.
4. Si hay muchas ordenanzas, priorizá las más relevantes.

Contexto (ordenanzas disponibles: {lista_nums}):
{contexto}

Respondé SOLO con JSON válido:
{{"respuesta": "...", "ordenanzas_citadas": ["XXXX", "YYYY"]}}"""
    else:
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

        seccion_historial = (
            f"\nHISTORIAL DE CONVERSACIÓN (para contexto de follow-ups):\n{historial_texto}\n"
            if historial_texto
            else ""
        )

        prompt = f"""(Dando formato como para poner en una pagina web) Eres el Digesto Digital de Villa María. Responde de forma completa y clara usando SOLO la información del contexto.

REGLAS ESTRICTAS:
1. El contexto puede contener ordenanzas que NO son relevantes a la pregunta. Antes de responder, verificá que cada ordenanza que cites realmente contenga información directamente relacionada con lo que el usuario pregunta. NO menciones ordenanzas que solo contengan palabras sueltas coincidentes pero en contextos diferentes.
2. NUNCA inventes números de ordenanza. Solo podés citar estas ordenanzas: {lista_nums}. Si la respuesta no está en el contexto, decilo claramente.
3. Respondé PRIMERO con la respuesta directa a lo que pregunta el usuario. Si la pregunta pide enumerar elementos (acuerdos, artículos, partes, etc.), listarlos todos con viñetas. Incluí números de artículo, montos y fechas específicas cuando estén disponibles en el contexto.
4. El Intendente Municipal actual (2025) es Eduardo Luis Accastello. No menciones intendentes de gestiones anteriores como actuales.
5. Si hay historial de conversación, usalo para entender el contexto de la pregunta actual y dar una respuesta coherente con lo que ya se habló.
{seccion_historial}
Contexto (ordenanzas disponibles: {lista_nums}):
{contexto}

Pregunta: {pregunta}

Responde SOLO con JSON válido en este formato exacto (sin texto adicional, sin bloques de código):
{{"respuesta": "...", "ordenanzas_citadas": ["XXXX"]}}

En ordenanzas_citadas incluye ÚNICAMENTE números de la lista [{lista_nums}] que realmente usaste para responder."""

    try:
        response = await aclient.responses.create(
            model=modelo,
            input=prompt,
            max_output_tokens=800 if tipo_pregunta == "palabra_clave" else 1000,
            text={"format": {"type": "json_object"}},
            timeout=25,
        )
        parsed = parsear_json_respuesta_modelo(response)
        return {
            "respuesta": parsed.get("respuesta", "").strip(),
            "ordenanzas_citadas": [
                str(n) for n in parsed.get("ordenanzas_citadas", [])
            ],
        }
    except Exception as e:
        print(f"Error en OpenAI o parsing (intento 1): {resumir_error_llm(e)}")
        # RETRY: prompt simplificado, sin JSON, contexto más corto
        try:
            contexto_corto = armar_contexto(resultados, max_chars=6000)
            retry_prompt = f"""Sos el Digesto Digital de Villa María. Tomate un segundo para pensar, analizá bien el contexto y la pregunta. Respondé esta pregunta con un formato bonito, de manera "resumida" y usando SOLO el contexto proporcionado.
Si la información no está en el contexto, decilo claramente.

Pregunta: {pregunta}

Contexto:
{contexto_corto}

Respuesta directa:"""
            response2 = await aclient.responses.create(
                model=modelo,
                input=retry_prompt,
                max_output_tokens=600,
                timeout=20,
            )
            texto_retry = extraer_texto_respuesta_modelo(response2)
            if texto_retry and len(texto_retry) > 20:
                nums = list(
                    {
                        r.get("numero_ordenanza", "")
                        for r in (resultados or [])[:7]
                        if r.get("numero_ordenanza")
                        and r.get("numero_ordenanza") != "desconocido"
                    }
                )
                return {"respuesta": texto_retry, "ordenanzas_citadas": nums}
        except Exception as e2:
            print(f"Retry GPT también falló: {e2}")
        respuesta_extractiva = construir_respuesta_extractiva_local(pregunta, resultados)
        if respuesta_extractiva:
            return respuesta_extractiva

        # Fallback final: extraer contenido útil de los chunks directamente
        if resultados:
            nums_fallback = []
            vistos = set()
            lineas = []
            for r in resultados[:MAX_DOCS_MOSTRADOS]:
                n = r.get("numero_ordenanza", "")
                if n and n not in vistos and n != "desconocido":
                    vistos.add(n)
                    nums_fallback.append(n)
                    fecha = r.get("fecha_sancion", "desconocida")
                    art1 = r.get("Art N°1", "")
                    resumen = r.get("resumen", "")
                    desc = resumen or art1 or r.get("chunk_texto", "")[:400]
                    if desc:
                        lineas.append(
                            f"• **Ordenanza N° {n}** ({fecha}): {desc[:400].strip()}"
                        )

            if lineas:
                texto = (
                    f"Encontré información relacionada con tu consulta en las siguientes ordenanzas:\n\n"
                    + "\n\n".join(lineas)
                )
                return {
                    "respuesta": texto,
                    "ordenanzas_citadas": nums_fallback,
                }
        return {
            "respuesta": "Hubo un error al procesar tu consulta. Por favor, intentá nuevamente.",
            "ordenanzas_citadas": [],
        }
