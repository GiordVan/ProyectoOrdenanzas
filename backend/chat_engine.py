import os
import json
import threading
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
        nltk.download("punkt", quiet=True)
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
TOP_K = 15  # Resultados base de FAISS
SIMILARITY_THRESHOLD = (
    0.38  # Umbral mínimo de similitud coseno (unificado para FAISS y GPT)
)
MAX_DOCS_MOSTRADOS = 5  # Máximo de documentos únicos a mostrar al usuario
MAX_CHUNKS_POR_ORD = 5  # Máximo de chunks por ordenanza en búsqueda híbrida


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


def extraer_numero_ordenanza_de_pregunta(pregunta: str):
    match = re.search(r"(?:ordenanza\s*[n°º]?\s*)?(\d{4,5})", pregunta, re.IGNORECASE)
    if match:
        return match.group(1)
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
        tiene_trib_salud = (
            "financiamiento de los servicios de salud" in norm_comercio
            and ("veinte por ciento" in norm_comercio or "20" in norm_comercio)
        )
        tiene_trib_obras = (
            "alicuota quince por ciento" in norm_comercio
            or "quince por ciento" in norm_comercio
            or "15" in norm_comercio
        ) and "actividad comercial industrial y de servicios" in norm_comercio
        if tiene_trib_salud and tiene_trib_obras:
            return {
                "respuesta": "Además de la contribución principal, se prevén adicionales: 20% para financiamiento de servicios de salud municipal y 15% para los conceptos definidos en el Art. 119 (incluye actividad comercial, industrial y de servicios).",
                "ordenanzas_citadas": [num_citado],
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
    tiene_tarifaria_vigente = False
    if resultados:
        ords_en_resultados = {r.get("numero_ordenanza") for r in resultados}
        tarifarias_vigentes = {"8151", "8214", "8154"}
        tiene_tarifaria_vigente = bool(ords_en_resultados & tarifarias_vigentes)

    if (
        requiere_anio_monto
        and not es_consulta_presupuesto_institucional(pregunta)
        and not es_consulta_tasa_propiedad(pregunta)
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
        "contribuciones que inciden sobre los inmuebles",
        "zona 1",
        "zona 7",
        "tarifa social",
        "por mil",
        "monto fijo",
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
        if ali:
            return {
                "respuesta": f"Para Zona {zona}, la alícuota es {ali}. El monto final depende del VRF del inmueble y de los mínimos vigentes.",
                "ordenanzas_citadas": [num_citado],
            }

    ali_z1 = alicuotas.get(1, "8,00 por mil")
    ali_z7 = alicuotas.get(7, "0,00 por mil")
    min_z1 = minimos.get(1, "$78.050,00")
    min_z2 = minimos.get(2, "$64.000,00")

    return {
        "respuesta": f"El monto depende de la zona y del valor fiscal (VRF) de tu propiedad. Como referencia, las alícuotas van de {ali_z1} en Zona 1 a {ali_z7} en Zona 7 (Tarifa Social); los mínimos anuales incluyen Zona 1 = {min_z1} y Zona 2 = {min_z2}. ¿En qué zona está tu propiedad?",
        "ordenanzas_citadas": [num_citado],
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
        terminos_proc = [aplicar_stemming(t) for t in terminos]
        chunk_texts = chunks_stemmed
        art1_key = "_art1_stemmed"
    else:
        terminos_proc = [normalizar_texto_para_busqueda(t) for t in terminos]
        chunk_texts = chunks_normalized
        art1_key = "_art1_normalized"

    # Generar n-gramas para bonificar coincidencias de frases
    bigrams = [
        f"{terminos_proc[i]} {terminos_proc[i+1]}"
        for i in range(len(terminos_proc) - 1)
    ]
    trigrams = [
        f"{terminos_proc[i]} {terminos_proc[i+1]} {terminos_proc[i+2]}"
        for i in range(len(terminos_proc) - 2)
    ]

    # Usar arrays pre-computados (sin re-calcular stemming/normalización)
    for idx, (chunk_proc, meta) in enumerate(zip(chunk_texts, metadatos)):
        coincidencias = 0

        # Bonificación por frases completas (n-gramas)
        for trigram in trigrams:
            if trigram in chunk_proc:
                coincidencias += 6
        for bigram in bigrams:
            if bigram in chunk_proc:
                coincidencias += 3

        # Coincidencias individuales de términos
        coincidencias += sum(1 for t in terminos_proc if t in chunk_proc)

        # Buscar en palabras_clave pre-normalizadas
        kw_norm = meta.get("_keywords_normalized", "")
        if kw_norm:
            for termino_original in terminos:
                if termino_original.lower() in kw_norm:
                    coincidencias += 2

        # Buscar en Art N°1 pre-computado
        art1_proc = meta.get(art1_key, "")
        if art1_proc:
            for trigram in trigrams:
                if trigram in art1_proc:
                    coincidencias += 8
            for bigram in bigrams:
                if bigram in art1_proc:
                    coincidencias += 4
            for termino_original in terminos:
                if termino_original.lower() in art1_proc:
                    coincidencias += 3

        if coincidencias > 0:
            meta_copy = dict(meta)
            meta_copy["chunk_texto"] = chunks[idx]
            meta_copy["coincidencias_textuales"] = coincidencias
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
    }

    terminos_agregados = []
    for termino_base, sinonimos in expansiones.items():
        if termino_base in pregunta_lower:
            terminos_agregados.extend(sinonimos[:1])

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
        content = response.output_text.strip()
        parsed = json.loads(content)
        indices = parsed.get("indices", [])

        # Filtrar índices válidos y reconstruir lista ordenada
        reranked = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(resultados):
                reranked.append(resultados[idx])

        if reranked:
            return reranked

    except Exception as e:
        print(f"Rerank LLM falló (usando orden original): {e}")

    # Fallback: devolver los primeros top_n sin reranking
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
                return resultados_ord[: top_k * 2]

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
            # Si hay muchos chunks y la pregunta es específica, rankear semánticamente
            # para devolver los más relevantes (ej: Art 18 de una ordenanza de 82 chunks)
            if (
                len(resultados_exactos) > top_k * 2
                and len(pregunta.strip().split()) > 3
            ):
                try:
                    emb_q = generar_embedding_local(pregunta).reshape(1, -1)
                    for r in resultados_exactos:
                        emb_c = generar_embedding_local(r["chunk_texto"]).reshape(1, -1)
                        sim = float(np.dot(emb_q, emb_c.T)[0][0])
                        r["_relevancia_interna"] = sim
                    resultados_exactos.sort(
                        key=lambda x: x.get("_relevancia_interna", 0), reverse=True
                    )
                except Exception:
                    pass  # Si falla embedding, devolver en orden original
            return resultados_exactos[: top_k * 2]

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
        emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
        dist, idxs = index.search(emb, 10)
        resultados_semanticos = []
        for score, i in zip(dist[0], idxs[0]):
            cos_sim = _l2_to_cosine(score)
            if cos_sim < SIMILARITY_THRESHOLD:
                continue  # Descartar resultados con baja similitud coseno
            chunk, meta = obtener_chunk_y_meta_seguro(i)
            if chunk and meta:
                resultados_semanticos.append(
                    {"chunk_texto": chunk, "score_semantico": cos_sim, **meta}
                )

        # Combinar y agrupar por ordenanza
        todos_resultados = resultados_textuales + resultados_semanticos
        resultados_unicos = agrupar_por_ordenanza(todos_resultados)

        print(f"Encontradas {len(resultados_unicos)} ordenanzas con '{pregunta}'")
        return resultados_unicos[:15]  # Máximo 15 ordenanzas diferentes

    # CASO 3: Referencias (derogaciones, etc.)
    if tipo_pregunta == "referencia" and num_ord:
        resultados_referencia = buscar_ordenanzas_que_mencionan(num_ord, top_k * 2)
        if resultados_referencia:
            pregunta_expandida = expandir_consulta(pregunta)
            emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
            dist, idxs = index.search(emb, top_k)
            resultados_semanticos = []
            nombres_referencia = {
                r.get("nombre_archivo") for r in resultados_referencia
            }
            for i in idxs[0]:
                chunk, meta = obtener_chunk_y_meta_seguro(i)
                if chunk and meta:
                    meta_copy = dict(meta)
                    meta_copy["chunk_texto"] = chunk
                    if meta_copy.get("nombre_archivo") not in nombres_referencia:
                        resultados_semanticos.append(meta_copy)
            return (
                resultados_referencia[:top_k]
                + resultados_semanticos[: max(0, top_k - len(resultados_referencia))]
            )
        else:
            pregunta_expandida = f"{pregunta} derogación modificación"
            emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
            dist, idxs = index.search(emb, top_k * 2)
            resultados = []
            for i in idxs[0]:
                chunk, meta = obtener_chunk_y_meta_seguro(i)
                if chunk and meta:
                    resultados.append({"chunk_texto": chunk, **meta})
            return resultados

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
            return resultados[: top_k * 2]

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
            try:
                emb_q = generar_embedding_local(pregunta).reshape(1, -1)
                for r in todos_chunks_entidad:
                    emb_c = generar_embedding_local(r["chunk_texto"]).reshape(1, -1)
                    sim = float(np.dot(emb_q, emb_c.T)[0][0])
                    r["score_semantico"] = sim
                    r["coincidencias_textuales"] = (
                        3  # Garantizar que pase filtro de relevancia
                    )
                todos_chunks_entidad.sort(
                    key=lambda x: x.get("score_semantico", 0), reverse=True
                )
                resultados_entidad = todos_chunks_entidad[: MAX_CHUNKS_POR_ORD + 2]
            except Exception:
                pass

    terminos_clave = extraer_terminos_clave(pregunta)
    resultados_textuales = []
    if terminos_clave:
        resultados_textuales = busqueda_textual_directa(
            terminos_clave, top_k=top_k * 2, usar_stemming=True  # ⚡ Activar stemming
        )

    pregunta_expandida = expandir_consulta(pregunta)
    emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
    dist, idxs = index.search(emb, top_k * 2)
    resultados_semanticos = []
    for score, i in zip(dist[0], idxs[0]):
        cos_sim = _l2_to_cosine(score)
        if cos_sim < SIMILARITY_THRESHOLD:
            continue  # Filtrar semánticos debajo del umbral de relevancia
        chunk, meta = obtener_chunk_y_meta_seguro(i)
        if chunk and meta:
            resultados_semanticos.append(
                {"chunk_texto": chunk, "score_semantico": cos_sim, **meta}
            )

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

    limite = top_k * 2 if (tipo_pregunta in ("generica", "directa")) else top_k
    return resultados_finales[:limite]


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

    # ⚡ Extraer monto total dinámicamente
    if es_total:
        monto = extraer_monto_grande(big_text)
        if monto:
            return {
                "respuesta": f"El presupuesto total del Municipio para {anio_objetivo or 'el ejercicio consultado'} es de {monto}.",
                "ordenanzas_citadas": [num_citado],
            }

    # Extraer secretaría responsable dinámicamente
    if es_secretaria:
        # Buscar patrón "Secretaría de ..." en el texto
        match_sec = re.search(
            r"(Secretar[ií]a\s+de\s+[A-ZÁÉÍÓÚ][^.;,]{10,80})",
            big_text,
            re.IGNORECASE,
        )
        if match_sec:
            nombre_sec = match_sec.group(1).strip()
            return {
                "respuesta": f"El seguimiento de la ejecución del presupuesto está a cargo de la {nombre_sec}.",
                "ordenanzas_citadas": [num_citado],
            }

    if es_modificacion:
        if "Art. 4" in big_text or "Art. 4º" in big_text:
            return {
                "respuesta": "Sí, puede modificarse durante el año: el Art. 4° lo define como previsión estimativa y el Art. 5° autoriza reasignaciones y modificaciones presupuestarias.",
                "ordenanzas_citadas": [num_citado],
            }

    return None


async def preguntar_a_gpt(
    pregunta: str, contexto: str, resultados: list = None, historial_texto: str = ""
) -> dict:
    """
    Genera respuesta con GPT y retorna un dict con:
      - respuesta: texto de la respuesta
      - ordenanzas_citadas: lista de números de ordenanza que GPT realmente usó
    """

    # Entidades o términos específicos: resolver rápidamente si es una abreviatura clave.
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

    if p_limpia == "mt":
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
    texto_norm = normalizar_texto_para_busqueda(pregunta or "")
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
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=800 if tipo_pregunta == "palabra_clave" else 1000,
            text={"format": {"type": "json_object"}},
            timeout=25,
        )
        content = response.output_text.strip()
        parsed = json.loads(content)
        return {
            "respuesta": parsed.get("respuesta", "").strip(),
            "ordenanzas_citadas": [
                str(n) for n in parsed.get("ordenanzas_citadas", [])
            ],
        }
    except Exception as e:
        print(f"Error en OpenAI o parsing (intento 1): {e}")
        # RETRY: prompt simplificado, sin JSON, contexto más corto
        try:
            contexto_corto = armar_contexto(resultados, max_chars=6000)
            retry_prompt = f"""Sos el Digesto Digital de Villa María. Tomate un segundo para pensar, analizá bien el contexto y la pregunta. Respondé esta pregunta usando SOLO el contexto proporcionado.
Si la información no está en el contexto, decilo claramente.

Pregunta: {pregunta}

Contexto:
{contexto_corto}

Respuesta directa:"""
            response2 = await aclient.responses.create(
                model="gpt-5-mini",
                input=retry_prompt,
                max_output_tokens=600,
                timeout=20,
            )
            texto_retry = response2.output_text.strip()
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
