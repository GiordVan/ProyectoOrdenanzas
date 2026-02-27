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
from datetime import datetime
from functools import lru_cache

# ⚡ NUEVO: Importar stemmer español
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
    print("⚠️ NLTK no instalado. Instalar con: pip install nltk")
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
TOP_K = 10  # ⚡ Aumentado de 3 a 10 para mejor contexto
SIMILARITY_THRESHOLD = 0.60  # ⚡ Umbral mínimo de similitud coseno (0.0-1.0)
MAX_DOCS_MOSTRADOS = 5  # Máximo de documentos únicos a mostrar al usuario

# Variables globales
index = None
metadatos = []
chunks = []
_index_loaded = False
_index_lock = threading.Lock()


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
    """Carga el índice FAISS y metadatos livianos en RAM."""
    global index, metadatos, chunks

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

    print(f"📊 Cargados {len(chunks)} chunks y {len(metadatos)} metadatos expandidos")
    if len(chunks) != len(metadatos):
        print(
            f"⚠️ ADVERTENCIA: Desajuste chunks ({len(chunks)}) vs metadatos ({len(metadatos)})"
        )


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
    Búsqueda textual con stemming y palabras clave.
    Busca en: chunks + palabras_clave + Art N°1 de metadatos.
    """
    resultados = []

    if usar_stemming and stemmer:
        terminos_stem = [aplicar_stemming(t) for t in terminos]
    else:
        terminos_stem = [normalizar_texto_para_busqueda(t) for t in terminos]

    # Usar zip para iterar de forma segura sobre chunks y metadatos
    # Esto evita IndexError si tienen diferente longitud
    for chunk, meta in zip(chunks, metadatos):
        if usar_stemming and stemmer:
            chunk_stem = aplicar_stemming(chunk)
        else:
            chunk_stem = normalizar_texto_para_busqueda(chunk)

        # Buscar en el chunk
        coincidencias = sum(1 for t in terminos_stem if t in chunk_stem)

        # ⚡ Buscar en palabras_clave de metadatos
        palabras_clave = meta.get("palabras_clave", [])
        if palabras_clave:
            palabras_clave_norm = " ".join(palabras_clave).lower()
            for termino_original in terminos:
                if termino_original.lower() in palabras_clave_norm:
                    coincidencias += 2  # Más peso a coincidencias en palabras clave

        # ⚡ NUEVO: Buscar en Art N°1 (campo importante que contiene nombres de empresas, etc.)
        art1 = meta.get("Art N°1", "")
        if art1:
            art1_lower = art1.lower()
            for termino_original in terminos:
                if termino_original.lower() in art1_lower:
                    coincidencias += 3  # ⚡ Alto peso a coincidencias en Art N°1

        if coincidencias > 0:
            meta_copy = dict(meta)
            meta_copy["chunk_texto"] = chunk
            meta_copy["coincidencias_textuales"] = coincidencias
            resultados.append(meta_copy)

    resultados.sort(key=lambda x: x.get("coincidencias_textuales", 0), reverse=True)
    return resultados[:top_k]


def expandir_consulta(pregunta: str) -> str:
    """Expande la consulta con variaciones y sinónimos."""
    pregunta_lower = pregunta.lower()
    expansion = pregunta

    # Diccionario de expansiones
    expansiones = {
        "acta": ["convenio", "acuerdo"],
        "acuerdo": ["convenio", "pacto"],
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


def extraer_terminos_clave(pregunta: str) -> list:
    terminos = []
    stopwords = {
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
        "esta",
        "este",
        "esto",
        "este",
        "aquella",
        "aquel",
        "aquello",
        "estos",
        "estas",
        "aquellos",
        "aquellas",
        "villa",
        "maria",
        "municipalidad",
        "ordenanza",
        "ciudad",
        "intendente",
        "concejo",
        "deliberante",
        "villa",
        "maria",
        "cordoba",
        "ratifica",
        "ratificase",
        "ratificanse",
        "artículo",
        "articulo",
        "provincia",
    }

    palabras = re.findall(r"\b\w{3,}\b", pregunta.lower())
    terminos = [p for p in palabras if p not in stopwords]

    siglas = re.findall(r"\b[A-ZÁÉÍÓÚ]{2,}\b", pregunta)
    terminos.extend([s.lower() for s in siglas])

    return list(set(terminos))


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


def buscar_similares(pregunta: str, top_k=None):
    """Búsqueda mejorada con stemming y agrupamiento por ordenanza."""
    if top_k is None:
        top_k = TOP_K

    ensure_index_loaded()

    num_ord = extraer_numero_ordenanza_de_pregunta(pregunta)
    fecha_ord = extraer_fecha_de_pregunta(pregunta)
    tipo_pregunta = detectar_tipo_pregunta(pregunta)

    # 🔥 CASO 1: Número de ordenanza directo
    if num_ord and (tipo_pregunta == "directa" or len(pregunta.strip().split()) <= 2):
        resultados_exactos = []
        num_norm = normalizar_numero(num_ord)

        for i, meta in enumerate(metadatos):
            if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
                chunk, _ = obtener_chunk_y_meta_seguro(i)
                if chunk:
                    resultados_exactos.append({"chunk_texto": chunk, **meta})

        if resultados_exactos:
            return resultados_exactos[: top_k * 2]

    # 🔥 CASO 2: Búsqueda por PALABRA CLAVE (nueva lógica)
    if tipo_pregunta == "palabra_clave":
        print(f"🔍 Búsqueda por palabra clave: {pregunta}")

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
            if score < SIMILARITY_THRESHOLD:
                continue  # ⚡ Descartar resultados poco relevantes
            chunk, meta = obtener_chunk_y_meta_seguro(i)
            if chunk and meta:
                resultados_semanticos.append(
                    {"chunk_texto": chunk, "score_semantico": float(score), **meta}
                )

        # Combinar y agrupar por ordenanza
        todos_resultados = resultados_textuales + resultados_semanticos
        resultados_unicos = agrupar_por_ordenanza(todos_resultados)

        print(f"✓ Encontradas {len(resultados_unicos)} ordenanzas con '{pregunta}'")
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
        if score < SIMILARITY_THRESHOLD:
            continue  # ⚡ Filtrar semánticos debajo del umbral de relevancia
        chunk, meta = obtener_chunk_y_meta_seguro(i)
        if chunk and meta:
            resultados_semanticos.append(
                {"chunk_texto": chunk, "score_semantico": float(score), **meta}
            )

    # Combinar: textuales primero (más precisos), luego semánticos
    # Los textuales ya tienen coincidencias_textuales como score
    resultados_finales = []
    nombres_incluidos = set()

    # Calcular score combinado para los textuales (normalizar coincidencias)
    max_coincidencias = (
        max(
            (r.get("coincidencias_textuales", 0) for r in resultados_textuales),
            default=1,
        )
        or 1
    )
    for r in resultados_textuales:
        nombre = r.get("nombre_archivo", "")
        if nombre not in nombres_incluidos:
            r["score_combinado"] = (
                r.get("coincidencias_textuales", 0) / max_coincidencias
            )
            resultados_finales.append(r)
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

    limite = top_k * 2 if (tipo_pregunta in ("generica", "directa")) else top_k
    return resultados_finales[:limite]


def armar_contexto(resultados, max_chars=5000, incluir_metadatos=True):
    """Arma contexto truncado con metadatos completos."""
    contexto = ""

    if incluir_metadatos:
        # Formato con metadatos completos
        for r in resultados:
            num = r.get("numero_ordenanza", "N/A")
            fecha = r.get("fecha_sancion", "desconocida")
            fragmento = r["chunk_texto"][:500]  # ⚡ Un poco más de texto por fragmento

            contexto += f"\n[Ordenanza N° {num} - {fecha}]\n{fragmento}\n"

            if len(contexto) > max_chars:
                break
    else:
        # Formato simple (legacy)
        for r in resultados:
            fragmento = r["chunk_texto"][:400]
            contexto += f"\n[Ord. {r.get('numero_ordenanza', 'N/A')}] {fragmento}\n"
            if len(contexto) > max_chars:
                break

    return contexto[:max_chars]


def preguntar_a_gpt(pregunta: str, contexto: str, resultados: list = None) -> dict:
    """
    Genera respuesta con GPT y retorna un dict con:
      - respuesta: texto de la respuesta
      - ordenanzas_citadas: lista de números de ordenanza que GPT realmente usó
    """

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
        # ⚡ Respuesta estructurada sin GPT para palabras clave
        # Pero primero verificar que el término realmente aparece en cada resultado
        pregunta_lower = pregunta.lower().strip()

        if resultados:
            ordenanzas_info = []
            vistas = set()

            for r in resultados[:15]:  # Máximo 15
                num = r.get("numero_ordenanza", "N/A")
                if num in vistas or num == "N/A" or num == "desconocido":
                    continue

                # ⚡ VERIFICACIÓN: El término debe aparecer literalmente en el contenido
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
                    continue  # ⚡ No incluir si el término no aparece literalmente

                vistas.add(num)

                fecha = r.get("fecha_sancion", "desconocida")
                art1_raw = r.get("Art N°1", "")

                # Extraer una breve descripción del Art 1
                if art1_raw:
                    descripcion = art1_raw[:150].strip()
                    if len(art1_raw) > 150:
                        descripcion += "..."
                else:
                    descripcion = r["chunk_texto"][:100].strip() + "..."

                ordenanzas_info.append(
                    {"num": num, "fecha": fecha, "descripcion": descripcion}
                )

            if ordenanzas_info:
                ordenanzas_info.sort(
                    key=lambda x: int(x["num"]) if x["num"].isdigit() else 0,
                    reverse=True,
                )

                # ⚡ Limitar la lista mostrada para consistencia con los documentos
                ordenanzas_mostradas = ordenanzas_info[:MAX_DOCS_MOSTRADOS]
                total = len(ordenanzas_info)

                lineas = []
                for info in ordenanzas_mostradas:
                    lineas.append(f"• **Ordenanza N° {info['num']}** ({info['fecha']})")
                    lineas.append(f"  {info['descripcion']}")
                    lineas.append("")

                lista_formateada = "\n".join(lineas)
                nums_citados = [info["num"] for info in ordenanzas_info]

                # Mensaje con total real y cuántas se muestran
                if total > MAX_DOCS_MOSTRADOS:
                    encabezado = f'El término **"{pregunta}"** aparece en {total} ordenanzas (mostrando las {len(ordenanzas_mostradas)} más relevantes):'
                else:
                    encabezado = f'El término **"{pregunta}"** aparece en {total} ordenanza{"s" if total > 1 else ""}:'

                return {
                    "respuesta": f"{encabezado}\n\n{lista_formateada}",
                    "ordenanzas_citadas": nums_citados,
                }

        # Fallback con GPT si no hay resultados estructurados
        prompt = f"""Eres el Digesto Digital de Villa María. El usuario busca "{pregunta}".

Lista las ordenanzas encontradas con este formato:
- Ordenanza N° XXXX (DD/MM/AAAA): Breve descripción

Contexto:
{contexto}

Responde SOLO con JSON en este formato exacto:
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

        prompt = f"""(Dando formato como para poner en una pagina web) Eres el Digesto Digital de Villa María. Responde en máximo 2 oraciones usando SOLO la información del contexto.

Contexto (ordenanzas disponibles: {lista_nums}):
{contexto}

Pregunta: {pregunta}

Responde SOLO con JSON válido en este formato exacto (sin texto adicional, sin bloques de código):
{{"respuesta": "...", "ordenanzas_citadas": ["XXXX"]}}

En ordenanzas_citadas incluye ÚNICAMENTE los números de ordenanza que realmente usaste para responder."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300 if tipo_pregunta == "palabra_clave" else 200,
            timeout=10,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return {
            "respuesta": parsed.get("respuesta", "").strip(),
            "ordenanzas_citadas": [
                str(n) for n in parsed.get("ordenanzas_citadas", [])
            ],
        }
    except Exception as e:
        print(f"Error en OpenAI o parsing: {e}")
        # Fallback: devolver todos los resultados sin filtrar
        nums_fallback = []
        if resultados:
            vistos = set()
            for r in resultados:
                n = r.get("numero_ordenanza", "")
                if n and n not in vistos:
                    vistos.add(n)
                    nums_fallback.append(n)
        if resultados:
            num_ordenanzas = len(
                set(
                    r.get("numero_ordenanza", "N/A")
                    for r in resultados
                    if r.get("numero_ordenanza") != "N/A"
                )
            )
            return {
                "respuesta": f"Encontradas {num_ordenanzas} ordenanzas relacionadas con '{pregunta}'. Ver documentos para detalles.",
                "ordenanzas_citadas": nums_fallback,
            }
        return {
            "respuesta": "Error al generar respuesta. Intenta nuevamente.",
            "ordenanzas_citadas": [],
        }


async def preguntar_a_gpt_stream(pregunta: str, contexto: str, resultados: list = None):
    """
    Genera respuesta con AsyncOpenAI usando stream=True y la devuelve como generador.
    Como el streaming requiere JSON chunks, cambiaremos el formato o
    extreremos los tokens progresivamente e incluiremos metadatos en un formato sencillo de leer.
    """
    tipo_pregunta = detectar_tipo_pregunta(pregunta)

    # Pre-calculamos números disponibles
    numeros_disponibles = []
    if resultados:
        vistos = set()
        for r in resultados:
            n = r.get("numero_ordenanza", "")
            if n and n not in vistos and n != "desconocido":
                vistos.add(n)
                numeros_disponibles.append(n)

    lista_nums = ", ".join(numeros_disponibles) if numeros_disponibles else "ninguna"

    # Hacemos un prompt que pida responder libremente sin forzar JSON rígido,
    # y que al final ponga las citas.
    prompt = f"""Eres el Digesto Digital de Villa María. Responde de manera clara y concisa usando SOLO la información del contexto. 

Contexto (ordenanzas disponibles: {lista_nums}):
{contexto}

Pregunta: {pregunta}

Responde directamente usando un formato limpio Markdown. No uses la palabra 'JSON'. 
IMPORTANTE: Al final de tu respuesta, en una línea nueva, escribe exactamente "CITAS: " seguido de los números de ordenanza que realmente usaste, separados por coma (ejemplo: "CITAS: 5040, 8050"). Si no usaste ninguna, escribe "CITAS: NINGUNA".
"""

    try:
        stream = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        print(f"Error en OpenAI Async Stream: {e}")
        yield "Hubo un error al generar la respuesta. Por favor, intenta de nuevo."
