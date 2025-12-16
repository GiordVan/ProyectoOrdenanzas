import os
import json
import threading
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ⚡ NUEVO: Importar stemmer español
try:
    from nltk.stem import SnowballStemmer
    import nltk
    # Descargar recursos si es necesario
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    stemmer = SnowballStemmer('spanish')
except ImportError:
    print("⚠️ NLTK no instalado. Instalar con: pip install nltk")
    stemmer = None

# --- Configuración ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DATA_PATH = "Data"
INDEX_FILE = os.path.join(DATA_PATH, "index.faiss")
METADATA_FILE = os.path.join(DATA_PATH, "metadatos.json")
CHUNKS_FILE = os.path.join(DATA_PATH, "chunks.json")
TOP_K = 3

# Variables globales
index = None
metadatos = []
chunks = []
_index_loaded = False
_index_lock = threading.Lock()
embedding_model = SentenceTransformer('BAAI/bge-m3', device='cpu')

def ensure_index_loaded():
    global _index_loaded, index, metadatos, chunks
    if not _index_loaded:
        with _index_lock:
            if not _index_loaded:
                cargar_indice_y_metadatos()
                _index_loaded = True

def normalizar_numero(num: str) -> str:
    return re.sub(r'\D', '', num)

def cargar_indice_y_metadatos():
    """Carga el índice FAISS y metadatos livianos en RAM."""
    global index, metadatos, chunks

    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"No se encontró el archivo {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadatos_comprimidos = json.load(f)
    
    # ⚡ Expandir metadatos comprimidos si es necesario
    metadatos = []
    for meta in metadatos_comprimidos:
        if "chunk_indices" in meta:
            # Formato comprimido: expandir
            chunk_indices = meta.pop("chunk_indices")
            total_chunks = meta.get("total_chunks", len(chunk_indices))
            
            for chunk_id in chunk_indices:
                meta_expandido = meta.copy()
                meta_expandido["chunk_id"] = chunk_id
                meta_expandido["total_chunks"] = total_chunks
                metadatos.append(meta_expandido)
        else:
            # Formato legacy: usar tal cual
            metadatos.append(meta)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

def generar_embedding_local(texto: str):
    return embedding_model.encode(texto, normalize_embeddings=True)

def extraer_numero_ordenanza_de_pregunta(pregunta: str):
    match = re.search(r'(?:ordenanza\s*[n°º]?\s*)?(\d{4,5})', pregunta, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extraer_fecha_de_pregunta(pregunta: str):
    meses = {
        "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
        "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
        "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
    }

    match_texto = re.search(rf'({"|".join(meses.keys())})\s*(?:de\s*)?(\d{{4}})', pregunta, re.IGNORECASE)
    if match_texto:
        mes_palabra = match_texto.group(1).lower()
        anio = match_texto.group(2)
        return f"{meses[mes_palabra]}/{anio}"

    match_num = re.search(r'(0?[1-9]|1[0-2])/(\d{4})', pregunta)
    if match_num:
        mes = int(match_num.group(1))
        anio = match_num.group(2)
        return f"{mes:02d}/{anio}"

    return None

def normalizar_texto_para_busqueda(texto: str) -> str:
    """Normaliza texto eliminando acentos y caracteres especiales."""
    import unicodedata
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
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
    return ' '.join(palabras_stem)

def busqueda_textual_directa(terminos: list, top_k=10, usar_stemming=True):
    """
    Búsqueda textual con stemming y palabras clave.
    Busca en: chunks + palabras_clave de metadatos.
    """
    resultados = []
    
    if usar_stemming and stemmer:
        terminos_stem = [aplicar_stemming(t) for t in terminos]
    else:
        terminos_stem = [normalizar_texto_para_busqueda(t) for t in terminos]
    
    for i, chunk in enumerate(chunks):
        if usar_stemming and stemmer:
            chunk_stem = aplicar_stemming(chunk)
        else:
            chunk_stem = normalizar_texto_para_busqueda(chunk)
        
        meta = metadatos[i]
        
        # Buscar en el chunk
        coincidencias = sum(1 for t in terminos_stem if t in chunk_stem)
        
        # ⚡ NUEVO: Buscar también en palabras_clave de metadatos
        palabras_clave = meta.get("palabras_clave", [])
        if palabras_clave:
            palabras_clave_norm = ' '.join(palabras_clave).lower()
            for termino_original in terminos:  # Usar términos originales para match exacto
                if termino_original.lower() in palabras_clave_norm:
                    coincidencias += 2  # ⚡ Más peso a coincidencias en palabras clave
        
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
    - 'palabra_clave': Una sola palabra o término (ej: "inaugurese", "SUOEM")
    - 'generica': Búsqueda simple sin contexto
    - 'referencia': Pregunta sobre derogaciones, modificaciones
    - 'directa': Pregunta específica con contexto
    """
    pregunta_limpia = pregunta.strip()
    palabras = pregunta_limpia.split()
    
    # Si es una sola palabra, es búsqueda por palabra clave
    if len(palabras) == 1:
        return 'palabra_clave'
    
    palabras_interrogativas = ['qué', 'que', 'cuál', 'cual', 'cuándo', 'cuando', 
                               'cómo', 'como', 'dónde', 'donde', 'quién', 'quien',
                               'por', 'para']
    
    tiene_interrogativa = any(palabra.lower() in palabras_interrogativas for palabra in palabras)
    
    if len(palabras) <= 4 and not tiene_interrogativa:
        return 'generica'
    
    pregunta_lower = pregunta.lower()
    patrones_referencia = [
        r'cu[áa]ndo\s+(?:se\s+)?(?:dio\s+de\s+baja|derog[óo]|modific[óo]|suspend[ióo])',
        r'(?:fue\s+)?(?:dada\s+de\s+baja|derogada|modificada|suspendida)',
        r'qu[ée]\s+ordenanza\s+(?:deroga|modifica|suspende|da\s+de\s+baja)',
        r'qu[ée]\s+norma\s+(?:deroga|modifica|suspende)',
        r'est[áa]\s+(?:vigente|derogada|activa)',
        r'sigue\s+en\s+vigor',
    ]
    
    for patron in patrones_referencia:
        if re.search(patron, pregunta_lower):
            return "referencia"
    
    return "directa"

def buscar_ordenanzas_que_mencionan(numero_ordenanza: str, top_k=10):
    num_norm = normalizar_numero(numero_ordenanza)
    resultados = []
    
    patrones = [
        rf'ordenanza\s*[n°º]?\s*{numero_ordenanza}',
        rf'ordenanza\s*[n°º]?\s*{num_norm}',
        rf'\b{numero_ordenanza}\b',
    ]
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        meta = metadatos[i]
        
        if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
            continue
        
        for patron in patrones:
            if re.search(patron, chunk_lower, re.IGNORECASE):
                palabras_clave = ['derog', 'modific', 'suspend', 'dej', 'sin efecto', 
                                 'baja', 'anula', 'revoca', 'sustitu', 'reemplaza']
                
                if any(palabra in chunk_lower for palabra in palabras_clave):
                    meta_copy = dict(meta)
                    meta_copy["chunk_texto"] = chunk
                    meta_copy["relevancia_referencia"] = True
                    resultados.append(meta_copy)
                    break
    
    return resultados[:top_k]

def extraer_terminos_clave(pregunta: str) -> list:
    terminos = []
    stopwords = {'el', 'la', 'de', 'del', 'en', 'y', 'a', 'que', 'es', 'por', 
                 'para', 'con', 'un', 'una', 'los', 'las', 'se', 'sobre'}
    
    palabras = re.findall(r'\b\w{3,}\b', pregunta.lower())
    terminos = [p for p in palabras if p not in stopwords]
    
    siglas = re.findall(r'\b[A-ZÁÉÍÓÚ]{2,}\b', pregunta)
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

def buscar_similares(pregunta: str, top_k=TOP_K):
    """Búsqueda mejorada con stemming y agrupamiento por ordenanza."""
    num_ord = extraer_numero_ordenanza_de_pregunta(pregunta)
    fecha_ord = extraer_fecha_de_pregunta(pregunta)
    tipo_pregunta = detectar_tipo_pregunta(pregunta)

    # 🔥 CASO 1: Número de ordenanza directo
    if num_ord and (tipo_pregunta == "directa" or len(pregunta.strip().split()) <= 2):
        resultados_exactos = []
        num_norm = normalizar_numero(num_ord)
        
        for i, meta in enumerate(metadatos):
            if normalizar_numero(meta.get("numero_ordenanza", "")) == num_norm:
                resultados_exactos.append({
                    "chunk_texto": chunks[i],
                    **meta
                })
        
        if resultados_exactos:
            return resultados_exactos[:top_k * 2]

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
            usar_stemming=True
        )
        
        # Búsqueda semántica complementaria
        emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
        dist, idxs = index.search(emb, 10)
        resultados_semanticos = [{"chunk_texto": chunks[i], **metadatos[i]} for i in idxs[0]]
        
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
            nombres_referencia = {r.get("nombre_archivo") for r in resultados_referencia}
            for i in idxs[0]:
                meta = dict(metadatos[i])
                meta["chunk_texto"] = chunks[i]
                if meta.get("nombre_archivo") not in nombres_referencia:
                    resultados_semanticos.append(meta)
            return resultados_referencia[:top_k] + resultados_semanticos[:max(0, top_k - len(resultados_referencia))]
        else:
            pregunta_expandida = f"{pregunta} derogación modificación"
            emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
            dist, idxs = index.search(emb, top_k * 2)
            return [{"chunk_texto": chunks[i], **metadatos[i]} for i in idxs[0]]

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
            return resultados[:top_k * 2]

    # CASO 5: Búsqueda híbrida general
    terminos_clave = extraer_terminos_clave(pregunta)
    resultados_textuales = []
    if terminos_clave:
        resultados_textuales = busqueda_textual_directa(
            terminos_clave, 
            top_k=top_k * 2,
            usar_stemming=True  # ⚡ Activar stemming
        )

    pregunta_expandida = expandir_consulta(pregunta)
    emb = generar_embedding_local(pregunta_expandida).reshape(1, -1)
    dist, idxs = index.search(emb, top_k * 2)
    resultados_semanticos = [{"chunk_texto": chunks[i], **metadatos[i]} for i in idxs[0]]

    # Combinar resultados
    resultados_finales = []
    nombres_incluidos = set()

    for r in resultados_textuales:
        nombre = r.get("nombre_archivo", "")
        if nombre not in nombres_incluidos:
            resultados_finales.append(r)
            nombres_incluidos.add(nombre)

    for r in resultados_semanticos:
        nombre = r.get("nombre_archivo", "")
        if nombre not in nombres_incluidos:
            resultados_finales.append(r)
            nombres_incluidos.add(nombre)
        limite = top_k * 2 if tipo_pregunta == "generica" else top_k
        if len(resultados_finales) >= limite:
            break

    if tipo_pregunta == "generica":
        return resultados_finales[:top_k * 2]
    else:
        return resultados_finales[:top_k]

def armar_contexto(resultados, max_chars=1500, incluir_metadatos=True):
    """Arma contexto truncado con metadatos completos."""
    contexto = ""
    
    if incluir_metadatos:
        # Formato con metadatos completos
        for r in resultados:
            num = r.get('numero_ordenanza', 'N/A')
            fecha = r.get('fecha_sancion', 'desconocida')
            fragmento = r['chunk_texto'][:350]
            
            contexto += f"\n[Ordenanza N° {num} - {fecha}]\n{fragmento}\n"
            
            if len(contexto) > max_chars:
                break
    else:
        # Formato simple (legacy)
        for r in resultados:
            fragmento = r['chunk_texto'][:400]
            contexto += f"\n[Ord. {r.get('numero_ordenanza', 'N/A')}] {fragmento}\n"
            if len(contexto) > max_chars:
                break
    
    return contexto[:max_chars]

def preguntar_a_gpt(pregunta: str, contexto: str, resultados: list = None) -> str:
    """Genera respuesta con GPT optimizada."""
    
    tipo_pregunta = detectar_tipo_pregunta(pregunta)
    
    if tipo_pregunta == "palabra_clave":
        # ⚡ FORMATO MEJORADO: Extraer info de resultados directamente
        if resultados:
            ordenanzas_info = []
            vistas = set()
            
            for r in resultados[:10]:  # Máximo 10
                num = r.get('numero_ordenanza', 'N/A')
                if num in vistas or num == 'N/A':
                    continue
                vistas.add(num)
                
                fecha = r.get('fecha_sancion', 'desconocida')
                fragmento = r['chunk_texto'][:200].strip()
                
                # ⚡ NUEVO: Mostrar palabras clave relevantes
                palabras_clave = r.get('palabras_clave', [])
                tags = f" [{', '.join(palabras_clave[:3])}]" if palabras_clave else ""
                
                ordenanzas_info.append(f"- Ordenanza N° {num} ({fecha}): {fragmento}{tags}")
            
            if ordenanzas_info:
                lista_ordenanzas = "\n".join(ordenanzas_info)
                return f"Se encontraron las siguientes ordenanzas con el término '{pregunta}':\n\n{lista_ordenanzas}"
        
        # Fallback con GPT si no hay resultados estructurados
        prompt = f"""Eres el Digesto Digital de Villa María. El usuario busca "{pregunta}".

Lista las ordenanzas encontradas con este formato:
- Ordenanza N° XXXX (DD/MM/AAAA): Breve descripción

Contexto:
{contexto}

Respuesta:"""
    else:
        prompt = f"""(Dando formato como para poner en una pagina web)Ordenanzas de Villa María. Responde en máximo 2 oraciones.

Contexto:
{contexto}

Pregunta: {pregunta}
Respuesta:"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250 if tipo_pregunta == "palabra_clave" else 150,
            timeout=8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error en OpenAI: {e}")
        if resultados:
            num_ordenanzas = len(set(r.get('numero_ordenanza', 'N/A') for r in resultados if r.get('numero_ordenanza') != 'N/A'))
            return f"Encontradas {num_ordenanzas} ordenanzas relacionadas con '{pregunta}'. Ver documentos para detalles."
        return "Error al generar respuesta. Intenta nuevamente."