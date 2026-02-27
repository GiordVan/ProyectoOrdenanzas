import os
import json
import re
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import openai
from collections import defaultdict, Counter

# Configuración
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CARPETA_PDFS = "PDFs"
CARPETA_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend", "Data"
)
MAX_CHUNK_LEN = 500
NUM_THREADS = 8

# --- Categorías del Digesto ---
CATEGORIAS_DIGESTO_COMPLETO = {
    "A": "Procedimiento administrativo",
    "B": "Policía municipal",
    "C": "Tránsito",
    "D": "Transporte",
    "E": "Dominio Público",
    "F": "Cartelería y publicidad",
    "G": "Uso del suelo de dominio privado",
    "H": "Deportes",
    "I": "Cultura",
    "J": "Régimen Impositivo y Fiscal",
    "K": "Espectáculos Públicos",
    "L": "Régimen de contrataciones",
    "M": "Promoción Industrial",
    "N": "Habilitaciones comerciales",
    "O": "Obras Privadas y Edificación",
    "P": "Régimen de Faltas y sanciones",
    "Q": "Medio Ambiente",
    "R": "Cementerio",
    "S": "Obra Pública",
    "T": "Régimen Político Institucional",
    "U": "Participación vecinal",
    "V": "Régimen laboral del empleado municipal",
    "W": "Denominación de calles y espacios públicos",
    "X": "Servicios Públicos",
    "Y": "Seguridad y Acción Social",
    "Z": "Uso del espacio público",
}


def inferir_categoria_y_temas(texto_art1):
    if not texto_art1:
        return "Z", CATEGORIAS_DIGESTO_COMPLETO["Z"], ["genérico"]
    texto = texto_art1.lower()
    if any(
        kw in texto
        for kw in [
            "transporte",
            "emtupse",
            "pasajeros",
            "línea",
            "recorrido",
            "micro centro",
        ]
    ):
        return (
            "D",
            CATEGORIAS_DIGESTO_COMPLETO["D"],
            [
                "transporte urbano",
                "servicio nocturno",
                "jóvenes",
                "EMTUPSE",
                "boliches",
            ],
        )
    if any(kw in texto for kw in ["impuesto", "tasa", "tributo", "factura", "fiscal"]):
        return (
            "J",
            CATEGORIAS_DIGESTO_COMPLETO["J"],
            ["impuestos", "tasas municipales", "facturación"],
        )
    if any(kw in texto for kw in ["tránsito", "vehículo", "conducir", "ruta"]):
        return (
            "C",
            CATEGORIAS_DIGESTO_COMPLETO["C"],
            ["tránsito", "reglamentación vial"],
        )
    return "Z", CATEGORIAS_DIGESTO_COMPLETO["Z"], ["uso del espacio público"]


def limpiar_texto(texto):
    return re.sub(r"\s+", " ", str(texto)).strip() if texto else ""


# --- EXTRACCIÓN DE PALABRAS CLAVE ---
def extraer_palabras_clave(texto_completo: str, art1: str) -> list:
    """
    Extrae palabras clave relevantes del texto:
    - Nombres propios (personas, lugares, instituciones)
    - Siglas y acrónimos
    - Términos técnicos importantes
    """
    palabras_clave = set()

    # Combinar Art. 1 + primeros 1000 chars del texto completo
    texto_analisis = f"{art1} {texto_completo[:1000]}"

    # === 1. NOMBRES PROPIOS (Mayúscula inicial + apellidos) ===
    # Patrón: "Miguel Ocampo", "Carlos Galoppo", "Fernando Bonfiglioli"
    nombres_propios = re.findall(
        r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+\b", texto_analisis
    )
    for nombre in nombres_propios:
        # Filtrar nombres comunes que no son personas
        if nombre not in [
            "Villa María",
            "Concejo Deliberante",
            "Departamento Ejecutivo",
        ]:
            palabras_clave.add(nombre)

    # === 2. INSTITUCIONES Y ENTIDADES ===
    # Buscar patrones específicos
    patrones_instituciones = [
        r"(?:Fundación|Museo|Centro|Instituto|Empresa|Asociación|Club)\s+[A-ZÁÉÍÓÚÑ][\w\s]{3,50}",
        r"(?:EMTUPSE|SUOEM|UOM|CGT|UEPC)",  # Siglas conocidas
        r"(?:Municipal|Nacional|Provincial)\s+de\s+[\w\s]{3,30}",
    ]

    for patron in patrones_instituciones:
        matches = re.findall(patron, texto_analisis, re.IGNORECASE)
        for match in matches:
            palabras_clave.add(match.strip())

    # === 3. SIGLAS (2-6 letras mayúsculas) ===
    siglas = re.findall(r"\b[A-ZÁÉÍÓÚÑ]{2,6}\b", texto_analisis)
    for sigla in siglas:
        # Filtrar palabras comunes mal detectadas
        if sigla not in [
            "EL",
            "LA",
            "DE",
            "CON",
            "PARA",
            "QUE",
            "VILLA",
            "MARIA",
            "ART",
        ]:
            palabras_clave.add(sigla)

    # === 4. TÉRMINOS CLAVE DEL CONTEXTO ===
    # Extraer sustantivos importantes (palabras con mayúscula en medio de oración)
    terminos_contexto = re.findall(
        r"(?:Visitante|Ilustre|Reconocimiento|Homenaje|Inauguración|Declaración|"
        r"Adhesión|Beneplácito|Aniversario|Celebración|Conmemoración|"
        r"Transporte|Impuesto|Tasa|Tributo|Habilitación|Permiso|"
        r"Servicios?|Obras?|Proyectos?)\b",
        texto_analisis,
        re.IGNORECASE,
    )
    palabras_clave.update([t.capitalize() for t in terminos_contexto])

    # === 5. LUGARES ESPECÍFICOS ===
    # Calles, barrios, plazas, etc.
    lugares = re.findall(
        r"(?:Calle|Avenida|Plaza|Parque|Barrio|Boulevard)\s+[A-ZÁÉÍÓÚÑ][\w\s]{3,30}",
        texto_analisis,
        re.IGNORECASE,
    )
    palabras_clave.update([l.strip() for l in lugares])

    # === 6. NÚMEROS DE OTRAS ORDENANZAS MENCIONADAS ===
    ordenanzas_ref = re.findall(
        r"[Oo]rdenanza\s*[Nn]?[°º]?\s*(\d{4,5})", texto_analisis
    )
    for ord_num in ordenanzas_ref:
        palabras_clave.add(f"Ord.{ord_num}")

    # === 7. LIMPIEZA FINAL ===
    # Eliminar palabras muy cortas o muy largas
    palabras_clave_filtradas = [
        pc for pc in palabras_clave if 3 <= len(pc) <= 50 and pc.strip()
    ]

    # Eliminar duplicados (case-insensitive)
    palabras_unicas = {}
    for pc in palabras_clave_filtradas:
        key = pc.lower()
        if key not in palabras_unicas:
            palabras_unicas[key] = pc

    # Ordenar alfabéticamente y limitar a 20
    resultado = sorted(palabras_unicas.values())[:20]

    return resultado


def limpiar_texto(texto):
    return re.sub(r"\s+", " ", str(texto)).strip() if texto else ""


# --- Detección de PDF escaneado ---
def es_pdf_escaneado(ruta_pdf):
    """
    Detecta si un PDF es escaneado (imagen) o digital (texto nativo).
    Un PDF escaneado tiene poco o ningún texto extraíble directamente.

    Returns:
        True si es escaneado, False si es digital, None si no se puede determinar
    """
    try:
        reader = PdfReader(ruta_pdf)
        texto = "".join((page.extract_text() or "") for page in reader.pages)
        texto_limpio = texto.strip()

        # Si tiene menos de 100 caracteres, probablemente es escaneado
        if len(texto_limpio) < 100:
            return True

        # Si tiene texto sustancial, es digital
        if len(texto_limpio) > 500:
            return False

        # Zona gris: calcular ratio de caracteres legibles vs total
        caracteres_legibles = len(re.findall(r"[a-záéíóúñA-ZÁÉÍÓÚÑ0-9]", texto_limpio))
        if len(texto_limpio) > 0 and caracteres_legibles / len(texto_limpio) < 0.5:
            return True  # Mucho ruido = probablemente escaneado mal procesado

        return None  # No se puede determinar con certeza

    except Exception as e:
        print(f"⚠️ Error al detectar tipo de PDF {ruta_pdf}: {e}")
        return None


# --- Extracción de texto ---
def extraer_texto_pdf(ruta_pdf):
    try:
        reader = PdfReader(ruta_pdf)
        texto = "".join((page.extract_text() or "") + "\n" for page in reader.pages)
        if len(texto.strip()) > 100:
            return texto.strip()
    except Exception as e:
        print(f"⚠️ Falló extracción directa en {ruta_pdf}: {e}")

    print(f"📄 Aplicando OCR a {ruta_pdf}...")
    try:
        imagenes = convert_from_path(
            ruta_pdf, poppler_path=r"C:\poppler\Library\bin", dpi=200
        )
        texto_ocr = "".join(
            pytesseract.image_to_string(img, lang="spa") + "\n" for img in imagenes
        )
        return texto_ocr.strip()
    except Exception as e:
        print(f"❌ Error en OCR de {ruta_pdf}: {e}")
        return ""


# --- EXTRACCIÓN DE METADATOS MEJORADA ---
def numero_texto_a_digito(texto_numero):
    numeros = {
        "un": 1,
        "uno": 1,
        "una": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
        "once": 11,
        "doce": 12,
        "trece": 13,
        "catorce": 14,
        "quince": 15,
        "dieciseis": 16,
        "dieciséis": 16,
        "diecisiete": 17,
        "dieciocho": 18,
        "diecinueve": 19,
        "veinte": 20,
        "veintiun": 21,
        "veintiuno": 21,
        "veintiuna": 21,
        "veintidos": 22,
        "veintidós": 22,
        "veintitres": 23,
        "veintitrés": 23,
        "veinticuatro": 24,
        "veinticinco": 25,
        "veintiseis": 26,
        "veintiséis": 26,
        "veintisiete": 27,
        "veintiocho": 28,
        "veintinueve": 29,
        "treinta": 30,
        "treinta y un": 31,
        "treinta y uno": 31,
    }
    return numeros.get(texto_numero.lower().strip(), None)


def extraer_metadatos_ordenanza(texto):
    """Extracción mejorada de número y fecha de ordenanza"""
    numero = None
    fecha = None

    # === EXTRACCIÓN DE NÚMERO ===
    # Patrón más flexible para números de ordenanza
    patterns_numero = [
        r"ORDENANZA\s+N[°º]?\s*[:.]?\s*(\d{1,2}\.?\d{3,4})",  # ORDENANZA Nº 6.000
        r"ORDENANZA\s+N[°º]?\s+(\d{4,5})",  # ORDENANZA Nº 6000
        r"ORDENANZA\s+(\d{4,5})",  # ORDENANZA 6000
    ]

    for pattern in patterns_numero:
        match = re.search(
            pattern, texto[:500], re.IGNORECASE
        )  # Buscar en primeros 500 chars
        if match:
            numero = match.group(1).replace(".", "").strip()
            print(f"✓ Número detectado: {numero}")
            break

    # === EXTRACCIÓN DE FECHA ===
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
        "setiembre": "09",
        "octubre": "10",
        "noviembre": "11",
        "diciembre": "12",
    }

    anios_texto = {
        "dos mil": "2000",
        "dos mil uno": "2001",
        "dos mil dos": "2002",
        "dos mil tres": "2003",
        "dos mil cuatro": "2004",
        "dos mil cinco": "2005",
        "dos mil seis": "2006",
        "dos mil siete": "2007",
        "dos mil ocho": "2008",
        "dos mil nueve": "2009",
        "dos mil diez": "2010",
        "dos mil once": "2011",
        "dos mil doce": "2012",
        "dos mil trece": "2013",
        "dos mil catorce": "2014",
        "dos mil quince": "2015",
        "dos mil dieciseis": "2016",
        "dos mil dieciséis": "2016",
        "dos mil diecisiete": "2017",
        "dos mil dieciocho": "2018",
        "dos mil diecinueve": "2019",
        "dos mil veinte": "2020",
        "dos mil veintiuno": "2021",
        "dos mil veintidos": "2022",
        "dos mil veintidós": "2022",
        "dos mil veintitres": "2023",
        "dos mil veintitrés": "2023",
        "dos mil veinticuatro": "2024",
        "dos mil veinticinco": "2025",
        "dos mil veintiseis": "2026",
        "dos mil veintiséis": "2026",
    }

    # Buscar en últimos 3000 caracteres (donde suele estar la fecha)
    texto_final = texto[-3000:]

    # === PATRÓN 1: Fecha en texto completo ===
    # "A LOS VEINTINUEVE DÍAS DEL MES DE AGOSTO DEL AÑO DOS MIL OCHO"
    match_texto = re.search(
        r"A\s+LOS\s+([\wÁÉÍÓÚáéíóúÑñ\s]+?)\s+D[IÍ]AS?\s+DEL\s+MES\s+DE\s+([A-ZÑÁÉÍÓÚ]+)\s+DEL\s+(?:AÑO\s+)?([\wÁÉÍÓÚáéíóúÑñ\s]+)",
        texto_final,
        re.IGNORECASE,
    )

    if match_texto:
        dia_texto, mes_texto, anio_texto = match_texto.groups()

        # Convertir día
        dia = numero_texto_a_digito(dia_texto.strip())

        # Convertir mes
        mes = meses.get(mes_texto.lower().strip())

        # Convertir año (limpiar espacios extras)
        anio_clean = " ".join(anio_texto.lower().split())
        anio = anios_texto.get(anio_clean)

        if dia and mes and anio:
            fecha = f"{dia:02d}/{mes}/{anio}"
            print(f"✓ Fecha detectada (texto): {fecha}")
            return numero or "desconocido", fecha

    # === PATRÓN 2: Fecha con números ===
    # "29 DE AGOSTO DE 2008" o "29 DE AGOSTO DEL 2008"
    patterns_fecha = [
        r"\b(\d{1,2})\s+DE\s+([A-ZÑÁÉÍÓÚ]+)\s+DEL?\s+(\d{4})\b",
        r"\b(\d{1,2})\s+de\s+([a-zñáéíóú]+)\s+del?\s+(\d{4})\b",
    ]

    for pattern in patterns_fecha:
        fechas = list(re.finditer(pattern, texto_final, re.IGNORECASE))
        if fechas:
            # Tomar la ÚLTIMA ocurrencia (suele ser la fecha de sanción)
            match = fechas[-1]
            dia, mes_texto, anio = match.groups()

            try:
                dia_int = int(dia)
                anio_int = int(anio)

                # Validar rangos
                if 1 <= dia_int <= 31 and 1900 <= anio_int <= 2030:
                    mes = meses.get(mes_texto.lower().strip())
                    if mes:
                        fecha = f"{dia_int:02d}/{mes}/{anio}"
                        print(f"✓ Fecha detectada (números): {fecha}")
                        return numero or "desconocido", fecha
            except ValueError:
                continue

    # === PATRÓN 3: Fecha al final del documento ===
    # Último intento: buscar cualquier fecha en formato DD/MM/YYYY
    match_slash = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", texto_final)
    if match_slash:
        dia, mes, anio = match_slash.groups()
        try:
            if (
                1 <= int(dia) <= 31
                and 1 <= int(mes) <= 12
                and 1900 <= int(anio) <= 2030
            ):
                fecha = f"{int(dia):02d}/{int(mes):02d}/{anio}"
                print(f"✓ Fecha detectada (formato slash): {fecha}")
                return numero or "desconocido", fecha
        except ValueError:
            pass

    print(f"⚠️ No se detectó fecha en el documento")
    return numero or "desconocido", "desconocida"


def extraer_articulo_1(texto):
    """Extrae el texto completo del Artículo 1°"""
    patterns = [
        r"Art[\.º°]?\s*1[º°]?\s*[-–—:]?\s*(.*?)(?=\n\s*Art[\.º°]?\s*2|\Z)",
        r"Artículo\s+1[º°]?\s*[-–—:]?\s*(.*?)(?=\n\s*Artículo\s+2|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, texto, re.IGNORECASE | re.DOTALL)
        if match:
            return limpiar_texto(match.group(1))
    return ""


def dividir_en_chunks_mejorado(texto, max_len=MAX_CHUNK_LEN):
    articulos = re.split(
        r"(Art(?:ículo|iculo|\.)?\s*\d+[º°]?\s*[-–—]?)", texto, flags=re.IGNORECASE
    )
    if len(articulos) > 2:
        chunks = []
        for i in range(1, len(articulos), 2):
            if i + 1 < len(articulos):
                articulo = (articulos[i] + articulos[i + 1]).strip()
                if len(articulo) <= max_len:
                    chunks.append(articulo)
                else:
                    lineas = articulo.split("\n")
                    buffer = ""
                    for linea in lineas:
                        if len(buffer) + len(linea) + 1 <= max_len:
                            buffer += " " + linea.strip()
                        else:
                            if buffer:
                                chunks.append(buffer.strip())
                            buffer = linea.strip()
                    if buffer:
                        chunks.append(buffer.strip())
        if chunks:
            return chunks
    lineas = texto.split("\n")
    chunks, buffer = [], ""
    for linea in lineas:
        if len(buffer) + len(linea) + 1 <= max_len:
            buffer += " " + linea.strip()
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = linea.strip()
    if buffer:
        chunks.append(buffer.strip())
    return [c for c in chunks if c]


# --- Agrupamiento y combinación MEJORADA ---
def extraer_numero_desde_nombre(nombre):
    """Extrae número de ordenanza desde el nombre del archivo"""
    match = re.search(r"ordenanza[_\s]*(\d+)", nombre, re.IGNORECASE)
    return match.group(1) if match else None


def calcular_score_calidad(candidato):
    """Calcula un score de calidad del documento"""
    score = 0
    texto = candidato["texto"]

    # Criterios de calidad
    if "ORDENANZA" in texto.upper():
        score += 10
    if candidato["numero"] != "desconocido":
        score += 20
    if candidato["fecha"] != "desconocida":
        score += 20
    if candidato["art1"].strip():
        score += 15
    if "Art." in texto[:500] or "Artículo" in texto[:500]:
        score += 10
    if len(texto.split()) > 50:
        score += 10
    if not re.search(r"[^\x00-\x7F]{20,}", texto):  # Menos caracteres raros
        score += 15

    return score


def agrupar_ordenanzas():
    archivos = [f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith(".pdf")]
    grupos = defaultdict(list)
    for archivo in archivos:
        num = extraer_numero_desde_nombre(archivo)
        if num:
            grupos[num].append(archivo)
    return grupos


def combinar_textos_y_metadatos(archivos_grupo, carpeta_pdfs=CARPETA_PDFS):
    """
    Combina información de múltiples versiones del mismo documento
    Prioriza el documento con mejor calidad de extracción para embeddings.
    También detecta cuál PDF es escaneado para subir a la nube.

    Returns:
        tuple: (archivo_mejor_score, texto, metadatos, archivo_para_subir)
        archivo_para_subir: El PDF escaneado si se detecta, sino el de mejor score
    """
    candidatos = []

    print(f"  📂 Analizando {len(archivos_grupo)} archivos...")

    for archivo in archivos_grupo:
        ruta = os.path.join(carpeta_pdfs, archivo)

        # Detectar si es escaneado ANTES de extraer texto (más eficiente)
        es_escaneado = es_pdf_escaneado(ruta)

        texto = extraer_texto_pdf(ruta)

        if not texto.strip():
            print(f"    ⚠️ {archivo}: Sin texto extraíble")
            continue

        # Extraer metadatos
        num, fecha = extraer_metadatos_ordenanza(texto)
        art1 = extraer_articulo_1(texto)

        candidato = {
            "archivo": archivo,
            "texto": texto,
            "numero": num,
            "fecha": fecha,
            "art1": art1,
            "es_escaneado": es_escaneado,
        }

        # Calcular score
        candidato["score"] = calcular_score_calidad(candidato)
        candidatos.append(candidato)

        tipo_pdf = (
            "📷 SCAN"
            if es_escaneado
            else ("📄 Digital" if es_escaneado is False else "❓ Desconocido")
        )
        print(
            f"    {tipo_pdf} {archivo}: score={candidato['score']}, num={num}, fecha={fecha}"
        )

    if not candidatos:
        return None, None, {}, None

    # Seleccionar el mejor candidato por score (para embeddings)
    mejor = max(candidatos, key=lambda x: x["score"])
    print(
        f"  ✓ Mejor archivo (embeddings): {mejor['archivo']} (score: {mejor['score']})"
    )

    # Seleccionar archivo para subir: priorizar el escaneado
    archivo_subir = None
    escaneados = [c for c in candidatos if c["es_escaneado"] is True]

    if escaneados:
        # Si hay un escaneado claro, usar ese
        archivo_subir = escaneados[0]["archivo"]
        print(f"  📤 Archivo para subir (escaneado detectado): {archivo_subir}")
    else:
        # Fallback: usar el de mejor score
        archivo_subir = mejor["archivo"]
        print(f"  📤 Archivo para subir (fallback a mejor score): {archivo_subir}")

    # Si hay múltiples candidatos, intentar completar metadatos faltantes
    if len(candidatos) > 1:
        for c in candidatos:
            if c["archivo"] != mejor["archivo"]:
                if mejor["numero"] == "desconocido" and c["numero"] != "desconocido":
                    mejor["numero"] = c["numero"]
                    print(f"    ℹ️ Número tomado de {c['archivo']}: {c['numero']}")
                if mejor["fecha"] == "desconocida" and c["fecha"] != "desconocida":
                    mejor["fecha"] = c["fecha"]
                    print(f"    ℹ️ Fecha tomada de {c['archivo']}: {c['fecha']}")
                if not mejor["art1"].strip() and c["art1"].strip():
                    mejor["art1"] = c["art1"]
                    print(f"    ℹ️ Art. 1 tomado de {c['archivo']}")

    return (
        mejor["archivo"],
        mejor["texto"],
        {
            "numero_ordenanza": mejor["numero"],
            "fecha_sancion": mejor["fecha"],
            "Art N°1": limpiar_texto(mejor["art1"]),
        },
        archivo_subir,
    )


# --- Procesamiento por grupo ---
def procesar_grupo_ordenanza(numero_ord, lista_archivos, carpeta_pdfs=CARPETA_PDFS):
    """
    Procesa un grupo de archivos de una ordenanza.

    Returns:
        tuple: (metadatos, chunks, embeddings, archivo_para_subir)
    """
    resultado = combinar_textos_y_metadatos(lista_archivos, carpeta_pdfs)
    if not resultado[0]:
        print(f"⚠️ No se pudo procesar la ordenanza {numero_ord}")
        return [], [], [], None

    nombre_archivo, texto_base, metadatos_base, archivo_subir = resultado

    # Inferir metadatos
    cat_letra, cat_desc, temas = inferir_categoria_y_temas(metadatos_base["Art N°1"])

    # ⚡ NUEVO: Extraer palabras clave
    palabras_clave = extraer_palabras_clave(texto_base, metadatos_base["Art N°1"])
    print(
        f"    🔑 Palabras clave: {', '.join(palabras_clave[:5])}{'...' if len(palabras_clave) > 5 else ''}"
    )

    fecha_iso = "desconocida"
    if metadatos_base["fecha_sancion"] != "desconocida":
        try:
            from datetime import datetime

            d = datetime.strptime(metadatos_base["fecha_sancion"], "%d/%m/%Y")
            fecha_iso = d.strftime("%Y-%m-%d")
        except:
            pass

    # Chunks
    chunks_base = dividir_en_chunks_mejorado(texto_base)
    total_chunks = len(chunks_base)

    # Metadatos por chunk
    metadatos = []
    for i, chunk in enumerate(chunks_base):
        metadatos.append(
            {
                "nombre_archivo": nombre_archivo,
                "numero_ordenanza": metadatos_base["numero_ordenanza"],
                "tipo_norma": "ordenanza",
                "fecha_sancion": metadatos_base["fecha_sancion"],
                "fecha_sancion_iso": fecha_iso,
                "estado_vigencia": "vigente",
                "categoria": cat_letra,
                "descripcion_categoria": cat_desc,
                "temas": temas,
                "palabras_clave": palabras_clave,  # ⚡ NUEVO CAMPO
                "autoridad_emisora": "Concejo Deliberante de Villa María",
                "municipio": "Villa María",
                "provincia": "Córdoba",
                "pais": "Argentina",
                "normas_modificadas": [],
                "normas_derogadas": [],
                "normas_que_la_modifican": [],
                "texto_consolidado": False,
                "referencia_digesto": f"{cat_letra}-{metadatos_base['numero_ordenanza']}",
                "Art N°1": metadatos_base["Art N°1"],
                "chunk_id": i,
                "total_chunks": total_chunks,
                "pagina_inicial": 1,
                "pagina_final": 2,
            }
        )

    # Embeddings en batch con OpenAI
    embeddings = []
    if chunks_base:
        batch_size = 500
        for i in range(0, len(chunks_base), batch_size):
            batch = chunks_base[i : i + batch_size]
            response = openai.embeddings.create(
                input=batch, model="text-embedding-3-small"
            )
            embeddings.extend([data.embedding for data in response.data])

    return metadatos, chunks_base, embeddings, archivo_subir


# --- Ejecución con procesamiento incremental ---
def cargar_datos_existentes():
    """Carga datos existentes si existen, sino devuelve listas vacías."""
    metadatos_path = os.path.join(CARPETA_DATA, "metadatos.json")
    chunks_path = os.path.join(CARPETA_DATA, "chunks.json")
    index_path = os.path.join(CARPETA_DATA, "index.faiss")

    metadatos, chunks, embeddings_existentes = [], [], []

    if os.path.exists(metadatos_path):
        with open(metadatos_path, "r", encoding="utf-8") as f:
            metadatos_comprimidos = json.load(f)

        # ⚡ DESCOMPRIMIR: Expandir metadatos comprimidos
        for meta_comp in metadatos_comprimidos:
            chunk_indices = meta_comp.pop("chunk_indices", [0])
            total_chunks = meta_comp.get("total_chunks", len(chunk_indices))

            # Crear un metadato por cada chunk
            for chunk_id in chunk_indices:
                meta_expandido = meta_comp.copy()
                meta_expandido["chunk_id"] = chunk_id
                meta_expandido["total_chunks"] = total_chunks
                metadatos.append(meta_expandido)

        print(
            f"✓ Cargados {len(metadatos)} metadatos expandidos desde {len(metadatos_comprimidos)} ordenanzas"
        )

    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✓ Cargados {len(chunks)} chunks existentes")

    if os.path.exists(index_path):
        index_temp = faiss.read_index(index_path)
        embeddings_existentes = [
            index_temp.reconstruct(i) for i in range(index_temp.ntotal)
        ]
        print(f"✓ Cargados {len(embeddings_existentes)} embeddings existentes")

    return metadatos, chunks, embeddings_existentes


def obtener_ordenanzas_procesadas(metadatos):
    """Retorna set de números de ordenanzas ya procesadas."""
    return set(
        m.get("numero_ordenanza")
        for m in metadatos
        if m.get("numero_ordenanza") != "desconocido"
    )


def guardar_checkpoint(metadatos, chunks, embeddings):
    """Guarda un checkpoint incremental de los datos."""
    os.makedirs(CARPETA_DATA, exist_ok=True)

    # ⚡ COMPRESIÓN: Agrupar metadatos por ordenanza
    metadatos_comprimidos = {}

    for meta in metadatos:
        num_ord = meta["numero_ordenanza"]
        chunk_id = meta["chunk_id"]

        if num_ord not in metadatos_comprimidos:
            # Primera vez que vemos esta ordenanza: guardar todo menos chunk_id
            metadatos_comprimidos[num_ord] = {
                k: v for k, v in meta.items() if k not in ["chunk_id", "total_chunks"]
            }
            metadatos_comprimidos[num_ord]["total_chunks"] = meta["total_chunks"]
            metadatos_comprimidos[num_ord]["chunk_indices"] = []

        # Agregar índice del chunk
        metadatos_comprimidos[num_ord]["chunk_indices"].append(chunk_id)

    # Convertir a lista preservando el orden de aparición en 'metadatos'
    # Creamos una lista de las ordenanzas en el orden que aparecen
    orden_visto = []
    for meta in metadatos:
        num = meta["numero_ordenanza"]
        if num not in orden_visto:
            orden_visto.append(num)

    metadatos_lista = []
    for num in orden_visto:
        if num in metadatos_comprimidos:
            metadatos_lista.append(
                {**metadatos_comprimidos[num], "numero_ordenanza": num}
            )

    # Guardar formato comprimido
    with open(os.path.join(CARPETA_DATA, "metadatos.json"), "w", encoding="utf-8") as f:
        json.dump(metadatos_lista, f, ensure_ascii=False, indent=2)

    # Chunks se mantienen igual (necesarios para búsqueda)
    with open(os.path.join(CARPETA_DATA, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Guardar índice FAISS
    if embeddings:
        embeddings_np = np.array(embeddings, dtype="float32")
        index = faiss.IndexHNSWFlat(embeddings_np.shape[1], 32)
        index.hnsw.efConstruction = 64
        index.add(embeddings_np)
        faiss.write_index(index, os.path.join(CARPETA_DATA, "index.faiss"))


def procesar_todos_documentos(inicio=None, fin=None, guardar_cada=50):
    """
    Procesa documentos de forma incremental con checkpoints.
    PROCESA DE MAYOR A MENOR (de atrás hacia adelante).

    Args:
        inicio: Número de ordenanza inicial (ej: 7000). Si None, procesa todas.
        fin: Número de ordenanza final (ej: 7100). Si None, hasta el final.
        guardar_cada: Guarda checkpoint cada N ordenanzas procesadas.
    """
    # Cargar datos existentes
    print("\n📂 Cargando datos existentes...")
    todos_metadatos, todos_chunks, todos_embeddings = cargar_datos_existentes()
    ordenanzas_procesadas = obtener_ordenanzas_procesadas(todos_metadatos)

    if ordenanzas_procesadas:
        print(f"✓ Ya procesadas: {len(ordenanzas_procesadas)} ordenanzas")
        print(
            f"   Rango: {min(map(int, [o for o in ordenanzas_procesadas if o.isdigit()]))} - {max(map(int, [o for o in ordenanzas_procesadas if o.isdigit()]))}"
        )

    # Agrupar ordenanzas
    grupos = agrupar_ordenanzas()
    if not grupos:
        print("⚠️ No se encontraron PDFs.")
        return

    # Filtrar por rango si se especifica
    numeros_ordenanza = sorted([int(n) for n in grupos.keys() if n.isdigit()])

    if inicio is not None:
        numeros_ordenanza = [n for n in numeros_ordenanza if n >= inicio]
    if fin is not None:
        numeros_ordenanza = [n for n in numeros_ordenanza if n <= fin]

    # Filtrar ordenanzas ya procesadas
    numeros_ordenanza = [
        n for n in numeros_ordenanza if str(n) not in ordenanzas_procesadas
    ]

    # ⚡ INVERTIR ORDEN: Procesar de mayor a menor (de atrás hacia adelante)
    numeros_ordenanza = sorted(numeros_ordenanza, reverse=True)

    if not numeros_ordenanza:
        print("✓ Todas las ordenanzas en el rango ya están procesadas.")
        return

    print(f"\n{'='*60}")
    print(f"Procesando {len(numeros_ordenanza)} ordenanzas nuevas")
    print(f"🔄 ORDEN INVERSO: De mayor a menor")
    if inicio or fin:
        print(f"Rango: {numeros_ordenanza[0]} (inicio) → {numeros_ordenanza[-1]} (fin)")
    print(f"Guardado automático cada {guardar_cada} ordenanzas")
    print(f"{'='*60}\n")

    procesadas_en_sesion = 0
    errores = []

    for idx, num in enumerate(numeros_ordenanza, 1):
        try:
            print(
                f"📦 ORDENANZA {num} ({idx}/{len(numeros_ordenanza)}) ⬅ Procesando de atrás hacia adelante"
            )
            archivos = grupos[str(num)]

            m, c, e, _ = procesar_grupo_ordenanza(str(num), archivos)

            if m:
                todos_metadatos.extend(m)
                todos_chunks.extend(c)
                todos_embeddings.extend(e)
                procesadas_en_sesion += 1

                # Guardar checkpoint cada N ordenanzas
                if procesadas_en_sesion % guardar_cada == 0:
                    print(
                        f"\n💾 Guardando checkpoint ({procesadas_en_sesion} procesadas)..."
                    )
                    guardar_checkpoint(todos_metadatos, todos_chunks, todos_embeddings)
                    print(
                        f"✓ Checkpoint guardado. Total: {len(todos_metadatos)} chunks\n"
                    )
            else:
                errores.append((num, "No se pudo procesar"))

            print()

        except MemoryError:
            print(f"\n⚠️ ERROR DE MEMORIA en ordenanza {num}")
            print(f"💾 Guardando progreso antes de salir...")
            guardar_checkpoint(todos_metadatos, todos_chunks, todos_embeddings)
            print(f"\n{'='*60}")
            print(f"✅ Progreso guardado: {len(todos_metadatos)} chunks")
            print(f"⚠️ Procesamiento interrumpido por falta de memoria")
            print(f"📌 Para continuar desde donde quedó, ejecuta:")
            print(f"   python embedder.py --inicio {inicio or 'N'} --fin {num - 1}")
            print(f"{'='*60}\n")
            return

        except Exception as e:
            print(f"\n❌ ERROR en ordenanza {num}: {e}")
            errores.append((num, str(e)))
            # Continuar con la siguiente
            continue

    # Guardar checkpoint final
    print(f"\n💾 Guardando datos finales...")
    guardar_checkpoint(todos_metadatos, todos_chunks, todos_embeddings)

    print(f"\n{'='*60}")
    print(f"✅ Proceso completado: {len(todos_metadatos)} chunks totales")
    print(f"✅ Procesadas en esta sesión: {procesadas_en_sesion} ordenanzas")
    if errores:
        print(f"⚠️ Errores: {len(errores)}")
        for num, error in errores[:5]:  # Mostrar primeros 5
            print(f"   - Ordenanza {num}: {error}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    # Parsear argumentos de línea de comandos
    inicio = None
    fin = None
    guardar_cada = 50

    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--inicio" and i + 2 < len(sys.argv):
                inicio = int(sys.argv[i + 2])
            elif arg == "--fin" and i + 2 < len(sys.argv):
                fin = int(sys.argv[i + 2])
            elif arg == "--guardar-cada" and i + 2 < len(sys.argv):
                guardar_cada = int(sys.argv[i + 2])
            elif arg == "--help":
                print(
                    """
Uso: python embedder.py [opciones]

Opciones:
  --inicio N          Procesar desde ordenanza N (ej: 7000)
  --fin N             Procesar hasta ordenanza N (ej: 7100)
  --guardar-cada N    Guardar checkpoint cada N ordenanzas (default: 50)
  --help              Mostrar esta ayuda

Ejemplos:
  python embedder.py                           # Procesar todas las ordenanzas
  python embedder.py --inicio 7000             # Desde 7000 hasta el final
  python embedder.py --inicio 7000 --fin 7100  # Solo de 7000 a 7100
  python embedder.py --guardar-cada 25         # Guardar cada 25 ordenanzas
                """
                )
                sys.exit(0)

    procesar_todos_documentos(inicio, fin, guardar_cada)
