"""
Evaluador automático de respuestas RAG.
Genera AnalisisRespuestas.json con scores detallados por pregunta.
"""
import json
import os
import sys
import asyncio
import re
from pathlib import Path

# Agregar el directorio raíz al path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Cargar .env del backend
load_dotenv(ROOT / "backend" / ".env")

# Configuración
BACKEND_DATA = ROOT / "backend" / "Data"
METADATOS_PATH = BACKEND_DATA / "metadatos.json"
CHUNKS_PATH = BACKEND_DATA / "chunks.json"
RESPUESTAS_PATH = ROOT / "Test" / "RespuestasTest.json"
OUTPUT_PATH = ROOT / "Test" / "AnalisisRespuestas.json"

aclient = AsyncOpenAI()

# Cargar datos
def cargar_datos():
    with open(METADATOS_PATH, "r", encoding="utf-8") as f:
        metadatos = json.load(f)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(RESPUESTAS_PATH, "r", encoding="utf-8") as f:
        respuestas = json.load(f)
    return metadatos, chunks, respuestas


def buscar_chunks_por_ordenanza(num_ord: str, metadatos: list, chunks: list) -> list:
    """Busca los chunks correspondientes a una ordenanza."""
    resultados = []
    offset = 0
    for meta in metadatos:
        n_chunks = meta.get("total_chunks", len(meta.get("chunk_indices", [])))
        if meta.get("numero_ordenanza", "") == num_ord:
            for i in range(n_chunks):
                idx = offset + i
                if idx < len(chunks):
                    resultados.append(chunks[idx])
            break
        offset += n_chunks
    return resultados


def obtener_contexto_referencia(item: dict, metadatos: list, chunks: list) -> str:
    """Obtiene el contexto de referencia para evaluar la respuesta."""
    ords_ref = item.get("ordenanzas_referenciadas", [])
    contexto_parts = []

    # Buscar por ordenanzas referenciadas
    for ref in ords_ref:
        # Limpiar referencia (puede ser "8241" o "Presupuesto 2025")
        num = re.sub(r'\D', '', ref)
        if num:
            chunks_ord = buscar_chunks_por_ordenanza(num, metadatos, chunks)
            if chunks_ord:
                texto = "\n".join(chunks_ord[:5])  # Max 5 chunks
                contexto_parts.append(f"=== Ordenanza {num} ===\n{texto[:3000]}")

    # Si no hay refs específicas, buscar en metadatos por keywords
    if not contexto_parts:
        pregunta_lower = item.get("pregunta", "").lower()
        for meta in metadatos:
            num = meta.get("numero_ordenanza", "")
            # Buscar si la pregunta menciona el número
            if num and num in item.get("pregunta", ""):
                chunks_ord = buscar_chunks_por_ordenanza(num, metadatos, chunks)
                if chunks_ord:
                    texto = "\n".join(chunks_ord[:3])
                    contexto_parts.append(f"=== Ordenanza {num} ===\n{texto[:2000]}")

    return "\n\n".join(contexto_parts) if contexto_parts else "(Sin contexto de referencia disponible)"


async def evaluar_respuesta(item: dict, contexto_ref: str) -> dict:
    """Evalúa una respuesta individual usando GPT."""
    pregunta = item.get("pregunta", "")
    respuesta = item.get("respuesta_modelo", "")
    categoria = item.get("categoria", "")
    dificultad = item.get("dificultad", "media")
    tipo_test = item.get("tipo_test", "")
    requiere_repregunta = item.get("requiere_repregunta_según_test", False)

    repregunta_enviada = item.get("repregunta_enviada")
    respuesta_repregunta = item.get("respuesta_repregunta")

    es_timeout = "ERROR EXCEPCION" in respuesta and "timed out" in respuesta
    es_fallback = respuesta.startswith("Encontré información relacionada")
    es_no_encontre = respuesta.startswith("No encontré información relevante")

    # Construir prompt de evaluación
    eval_prompt = f"""Sos un evaluador experto de sistemas RAG para consultas sobre ordenanzas municipales de Villa María (Argentina).

PREGUNTA: {pregunta}
CATEGORÍA: {categoria}
DIFICULTAD: {dificultad}
TIPO TEST: {tipo_test}
REQUIERE REPREGUNTA SEGÚN TEST: {requiere_repregunta}

RESPUESTA DEL MODELO:
{respuesta}

CONTEXTO DE REFERENCIA (chunks reales de las ordenanzas):
{contexto_ref[:4000]}
"""

    if repregunta_enviada and respuesta_repregunta:
        eval_prompt += f"""
REPREGUNTA ENVIADA: {repregunta_enviada}
RESPUESTA A LA REPREGUNTA:
{respuesta_repregunta}
"""

    eval_prompt += """
CRITERIOS DE EVALUACIÓN (evaluá cada uno por separado):

1. **comprension** (0-20): ¿El modelo entendió correctamente lo que se preguntaba?
   - 20: Entendió perfectamente
   - 10-15: Entendió parcialmente
   - 0-5: No entendió la pregunta

2. **exactitud** (0-30): ¿La respuesta es factualmente correcta según el contexto de referencia?
   - 25-30: Datos correctos con artículos/montos/fechas específicos
   - 15-24: Correcta pero incompleta o sin citas específicas
   - 5-14: Parcialmente correcta, algunos errores
   - 0-4: Incorrecta o inventada

3. **uso_contexto** (0-20): ¿Usó bien el contexto RAG (cita ordenanzas, artículos)?
   - 15-20: Cita artículos y ordenanzas específicas
   - 8-14: Menciona ordenanzas pero sin detalle
   - 0-7: No cita fuentes o cita fuentes incorrectas

4. **no_alucinacion** (0-20): ¿Evitó inventar datos que no están en el contexto?
   - 18-20: No alucina, dice "no encontré" cuando corresponde
   - 10-17: Alguna inferencia razonable
   - 0-9: Inventa datos o da información falsa

5. **claridad_completitud** (0-10): ¿La respuesta es clara, completa y bien estructurada?
   - 8-10: Clara, bien formateada, completa
   - 4-7: Aceptable pero mejorable
   - 0-3: Confusa, cortada o mal formateada

PENALIZACIONES (aplicar si corresponde):
- ALUCINACION: -25 si inventa hechos no presentes en el contexto
- DATOS_IGNORADOS: -15 si ignora datos obvios del contexto
- VAGUEDAD: -5 si responde de forma muy vaga pudiendo ser específico
- CONTRADICCION_PDF: -30 si contradice el contenido real de la ordenanza
- INCOMPRENSION_REPREGUNTA: -20 si no entiende la repregunta

SCORE REPREGUNTA (0-100, solo si hay repregunta):
Evaluar si la respuesta a la repregunta:
- Mantiene contexto de la conversación previa
- Responde correctamente a lo que se le aclaró
- No repite información ya dada
- No cae en fallback genérico

RESPUESTA FALLBACK: Si la respuesta empieza con "Encontré información relacionada..." es un fallback genérico que merece score bajo (~20-30).
TIMEOUT: Si la respuesta contiene "ERROR EXCEPCION" y "timed out" es un timeout técnico, score 0.
NO ENCONTRÉ: Si dice "No encontré información relevante" evaluar si es correcto (la info realmente no existe) o incorrecto (sí existe pero no la encontró).

Respondé SOLO con JSON válido, sin texto adicional:
{
  "comprension": <0-20>,
  "exactitud": <0-30>,
  "uso_contexto": <0-20>,
  "no_alucinacion": <0-20>,
  "claridad_completitud": <0-10>,
  "penalizaciones": [{"tipo": "NOMBRE", "puntos": -N, "motivo": "..."}],
  "score_total": <suma de sub-scores + penalizaciones, min 0>,
  "score_repregunta": <0-100 o null>,
  "notas": "breve explicación del score"
}"""

    try:
        response = await aclient.responses.create(
            model="gpt-4o-mini",
            input=eval_prompt,
            max_output_tokens=800,
            temperature=0.1,
        )
        texto = response.output_text.strip()

        # Limpiar markdown si viene envuelto
        if texto.startswith("```"):
            texto = re.sub(r'^```(?:json)?\s*', '', texto)
            texto = re.sub(r'\s*```$', '', texto)

        resultado = json.loads(texto)
        return resultado
    except json.JSONDecodeError as e:
        print(f"  Error parseando JSON de evaluación: {e}")
        print(f"  Texto recibido: {texto[:200]}")
        # Fallback manual
        if es_timeout:
            return {"comprension": 0, "exactitud": 0, "uso_contexto": 0, "no_alucinacion": 20,
                    "claridad_completitud": 0, "penalizaciones": [], "score_total": 0,
                    "score_repregunta": None, "notas": "Timeout técnico"}
        elif es_fallback:
            return {"comprension": 10, "exactitud": 5, "uso_contexto": 5, "no_alucinacion": 15,
                    "claridad_completitud": 3, "penalizaciones": [{"tipo": "VAGUEDAD", "puntos": -5, "motivo": "Fallback genérico"}],
                    "score_total": 33, "score_repregunta": None, "notas": "Fallback genérico"}
        return {"comprension": 10, "exactitud": 10, "uso_contexto": 10, "no_alucinacion": 15,
                "claridad_completitud": 5, "penalizaciones": [], "score_total": 50,
                "score_repregunta": None, "notas": "Error de evaluación, score estimado"}
    except Exception as e:
        print(f"  Error en evaluación GPT: {e}")
        return {"comprension": 0, "exactitud": 0, "uso_contexto": 0, "no_alucinacion": 0,
                "claridad_completitud": 0, "penalizaciones": [], "score_total": 0,
                "score_repregunta": None, "notas": f"Error: {str(e)}"}


async def main():
    print("Cargando datos...")
    metadatos, chunks, respuestas_data = cargar_datos()

    items = respuestas_data.get("resultados", [])
    total = len(items)
    print(f"Total de preguntas a evaluar: {total}")

    resultados_analisis = []

    # Evaluar en lotes de 5 para no saturar la API
    BATCH_SIZE = 5
    for batch_start in range(0, total, BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        tasks = []

        for item in batch:
            id_q = item.get("id", "?")
            pregunta = item.get("pregunta", "")
            print(f"  Evaluando ID {id_q}: {pregunta[:60]}...")

            contexto_ref = obtener_contexto_referencia(item, metadatos, chunks)
            tasks.append(evaluar_respuesta(item, contexto_ref))

        evaluaciones = await asyncio.gather(*tasks)

        for item, eval_result in zip(batch, evaluaciones):
            resultado = {
                "id": item.get("id"),
                "pregunta": item.get("pregunta"),
                "categoria": item.get("categoria"),
                "dificultad": item.get("dificultad"),
                "tipo_test": item.get("tipo_test"),
                "respuesta_modelo": item.get("respuesta_modelo", "")[:200] + ("..." if len(item.get("respuesta_modelo", "")) > 200 else ""),
                "es_fallback": item.get("respuesta_modelo", "").startswith("Encontré información"),
                "es_timeout": "ERROR EXCEPCION" in item.get("respuesta_modelo", ""),
                "tiene_repregunta": item.get("repregunta_enviada") is not None,
                "evaluacion": eval_result
            }
            resultados_analisis.append(resultado)

    # Calcular estadísticas
    scores = [r["evaluacion"]["score_total"] for r in resultados_analisis]
    scores_repregunta = [r["evaluacion"]["score_repregunta"] for r in resultados_analisis
                         if r["evaluacion"].get("score_repregunta") is not None]

    n_fallback = sum(1 for r in resultados_analisis if r["es_fallback"])
    n_timeout = sum(1 for r in resultados_analisis if r["es_timeout"])
    n_directa = total - n_fallback - n_timeout

    # Distribución de calidad
    muy_malos = sum(1 for s in scores if s < 30)
    malos = sum(1 for s in scores if 30 <= s < 50)
    moderados = sum(1 for s in scores if 50 <= s < 70)
    buenos = sum(1 for s in scores if 70 <= s < 90)
    muy_buenos = sum(1 for s in scores if s >= 90)

    # Penalizaciones más frecuentes
    todas_penalizaciones = []
    for r in resultados_analisis:
        todas_penalizaciones.extend(r["evaluacion"].get("penalizaciones", []))
    pen_counter = {}
    for p in todas_penalizaciones:
        tipo = p.get("tipo", "OTRO")
        pen_counter[tipo] = pen_counter.get(tipo, 0) + 1

    resumen = {
        "total_preguntas": total,
        "score_promedio": round(sum(scores) / len(scores), 1) if scores else 0,
        "score_mediana": round(sorted(scores)[len(scores) // 2], 1) if scores else 0,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "score_repregunta_promedio": round(sum(scores_repregunta) / len(scores_repregunta), 1) if scores_repregunta else 0,
        "n_con_repregunta": len(scores_repregunta),
        "n_fallback": n_fallback,
        "n_timeout": n_timeout,
        "n_directa": n_directa,
        "distribucion_calidad": {
            "muy_malos_0_29": muy_malos,
            "malos_30_49": malos,
            "moderados_50_69": moderados,
            "buenos_70_89": buenos,
            "muy_buenos_90_100": muy_buenos
        },
        "penalizaciones_frecuentes": pen_counter
    }

    output = {
        "metadata_evaluacion": {
            "fecha_evaluacion": "2026-03-06",
            "modelo_evaluador": "gpt-4o-mini",
            "version": "v2_post_fixes"
        },
        "resumen": resumen,
        "evaluaciones": resultados_analisis
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"RESUMEN DE EVALUACIÓN")
    print(f"{'='*60}")
    print(f"Total preguntas: {total}")
    print(f"Score promedio: {resumen['score_promedio']}/100")
    print(f"Score mediana: {resumen['score_mediana']}/100")
    print(f"Score min/max: {resumen['score_min']}/{resumen['score_max']}")
    print(f"Directas: {n_directa} | Fallbacks: {n_fallback} | Timeouts: {n_timeout}")
    print(f"\nDistribución de calidad:")
    print(f"  Muy malos (0-29):    {muy_malos}")
    print(f"  Malos (30-49):       {malos}")
    print(f"  Moderados (50-69):   {moderados}")
    print(f"  Buenos (70-89):      {buenos}")
    print(f"  Muy buenos (90-100): {muy_buenos}")
    if scores_repregunta:
        print(f"\nRepregunta score promedio: {resumen['score_repregunta_promedio']}/100")
        print(f"  Total con repregunta: {len(scores_repregunta)}")
    if pen_counter:
        print(f"\nPenalizaciones más frecuentes:")
        for tipo, count in sorted(pen_counter.items(), key=lambda x: -x[1]):
            print(f"  {tipo}: {count}")

    print(f"\nResultado guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
