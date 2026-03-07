import json
import requests
import uuid
import time

# ==========================
# CONFIG
# ==========================

API_URL = "http://127.0.0.1:8000/ask"
INPUT_FILE = "PreguntasTest.json"
OUTPUT_FILE = "RespuestasTest.json"

TIMEOUT = 60


# ==========================
# FUNCIONES
# ==========================


def cargar_preguntas():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def hacer_pregunta(pregunta, conversation_id):
    """
    Realiza la request al endpoint /ask.
    """

    payload = {"pregunta": pregunta, "conversation_id": conversation_id}

    try:
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return {
                "respuesta": data.get("respuesta", "").strip(),
                "escalado": bool(data.get("escalado", False)),
                "modelo_usado": data.get("modelo_usado"),
            }
        else:
            return {
                "respuesta": f"ERROR {response.status_code}: {response.text}",
                "escalado": False,
                "modelo_usado": None,
            }

    except Exception as e:
        return {
            "respuesta": f"ERROR EXCEPCION: {str(e)}",
            "escalado": False,
            "modelo_usado": None,
        }


def detecta_repregunta(respuesta):
    """
    Heurística simple para detectar si el modelo repregunta.
    Luego yo haré el análisis fino.
    """

    signos_interrogacion = respuesta.count("?")

    frases_clave = [
        "podrías especificar",
        "podria especificar",
        "a qué",
        "cuál específicamente",
        "necesito más información",
        "podrías aclarar",
        "puede indicar",
        "en qué zona",
        "qué tipo",
        "qué tasa",
        "qué tributo",
    ]

    repregunta_detectada = False
    texto_repregunta = None

    if signos_interrogacion > 0:
        for frase in frases_clave:
            if frase.lower() in respuesta.lower():
                repregunta_detectada = True
                texto_repregunta = respuesta
                break

    return repregunta_detectada, texto_repregunta


def evaluar_placeholder(respuesta):
    """
    Score preliminar automático MUY BÁSICO.
    Luego yo haré evaluación real sobre grounding.
    """

    if respuesta.startswith("ERROR"):
        return 0

    if len(respuesta) < 20:
        return 20

    if "no tengo información" in respuesta.lower():
        return 30

    return 60  # Score base neutro que luego ajustaremos manualmente


# ==========================
# MAIN
# ==========================


def main():
    data = cargar_preguntas()
    preguntas = data["preguntas"]

    resultados = []

    for item in preguntas:

        print(f"Procesando pregunta {item['id']}...")

        # Usar el mismo conversation_id para pregunta + repregunta
        conversation_id = str(uuid.uuid4())

        resultado_pregunta = hacer_pregunta(item["pregunta"], conversation_id)
        respuesta = resultado_pregunta["respuesta"]

        repregunto, texto_repregunta = detecta_repregunta(respuesta)

        # Si la pregunta tiene repregunta definida, enviarla con el MISMO conversation_id
        respuesta_repregunta = None
        resultado_repregunta = None
        if item.get("requiere_repregunta") and item.get("repregunta"):
            print(f"  → Enviando repregunta: {item['repregunta'][:60]}...")
            time.sleep(0.5)  # Pausa breve antes de la repregunta
            resultado_repregunta = hacer_pregunta(
                item["repregunta"], conversation_id
            )
            respuesta_repregunta = resultado_repregunta["respuesta"]

        resultado = {
            "id": item["id"],
            "pregunta": item["pregunta"],
            "categoria": item["categoria"],
            "ordenanzas_referenciadas": item["ordenanzas_referenciadas"],
            "dificultad": item.get("dificultad", "media"),
            "requiere_repregunta_según_test": item.get("requiere_repregunta", False),
            "motivo_ambiguedad": item.get("motivo_ambiguedad"),
            "tipo_test": item.get("tipo_test", "grounding"),
            "respuesta_modelo": respuesta,
            "escalado": resultado_pregunta["escalado"],
            "modelo_usado": resultado_pregunta["modelo_usado"],
            "repregunto": repregunto,
            "texto_repregunta": texto_repregunta,
            # Campos de repregunta/follow-up
            "repregunta_enviada": item.get("repregunta"),
            "respuesta_repregunta": respuesta_repregunta,
            "escalado_repregunta": (
                resultado_repregunta["escalado"] if resultado_repregunta else None
            ),
            "modelo_usado_repregunta": (
                resultado_repregunta["modelo_usado"] if resultado_repregunta else None
            ),
            "hubo_escalado": resultado_pregunta["escalado"]
            or bool(resultado_repregunta and resultado_repregunta["escalado"]),
            "calidad_respuesta_repregunta": (
                evaluar_placeholder(respuesta_repregunta)
                if respuesta_repregunta
                else None
            ),
            # Placeholder – luego lo ajustamos manualmente
            "calidad_respuesta": evaluar_placeholder(respuesta),
        }

        resultados.append(resultado)

        # pequeña pausa para no saturar el endpoint
        time.sleep(1)

    output = {"metadata_test": data["metadata"], "resultados": resultados}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    # Resumen final
    total = len(resultados)
    con_repregunta = sum(1 for r in resultados if r["respuesta_repregunta"])
    escaladas = sum(1 for r in resultados if r.get("escalado"))
    escaladas_repregunta = sum(1 for r in resultados if r.get("escalado_repregunta"))
    items_con_alguna_escalada = sum(1 for r in resultados if r.get("hubo_escalado"))
    print(f"\nEvaluación terminada.")
    print(f"  Total preguntas: {total}")
    print(f"  Con repregunta enviada: {con_repregunta}")
    print(f"  Escaladas en pregunta principal: {escaladas}")
    print(f"  Escaladas en repregunta: {escaladas_repregunta}")
    print(f"  Items con alguna escalada: {items_con_alguna_escalada}")
    print(f"  Archivo generado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
