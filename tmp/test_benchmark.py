import requests
import json
import time

QUESTIONS = [
    "Paviotti",
    "Martin Gill",
    "¿Cuál es el presupuesto total para Villa María en 2025?",
    "¿Qué departamento o secretaría es responsable de ejecutar este presupuesto?",
    "¿Puede el municipio cambiar el presupuesto durante el año o está fijo?",
    "¿Cuánto tengo que pagar de tasa anual por mi propiedad?",
    "¿Puedo pagar en cuotas o solo al contado?",
    "¿Califico para la tarifa social si tengo bajos ingresos?",
    "¿Cuánto cuesta mantener una tumba o panteón en el cementerio?",
    "¿Qué pasa si no pago la tasa del cementerio?",
    "¿Hay algún tributo adicional en mis servicios de agua?",
    "¿Qué tributos adicionales pago si tengo un comercio?",
    "¿Cuánto cuesta el servicio de ambulancia (SAMU) si no soy de Villa María?",
    "¿Debo pagar tributo especial si tengo camiones de transporte?",
    "¿Cuánto cuesta alquilar el Anfiteatro o espacios municipales para un evento?",
    "¿Me hacen descuento si es un evento cultural o benéfico?",
    "¿Cuánto cuesta que saquen mi auto del depósito municipal?",
    "¿Que seria MT?",
    "¿Cuánto cuesta aterrizar en el Aeropuerto de Villa María?"
]

URL = "http://localhost:8000/ask"

def run_tests():
    results = []
    print(f"🚀 Iniciando benchmark de {len(QUESTIONS)} preguntas...")
    
    for q in QUESTIONS:
        print(f"Testing: {q}...", end=" ", flush=True)
        try:
            start = time.time()
            resp = requests.post(URL, json={"pregunta": q, "conversation_id": "test_bench"}, timeout=30)
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                results.append({
                    "pregunta": q,
                    "respuesta": data.get("respuesta", ""),
                    "documentos": [d.get("numero_ordenanza") for d in data.get("documentos", [])],
                    "tiempo": round(elapsed, 2)
                })
                print("✅")
            else:
                print(f"❌ (Status {resp.status_code})")
        except Exception as e:
            print(f"⚠️ Error: {e}")
            
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Benchmark finalizado. Resultados guardados en benchmark_results.json")

if __name__ == "__main__":
    run_tests()
