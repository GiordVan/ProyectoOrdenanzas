"""
migrate_resumen.py — Agrega resúmenes generados por GPT a metadatos existentes sin re-embeber.

Uso:
    python Scripts/migrate_resumen.py

Lee el metadatos.json existente (formato comprimido), genera un resumen breve
con GPT-4o-mini usando el Art. 1 y los primeros chunks, y guarda de vuelta.

NO toca index.faiss ni chunks.json.
"""

import json
import os
import sys
import time
from dotenv import load_dotenv
import openai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CARPETA_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend", "Data"
)


def generar_resumen_con_gpt(texto_fragmento: str, art1: str, fecha: str, num_ord: str) -> str:
    """
    Genera un resumen breve (2-3 oraciones) de una ordenanza usando GPT-4o-mini.
    """
    prompt = f"""Sos un asistente legal municipal. Generá un resumen breve (2-3 oraciones) de la siguiente ordenanza municipal de Villa María, Córdoba.

El resumen debe explicar de forma clara y concisa qué establece la ordenanza, para que un ciudadano común pueda entenderlo rápidamente. No uses lenguaje técnico innecesario.

Ordenanza N°: {num_ord}
Fecha de sanción: {fecha}
Artículo 1°: {art1[:500] if art1 else "(no disponible)"}
Fragmento del texto: {texto_fragmento[:1500]}

Respondé SOLO con el texto del resumen, sin comillas ni formato adicional."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        resumen = response.choices[0].message.content.strip()
        # Limpiar comillas envolventes si las hay
        if resumen.startswith('"') and resumen.endswith('"'):
            resumen = resumen[1:-1]
        print(f"OK")
        return resumen
    except Exception as e:
        print(f"Error: {e}")
        return ""


def main():
    metadatos_path = os.path.join(CARPETA_DATA, "metadatos.json")
    chunks_path = os.path.join(CARPETA_DATA, "chunks.json")

    if not os.path.exists(metadatos_path):
        print(f"No se encontro {metadatos_path}")
        sys.exit(1)

    # === Cargar metadatos comprimidos ===
    with open(metadatos_path, "r", encoding="utf-8") as f:
        metadatos_comprimidos = json.load(f)

    # === Cargar chunks para tener texto disponible ===
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"  {len(chunks)} chunks cargados")

    total = len(metadatos_comprimidos)
    ya_tienen = sum(1 for m in metadatos_comprimidos if m.get("resumen"))
    print(f"  Generando resumenes para {total} ordenanzas ({ya_tienen} ya tienen resumen)...\n")

    modificados = 0
    errores = 0

    for idx, meta in enumerate(metadatos_comprimidos, 1):
        num_ord = meta.get("numero_ordenanza", "?")
        art1 = meta.get("Art N\u00b01", "")
        fecha = meta.get("fecha_sancion", "desconocida")

        # Si ya tiene resumen, saltar
        if meta.get("resumen") and len(meta["resumen"].strip()) > 10:
            print(f"  [{idx}/{total}] Ord. {num_ord}: ya tiene resumen, saltando")
            continue

        # Obtener fragmento de texto del primer chunk de esta ordenanza
        texto_fragmento = ""
        chunk_indices = meta.get("chunk_indices", [])
        if chunk_indices and chunks:
            # Calcular offset global del primer chunk de esta ordenanza
            global_offset = 0
            for prev_meta in metadatos_comprimidos[: idx - 1]:
                global_offset += len(prev_meta.get("chunk_indices", [0]))

            # Tomar primer chunk y segundo si existe (para mas contexto)
            fragmentos = []
            for i in range(min(2, len(chunk_indices))):
                ci = global_offset + i
                if ci < len(chunks):
                    fragmentos.append(chunks[ci])
            texto_fragmento = " ".join(fragmentos)

        print(f"  [{idx}/{total}] Ord. {num_ord}...", end=" ", flush=True)

        try:
            resumen = generar_resumen_con_gpt(texto_fragmento, art1, fecha, num_ord)
            if resumen:
                meta["resumen"] = resumen
                modificados += 1
            else:
                errores += 1
        except Exception as e:
            print(f"Error: {e}")
            errores += 1

        # Rate limit: pequena pausa para no saturar la API
        if idx % 20 == 0:
            time.sleep(1)

    # === Guardar metadatos actualizados ===
    backup_path = metadatos_path + ".backup_resumen"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(metadatos_path, backup_path)
        print(f"\n  Backup guardado en {backup_path}")

    with open(metadatos_path, "w", encoding="utf-8") as f:
        json.dump(metadatos_comprimidos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Migracion completada:")
    print(f"   Resumenes generados: {modificados}")
    print(f"   Errores: {errores}")
    print(f"   Total: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
