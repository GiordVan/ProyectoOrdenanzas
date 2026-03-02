"""
migrate_etiquetas.py — Agrega etiquetas semánticas a metadatos existentes sin re-embeber.

Uso:
    python Scripts/migrate_etiquetas.py

Lee el metadatos.json existente (formato comprimido), clasifica cada ordenanza
con GPT-4o-mini usando el Art. 1 y la fecha ya guardados, y guarda de vuelta.

NO toca index.faiss ni chunks.json.
"""

import json
import os
import sys
import time
from dotenv import load_dotenv
import openai

# Agregar directorio Scripts al path para importar el clasificador
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Embedder import clasificar_ordenanza_con_gpt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CARPETA_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend", "Data"
)


def main():
    metadatos_path = os.path.join(CARPETA_DATA, "metadatos.json")
    chunks_path = os.path.join(CARPETA_DATA, "chunks.json")

    if not os.path.exists(metadatos_path):
        print(f"❌ No se encontró {metadatos_path}")
        sys.exit(1)

    # === Cargar metadatos comprimidos ===
    with open(metadatos_path, "r", encoding="utf-8") as f:
        metadatos_comprimidos = json.load(f)

    # === Cargar chunks para tener texto disponible ===
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✓ {len(chunks)} chunks cargados")

    total = len(metadatos_comprimidos)
    print(f"🏷️  Clasificando {total} ordenanzas con GPT-4o-mini...\n")

    modificados = 0
    errores = 0

    for idx, meta in enumerate(metadatos_comprimidos, 1):
        num_ord = meta.get("numero_ordenanza", "?")
        art1 = meta.get("Art N°1", "")
        fecha = meta.get("fecha_sancion", "desconocida")

        # Si ya tiene etiquetas, saltar
        if meta.get("etiquetas") and len(meta["etiquetas"]) > 0:
            print(f"  [{idx}/{total}] Ord. {num_ord}: ya tiene etiquetas, saltando")
            continue

        # Obtener un fragmento de texto del primer chunk de esta ordenanza
        texto_fragmento = ""
        chunk_indices = meta.get("chunk_indices", [])
        if chunk_indices and chunks:
            # Calcular offset global del primer chunk de esta ordenanza
            global_offset = 0
            for prev_meta in metadatos_comprimidos[: idx - 1]:
                global_offset += len(prev_meta.get("chunk_indices", [0]))
            if global_offset < len(chunks):
                texto_fragmento = chunks[global_offset]

        print(f"  [{idx}/{total}] Ord. {num_ord}...", end=" ", flush=True)

        try:
            etiquetas = clasificar_ordenanza_con_gpt(texto_fragmento, art1, fecha)
            meta["etiquetas"] = etiquetas
            modificados += 1
        except Exception as e:
            print(f"⚠️ Error: {e}")
            meta["etiquetas"] = []
            errores += 1

        # Rate limit: pequeña pausa para no saturar la API
        if idx % 20 == 0:
            time.sleep(1)

    # === Guardar metadatos actualizados ===
    backup_path = metadatos_path + ".backup"
    if not os.path.exists(backup_path):
        import shutil

        shutil.copy2(metadatos_path, backup_path)
        print(f"\n💾 Backup guardado en {backup_path}")

    with open(metadatos_path, "w", encoding="utf-8") as f:
        json.dump(metadatos_comprimidos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Migración completada:")
    print(f"   Ordenanzas clasificadas: {modificados}")
    print(f"   Errores: {errores}")
    print(f"   Total: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
