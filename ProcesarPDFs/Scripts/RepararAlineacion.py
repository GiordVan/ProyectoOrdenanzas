import json
import os
import re
from collections import OrderedDict

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATA = os.path.join(BASE_DIR, "..", "..", "backend", "Data")

METADATOS_PATH = os.path.join(CARPETA_DATA, "metadatos.json")
CHUNKS_PATH = os.path.join(CARPETA_DATA, "chunks.json")


def reparar():
    print("🔍 Iniciando reparación de alineación de metadatos...")

    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(METADATOS_PATH):
        print("❌ No se encontraron los archivos necesarios.")
        return

    # 1. Cargar chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✓ Cargados {len(chunks)} chunks.")

    # 2. Cargar metadatos actuales (comprimidos)
    with open(METADATOS_PATH, "r", encoding="utf-8") as f:
        metadatos_raw = json.load(f)

    # Crear diccionario de búsqueda por número de ordenanza
    meta_lookup = {str(m["numero_ordenanza"]): m for m in metadatos_raw}
    print(f"✓ Cargados metadatos de {len(meta_lookup)} ordenanzas.")

    # 3. Identificar el orden real de las ordenanzas en chunks.json
    print("🔄 Analizando chunks para identificar orden de ordenanzas...")
    orden_real = []
    vistos = set()

    # Patrón para detectar el número de ordenanza en el chunk
    # Ejemplo: "ORDENANZA N° 8247" o "Nanaza NÂº 8247" (por el encoding roto)
    patron = r"(?:ORDENANZA|Nanaza)\s*[Nnº°Â\s]*[:.]?\s*(\d{4,5})"

    current_ord = None
    chunks_per_ord = OrderedDict()

    for idx, chunk in enumerate(chunks):
        match = re.search(patron, chunk, re.IGNORECASE)
        if match:
            num = match.group(1)
            if num != current_ord:
                if num not in vistos:
                    orden_real.append(num)
                    vistos.add(num)
                    chunks_per_ord[num] = []
                current_ord = num

        if current_ord:
            chunks_per_ord[current_ord].append(idx)

    print(f"✓ Detectado orden de {len(orden_real)} ordenanzas en los chunks.")

    # 4. Reconstruir metadatos en el orden detectado
    metadatos_nuevos = []

    for num in orden_real:
        if num in meta_lookup:
            meta = meta_lookup[num].copy()
            # Actualizar los índices de los chunks para esta ordenanza
            meta["chunk_indices"] = [
                i for i, idx_global in enumerate(chunks_per_ord[num])
            ]
            meta["total_chunks"] = len(chunks_per_ord[num])
            metadatos_nuevos.append(meta)
        else:
            print(
                f"⚠️ Alerta: No se encontraron metadatos para la Ordenanza {num} detectada en chunks."
            )

    # 5. Guardar copia de seguridad y el nuevo archivo
    backup_path = METADATOS_PATH + ".bak"
    os.rename(METADATOS_PATH, backup_path)
    print(f"✓ Backup creado en {backup_path}")

    with open(METADATOS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatos_nuevos, f, ensure_ascii=False, indent=2)

    print(f"✅ REPARACIÓN COMPLETADA. {len(metadatos_nuevos)} ordenanzas re-alineadas.")
    print("📌 Por favor, reinicia el servidor de backend para aplicar los cambios.")


if __name__ == "__main__":
    reparar()
