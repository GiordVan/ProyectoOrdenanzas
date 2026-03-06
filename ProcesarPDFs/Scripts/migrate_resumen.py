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


def _construir_muestra_representativa(chunks: list[str], max_chars: int = 6000) -> str:
    """
    Para ordenanzas largas, construye una muestra representativa del texto
    tomando el inicio, fragmentos del medio y el final, para que GPT
    entienda el alcance completo de la ordenanza.
    """
    texto_total = "\n\n".join(chunks)
    if len(texto_total) <= max_chars:
        return texto_total

    n = len(chunks)
    # Estrategia: primeros 2 chunks + 1 cada N del medio + últimos 2 chunks
    seleccionados = []

    # Inicio (primeros 2)
    seleccionados.extend(chunks[:2])

    # Medio: muestreo equidistante
    if n > 6:
        paso = max(1, (n - 4) // 4)  # ~4 muestras del medio
        for i in range(2, n - 2, paso):
            seleccionados.append(chunks[i])
    elif n > 4:
        seleccionados.append(chunks[n // 2])

    # Final (últimos 2)
    if n > 2:
        seleccionados.extend(chunks[-2:])

    # Deduplicar manteniendo orden
    vistos = set()
    unicos = []
    for c in seleccionados:
        if id(c) not in vistos:
            vistos.add(id(c))
            unicos.append(c)

    texto = "\n\n[...]\n\n".join(unicos)

    # Si aún excede, truncar
    if len(texto) > max_chars:
        texto = texto[:max_chars] + "\n[...texto truncado...]"

    return texto


def generar_resumen_con_gpt(
    texto_completo: str, fecha: str, num_ord: str, chunks: list[str] | None = None
) -> str:
    """
    Genera un resumen breve (2-3 oraciones) de una ordenanza usando GPT-4o-mini.
    Para ordenanzas largas, usa muestreo representativo para cubrir todo el alcance.
    """
    # Si tenemos chunks individuales, usar muestreo representativo
    if chunks and len(chunks) > 1:
        texto_para_gpt = _construir_muestra_representativa(chunks)
    else:
        texto_para_gpt = texto_completo[:6000]

    prompt = f"""Sos un asistente legal municipal. Generá un resumen breve (2-3 oraciones) de la siguiente ordenanza municipal de Villa María, Córdoba.

REGLAS:
- El resumen debe describir el ALCANCE GENERAL de la ordenanza, no detalles específicos de un solo artículo.
- Si la ordenanza regula múltiples temas (ej: tasas, tributos, servicios), mencioná los temas principales sin entrar en montos ni porcentajes específicos.
- Si es una ordenanza tarifaria, decí que es tarifaria y mencioná qué tipos de tributos/tasas regula.
- Explicá de forma clara y concisa para que un ciudadano común pueda entenderlo.
- No incluyas montos, porcentajes ni valores numéricos específicos.
- Máximo 3 oraciones.

Ordenanza N°: {num_ord}
Fecha de sanción: {fecha}
Cantidad de artículos/secciones: ~{len(chunks) if chunks else 'desconocida'}

Texto de la ordenanza (puede estar resumido por muestreo si es muy larga):
{texto_para_gpt}

Respondé SOLO con el texto del resumen, sin comillas ni formato adicional."""

    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=250,
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
    force = "--force" in sys.argv
    if force:
        print("  Modo --force: regenerando TODOS los resumenes\n")

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
    print(
        f"  Generando resumenes para {total} ordenanzas ({ya_tienen} ya tienen resumen)...\n"
    )

    modificados = 0
    errores = 0

    for idx, meta in enumerate(metadatos_comprimidos, 1):
        num_ord = meta.get("numero_ordenanza", "?")
        fecha = meta.get("fecha_sancion", "desconocida")

        # Si ya tiene resumen, saltar (pasar --force para regenerar todos)
        if not force and meta.get("resumen") and len(meta["resumen"].strip()) > 10:
            print(f"  [{idx}/{total}] Ord. {num_ord}: ya tiene resumen, saltando")
            continue

        # Obtener TODOS los chunks de la ordenanza
        chunks_ord = []
        chunk_indices = meta.get("chunk_indices", [])
        if chunk_indices and chunks:
            # Calcular offset global del primer chunk de esta ordenanza
            global_offset = 0
            for prev_meta in metadatos_comprimidos[: idx - 1]:
                global_offset += len(prev_meta.get("chunk_indices", [0]))

            # Tomar TODOS los chunks de esta ordenanza
            for i in range(len(chunk_indices)):
                ci = global_offset + i
                if ci < len(chunks):
                    chunks_ord.append(chunks[ci])

        texto_completo = "\n\n".join(chunks_ord)

        print(
            f"  [{idx}/{total}] Ord. {num_ord} ({len(chunks_ord)} chunks)...",
            end=" ",
            flush=True,
        )

        try:
            resumen = generar_resumen_con_gpt(
                texto_completo, fecha, num_ord, chunks_ord
            )
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
