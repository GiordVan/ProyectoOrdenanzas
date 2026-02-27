"""
ProcesarPDFs.py - Orquestador del Pipeline de Procesamiento de Ordenanzas

Procesa PDFs uno por uno:
1. Para cada ordenanza, embebe el grupo de archivos (ej: ordenanza_8000.pdf + ordenanza_8000_2.pdf)
2. Sube el PDF escaneado (o el de mejor score como fallback) a Cloudflare R2
3. Guarda checkpoints periódicos
"""

import os
import sys
from collections import defaultdict
import re

# Agregar carpeta Scripts al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Scripts"))

from Embedder import (
    procesar_grupo_ordenanza,
    cargar_datos_existentes,
    obtener_ordenanzas_procesadas,
    guardar_checkpoint,
    extraer_numero_desde_nombre,
)
from Uploader import subir_pdf_individual

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_PDFS = os.path.join(BASE_DIR, "..", "frontend", "public", "PDFs")
CARPETA_DATA = os.path.join(BASE_DIR, "..", "backend", "Data")


def agrupar_ordenanzas_local(carpeta_pdfs):
    """Agrupa archivos PDF por número de ordenanza."""
    archivos = [f for f in os.listdir(carpeta_pdfs) if f.lower().endswith(".pdf")]
    grupos = defaultdict(list)
    for archivo in archivos:
        match = re.search(r"ordenanza[_\s]*(\d+)", archivo, re.IGNORECASE)
        if match:
            num = match.group(1)
            grupos[num].append(archivo)
    return grupos


def procesar_pipeline(
    inicio=None, fin=None, guardar_cada=10, solo_embeber=False, solo_subir=False
):
    """
    Pipeline completo: embeber + subir, ordenanza por ordenanza.

    Args:
        inicio: Número de ordenanza inicial (ej: 8000). Si None, procesa todas.
        fin: Número de ordenanza final (ej: 8100). Si None, hasta el final.
        guardar_cada: Guarda checkpoint cada N ordenanzas procesadas.
        solo_embeber: Si True, solo genera embeddings sin subir.
        solo_subir: Si True, solo sube archivos sin generar embeddings.
    """
    print("\n" + "=" * 70)
    print("🚀 PIPELINE DE PROCESAMIENTO DE ORDENANZAS")
    print("=" * 70)

    # Cargar datos existentes para continuar donde quedó
    print("\n📂 Cargando datos existentes...")
    todos_metadatos, todos_chunks, todos_embeddings = cargar_datos_existentes()
    ordenanzas_procesadas = obtener_ordenanzas_procesadas(todos_metadatos)

    if ordenanzas_procesadas:
        print(f"✓ Ya procesadas: {len(ordenanzas_procesadas)} ordenanzas")

    # Agrupar PDFs
    grupos = agrupar_ordenanzas_local(CARPETA_PDFS)
    if not grupos:
        print("⚠️ No se encontraron PDFs en la carpeta.")
        return

    # Filtrar por rango
    numeros_ordenanza = sorted([int(n) for n in grupos.keys() if n.isdigit()])

    if inicio is not None:
        numeros_ordenanza = [n for n in numeros_ordenanza if n >= inicio]
    if fin is not None:
        numeros_ordenanza = [n for n in numeros_ordenanza if n <= fin]

    # Filtrar ya procesadas (solo para embeddings)
    if not solo_subir:
        numeros_ordenanza = [
            n for n in numeros_ordenanza if str(n) not in ordenanzas_procesadas
        ]

    # Procesar de mayor a menor
    numeros_ordenanza = sorted(numeros_ordenanza, reverse=True)

    if not numeros_ordenanza:
        print("✓ Todas las ordenanzas en el rango ya están procesadas.")
        return

    print(f"\n📋 Ordenanzas a procesar: {len(numeros_ordenanza)}")
    print(
        f"🔄 Orden: De mayor a menor ({numeros_ordenanza[0]} → {numeros_ordenanza[-1]})"
    )
    if solo_embeber:
        print("📝 Modo: SOLO EMBEBER (sin subir a nube)")
    elif solo_subir:
        print("☁️ Modo: SOLO SUBIR (sin generar embeddings)")
    else:
        print("🔗 Modo: EMBEBER + SUBIR")
    print(f"💾 Checkpoint cada {guardar_cada} ordenanzas")
    print("=" * 70 + "\n")

    procesadas_en_sesion = 0
    subidas_exitosas = 0
    errores = []

    for idx, num in enumerate(numeros_ordenanza, 1):
        try:
            print(f"\n{'─'*60}")
            print(f"📦 ORDENANZA {num} ({idx}/{len(numeros_ordenanza)})")
            print(f"{'─'*60}")

            archivos = grupos[str(num)]
            print(f"  📂 Archivos: {', '.join(archivos)}")

            archivo_subir = None

            # PASO 1: Embeber (si no es modo solo_subir)
            if not solo_subir:
                m, c, e, archivo_subir = procesar_grupo_ordenanza(
                    str(num), archivos, carpeta_pdfs=CARPETA_PDFS
                )

                if m:
                    todos_metadatos.extend(m)
                    todos_chunks.extend(c)
                    todos_embeddings.extend(e)
                    procesadas_en_sesion += 1
                    print(f"  ✅ Embeddings generados: {len(c)} chunks")
                else:
                    errores.append((num, "Error al generar embeddings"))
                    continue
            else:
                # En modo solo_subir, necesitamos determinar cuál archivo subir
                # Usamos el primer archivo como fallback simple
                archivo_subir = archivos[0] if archivos else None

            # PASO 2: Subir a Cloudflare (si no es modo solo_embeber)
            if not solo_embeber and archivo_subir:
                ruta_completa = os.path.join(CARPETA_PDFS, archivo_subir)
                if subir_pdf_individual(ruta_completa):
                    subidas_exitosas += 1
                else:
                    errores.append((num, "Error al subir PDF"))

            # Guardar checkpoint periódicamente
            if (
                not solo_subir
                and procesadas_en_sesion > 0
                and procesadas_en_sesion % guardar_cada == 0
            ):
                print(
                    f"\n💾 Guardando checkpoint ({procesadas_en_sesion} procesadas)..."
                )
                guardar_checkpoint(todos_metadatos, todos_chunks, todos_embeddings)
                print(f"✓ Checkpoint guardado. Total: {len(todos_metadatos)} chunks")

        except MemoryError:
            print(f"\n⚠️ ERROR DE MEMORIA en ordenanza {num}")
            print(f"💾 Guardando progreso antes de salir...")
            if not solo_subir:
                guardar_checkpoint(todos_metadatos, todos_chunks, todos_embeddings)
            print(f"✅ Progreso guardado. Ejecuta nuevamente para continuar.")
            return

        except Exception as e:
            print(f"\n❌ ERROR en ordenanza {num}: {e}")
            errores.append((num, str(e)))
            continue

    # Guardar checkpoint final
    if not solo_subir and procesadas_en_sesion > 0:
        print(f"\n💾 Guardando datos finales...")
        guardar_checkpoint(todos_metadatos, todos_chunks, todos_embeddings)

    # Resumen
    print(f"\n{'='*70}")
    print("📊 RESUMEN DEL PROCESAMIENTO")
    print(f"{'='*70}")
    if not solo_subir:
        print(f"  ✅ Embeddings generados: {procesadas_en_sesion} ordenanzas")
        print(f"  📦 Total chunks en base: {len(todos_metadatos)}")
    if not solo_embeber:
        print(f"  ☁️ PDFs subidos a R2: {subidas_exitosas}")
    if errores:
        print(f"  ⚠️ Errores: {len(errores)}")
        for num, error in errores[:5]:
            print(f"      - Ordenanza {num}: {error}")
        if len(errores) > 5:
            print(f"      ... y {len(errores) - 5} más")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de procesamiento de ordenanzas: embeber + subir a nube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python ProcesarPDFs.py                              # Procesar todas
  python ProcesarPDFs.py --inicio 8240 --fin 8248     # Solo rango específico
  python ProcesarPDFs.py --solo-embeber               # Solo generar embeddings
  python ProcesarPDFs.py --solo-subir --inicio 8000   # Solo subir archivos
        """,
    )

    parser.add_argument("--inicio", type=int, help="Número de ordenanza inicial")
    parser.add_argument("--fin", type=int, help="Número de ordenanza final")
    parser.add_argument(
        "--guardar-cada",
        type=int,
        default=10,
        help="Guardar checkpoint cada N ordenanzas (default: 10)",
    )
    parser.add_argument(
        "--solo-embeber",
        action="store_true",
        help="Solo generar embeddings, sin subir a Cloudflare",
    )
    parser.add_argument(
        "--solo-subir",
        action="store_true",
        help="Solo subir PDFs a Cloudflare, sin generar embeddings",
    )

    args = parser.parse_args()

    if args.solo_embeber and args.solo_subir:
        print("❌ No puedes usar --solo-embeber y --solo-subir al mismo tiempo")
        sys.exit(1)

    procesar_pipeline(
        inicio=args.inicio,
        fin=args.fin,
        guardar_cada=args.guardar_cada,
        solo_embeber=args.solo_embeber,
        solo_subir=args.solo_subir,
    )
