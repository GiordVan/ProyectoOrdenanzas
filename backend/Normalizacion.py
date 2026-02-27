import os
import glob


def main():
    # Busca y elimina archivos que terminen en "_2.pdf" en el directorio backend/PDFs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdfs_dir = os.path.join(base_dir, "PDFs")
    for archivo in glob.glob(os.path.join(pdfs_dir, "*_2.pdf"), recursive=True):
        try:
            os.remove(archivo)
            print(f"Eliminado: {archivo}")
        except Exception as e:
            print(f"Error al eliminar {archivo}: {e}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdfs_dir = os.path.join(base_dir, "PDFs")

    print(f"Buscando archivos en: {pdfs_dir}")
    archivos_para_eliminar = glob.glob(
        os.path.join(pdfs_dir, "*_2.pdf"), recursive=True
    )

    if not archivos_para_eliminar:
        print("No se encontraron archivos con terminación _2.pdf")
    else:
        print("Archivos que se eliminarán:")
        for archivo in archivos_para_eliminar:
            print(archivo)

        # Pregunta confirmación antes de borrar
        confirmacion = (
            input("\n¿Deseas eliminar estos archivos? (s/n): ").strip().lower()
        )
        if confirmacion == "s":
            print("\nEjecutando eliminación...")
            main()
        else:
            print("Operación cancelada.")
