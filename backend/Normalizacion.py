import os
import glob

def main():
    # Busca y elimina archivos que terminen en "_2.pdf" en el directorio actual
    for archivo in glob.glob("PDFs/*_2.pdf", recursive=True):
        try:
            os.remove(archivo)
            print(f"Eliminado: {archivo}")
        except Exception as e:
            print(f"Error al eliminar {archivo}: {e}")

if __name__ == "__main__":
    print("Archivos que se eliminarán:")
    for archivo in glob.glob("PDFs/*_2.pdf", recursive=True):
        print(archivo)
    
    # Pregunta confirmación antes de borrar
    confirmacion = input("\n¿Deseas eliminar estos archivos? (s/n): ").strip().lower()
    if confirmacion == 's':
        print("\nEjecutando eliminación...")
        main()
    else:
        print("Operación cancelada.")