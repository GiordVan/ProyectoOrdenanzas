import boto3
import os
import glob
from botocore.config import Config
from dotenv import load_dotenv

# Cargar .env
env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_file_path):
    load_dotenv(dotenv_path=env_file_path)
else:
    # Fallback: buscar en directorio actual
    if os.path.exists(".env"):
        load_dotenv(dotenv_path=".env")


def crear_cliente_r2():
    """
    Crea y retorna un cliente S3 configurado para Cloudflare R2.
    Returns: (s3_client, bucket_name) o (None, None) si faltan credenciales.
    """
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "ordenanzas-vm-pdfs")

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        print("❌ Faltan credenciales de R2 en las variables de entorno.")
        return None, None

    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
        verify=False,
    )

    return s3_client, BUCKET_NAME


def subir_pdf_individual(ruta_archivo: str, bucket_name: str = None) -> bool:
    """
    Sube un archivo PDF específico a Cloudflare R2.

    Args:
        ruta_archivo: Ruta absoluta o relativa al archivo PDF.
        bucket_name: Nombre del bucket (opcional, usa variable de entorno si no se especifica).

    Returns:
        True si la subida fue exitosa, False en caso contrario.
    """
    if not os.path.exists(ruta_archivo):
        print(f"❌ Archivo no encontrado: {ruta_archivo}")
        return False

    s3_client, default_bucket = crear_cliente_r2()
    if not s3_client:
        return False

    bucket = bucket_name or default_bucket
    nombre_objeto = os.path.basename(ruta_archivo)

    try:
        print(f"  📤 Subiendo {nombre_objeto} a R2...", end=" ")
        s3_client.upload_file(ruta_archivo, bucket, nombre_objeto)
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def subir_pdfs_a_r2():
    """
    Sube todos los archivos PDF de una carpeta local a un bucket R2.
    """
    s3_client, BUCKET_NAME = crear_cliente_r2()
    if not s3_client:
        raise ValueError("Faltan credenciales de R2 en las variables de entorno.")

    CARPETA_PDFS = os.getenv("PDF_FOLDER_PATH", "PDFs")

    print(f"BUCKET_NAME: {BUCKET_NAME}")
    print(f"CARPETA_PDFS: {CARPETA_PDFS}")

    # Encuentra todos los archivos PDF en la carpeta local
    ruta_pdfs = os.path.join(CARPETA_PDFS, "*.pdf")
    archivos_pdf = glob.glob(ruta_pdfs)

    if not archivos_pdf:
        print(f"No se encontraron archivos PDF en '{ruta_pdfs}'")
        return

    print(f"Se encontraron {len(archivos_pdf)} archivos PDF para subir.")

    # Sube cada archivo PDF al bucket R2
    for archivo_local_path in archivos_pdf:
        nombre_objeto = os.path.basename(archivo_local_path)
        print(f"Subiendo {archivo_local_path} -> {nombre_objeto} ...", end=" ")
        try:
            s3_client.upload_file(archivo_local_path, BUCKET_NAME, nombre_objeto)
            print("✅ OK")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("Subida de PDFs completada.")


if __name__ == "__main__":
    subir_pdfs_a_r2()
