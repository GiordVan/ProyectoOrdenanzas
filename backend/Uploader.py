import boto3
import os
import glob
from botocore.config import Config
from dotenv import load_dotenv 

env_file_path = '.env'
print(f"Verificando existencia de {env_file_path}...")
if os.path.exists(env_file_path):
    print("✅ El archivo .env existe.")
    # Carga el .env
    load_dotenv(dotenv_path=env_file_path)
else:
    print("❌ El archivo .env NO existe en esta carpeta.")

def subir_pdfs_a_r2():
    """
    Sube todos los archivos PDF de una carpeta local a un bucket R2.
    """
    # 1. Configura tus credenciales y detalles de R2
    # Es MUY recomendable usar variables de entorno para no exponer credenciales
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID") # Por ejemplo: "3d4f9e0305b21969d45642345ab0"
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID") # Tu Access Key ID
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY") # Tu Secret Access Key
    BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "ordenanzas-vm-pdfs") # Tu bucket
    CARPETA_PDFS = os.getenv("PDF_FOLDER_PATH", "PDFs") # Carpeta local con los PDFs

    print(f"R2_ACCOUNT_ID: {R2_ACCOUNT_ID}")
    print(f"R2_ACCESS_KEY_ID: {R2_ACCESS_KEY_ID}")
    print(f"R2_SECRET_ACCESS_KEY: {'***' if R2_SECRET_ACCESS_KEY else 'None'}") # Oculta parte de la clave por seguridad
    print(f"BUCKET_NAME: {BUCKET_NAME}")
    print(f"CARPETA_PDFS: {CARPETA_PDFS}")

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        raise ValueError("Faltan credenciales de R2 en las variables de entorno.")

    # 2. Configura el cliente S3 para R2
    s3_client = boto3.client(
    's3',
    endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto',
    config=Config(signature_version='s3v4') ,
    verify=False
)

    # 3. Encuentra todos los archivos PDF en la carpeta local
    ruta_pdfs = os.path.join(CARPETA_PDFS, "*.pdf")
    archivos_pdf = glob.glob(ruta_pdfs)

    if not archivos_pdf:
        print(f"No se encontraron archivos PDF en '{ruta_pdfs}'")
        return

    print(f"Se encontraron {len(archivos_pdf)} archivos PDF para subir.")

    # 4. Sube cada archivo PDF al bucket R2
    for archivo_local_path in archivos_pdf:
        # Extrae solo el nombre del archivo (ej: ordenanza_1234.pdf)
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
