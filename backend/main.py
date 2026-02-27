import uvicorn
import os
import sys
from api import app

# Añadir el directorio actual al path para asegurar que api.py sea importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Railway inyecta la variable de entorno PORT
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Iniciando servidor en el puerto: {port}")
    # Importamos app dentro del bloque para evitar issues de importación circular si los hubiera
    uvicorn.run(app, host="0.0.0.0", port=port)
