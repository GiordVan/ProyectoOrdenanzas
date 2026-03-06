# Proyecto Digesto de Ordenanzas (RAG)

Este proyecto es un sistema de búsqueda y consulta de ordenanzas basado en **RAG (Retrieval-Augmented Generation)**. Permite buscar ordenanzas de forma manual y realizar consultas en lenguaje natural a un asistente de IA.

## Tecnologías Utilizadas

### 🎨 Frontend

- **React 19**: Biblioteca principal para la interfaz de usuario.
- **Vite**: Herramienta de construcción y servidor de desarrollo ultrarrápido.
- **Tailwind CSS**: Framework para el diseño y estilado responsivo.
- **Lucide React**: Set de iconos modernos y ligeros.
- **React Markdown**: Renderizado de respuestas de la IA con formato enriquecido.
- **TypeScript**: Tipado estático para mayor robustez del código.

### ⚙️ Backend

- **FastAPI**: Framework asíncrono de alto rendimiento para la API.
- **AsyncOpenAI**: Cliente asíncrono para interactuar con los modelos de OpenAI (GPT-4o-mini).
- **FAISS (CPU)**: Biblioteca para búsqueda de similitud vectorial eficiente.
- **Uvicorn**: Servidor ASGI para ejecutar la aplicación FastAPI.
- **Python-dotenv**: Gestión de variables de entorno (.env).

### 📄 Procesamiento (Scripts)

- **Python**: Lenguaje base para los scripts de extracción y procesamiento.
- **OpenAI Embeddings**: Uso del modelo `text-embedding-3-small` para vectorizar textos.
- **NLTK**: Procesamiento de lenguaje natural para segmentación de texto (chunking).
- **PyMuPDF / pdf2image**: Extracción de texto y manejo de archivos PDF.

## Estructura del Proyecto

- `/frontend`: Aplicación cliente en React.
- `/backend`: API server y motor de búsqueda.
- `/ProcesarPDFs`: Scripts para cargar ordenanzas y generar el índice vectorial.
- `/Test`: Json con 100 preguntas para testear el asistente IA y su respectivo Script.
