import { useState } from "react";
import { Sparkles, FileText } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function BuscadorIA() {
  const [busqueda, setBusqueda] = useState("");
  const [respuesta, setRespuesta] = useState<string | null>(null);
  const [documentos, setDocumentos] = useState<any[]>([]);
  const [mostrarDocs, setMostrarDocs] = useState(false);
  const [cargando, setCargando] = useState(false);
  const [cargandoRespuesta, setCargandoRespuesta] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abrirPDF = (nombrePDF: string) => {
    if (!nombrePDF) return;
    const pdfConSufijo = nombrePDF.replace(".pdf", "_2.pdf");
    window.open(`${API_URL}/pdfs/${pdfConSufijo}`, "_blank");
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBusqueda(e.target.value);
  };

  const handleBuscarStream = async () => {
    if (!busqueda.trim()) return;
    setCargando(true);
    setCargandoRespuesta(true);
    setRespuesta(null);
    setDocumentos([]);
    setMostrarDocs(false);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/ask-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta: busqueda }),
      });

      if (!res.ok) throw new Error("Error al obtener respuesta");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No se pudo leer la respuesta");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.substring(6);
            if (data === "[DONE]") {
              setCargandoRespuesta(false);
              setCargando(false);
              return;
            }

            try {
              const parsed = JSON.parse(data);

              if (parsed.tipo === "documentos") {
                setDocumentos(parsed.documentos);
                setCargando(false);
              } else if (parsed.tipo === "respuesta") {
                setRespuesta(parsed.respuesta);
                setCargandoRespuesta(false);
              }
            } catch (e) {
              // Ignorar líneas malformadas
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setError("Hubo un problema al buscar. Intenta nuevamente.");
    } finally {
      setCargando(false);
      setCargandoRespuesta(false);
    }
  };

  const handleBuscar = handleBuscarStream;

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Barra de búsqueda */}
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg p-2 mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={busqueda}
            onChange={handleChange}
            onKeyDown={(e) => e.key === "Enter" && handleBuscar()}
            placeholder="Pregunta sobre las ordenanzas municipales..."
            className="flex-1 px-6 py-4 bg-transparent text-gray-800 placeholder-gray-400 focus:outline-none text-lg"
            disabled={cargando || cargandoRespuesta}
          />
          <button
            onClick={handleBuscar}
            className="px-8 py-4 bg-[#da1062] text-white rounded-xl hover:brightness-110 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-md hover:shadow-lg shadow-[#da1062]/20"
            disabled={cargando || cargandoRespuesta}
          >
            {cargando ? "Buscando..." : "Buscar"}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 text-red-800">
          {error}
        </div>
      )}

      {/* Documentos encontrados (antes de respuesta) */}
      {documentos.length > 0 && !respuesta && (
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="font-semibold mb-2 text-gray-800">
            📄 Documentos encontrados
          </h2>
          <p className="text-sm text-gray-500">Generando respuesta...</p>
        </div>
      )}

      {/* Indicador de carga */}
      {cargandoRespuesta && !respuesta && (
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-[#da1062] border-t-transparent"></div>
            <p className="text-gray-600">Generando respuesta con IA...</p>
          </div>
        </div>
      )}

      {/* Respuesta */}
      {respuesta && (
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg p-6 mb-6">
          <h2 className="font-semibold mb-4 text-gray-800 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#da1062]" />
            Respuesta
          </h2>
          <p className="text-gray-700 leading-relaxed mb-4 whitespace-pre-wrap">
            {respuesta}
          </p>

          {documentos.length > 0 && (
            <button
              onClick={() => setMostrarDocs(!mostrarDocs)}
              className="px-4 py-2 bg-pink-50 text-[#da1062] rounded-lg hover:bg-pink-100 transition font-medium"
            >
              {mostrarDocs
                ? "Ocultar documentos"
                : `Ver ${documentos.length} documentos`}
            </button>
          )}
        </div>
      )}

      {/* Lista de documentos */}
      {mostrarDocs && documentos.length > 0 && (
        <div className="space-y-3">
          {documentos.map((doc, i) => (
            <div
              key={i}
              className="bg-white/95 backdrop-blur-sm rounded-xl shadow p-5 hover:shadow-md transition"
            >
              <div className="flex justify-between items-start mb-3">
                <div>
                  <p className="font-semibold text-gray-800">
                    {doc.numero_ordenanza
                      ? `Ordenanza N° ${doc.numero_ordenanza}`
                      : doc.nombre}
                  </p>
                  {doc.fecha_sancion && (
                    <p className="text-xs text-gray-500 mt-1">
                      📅 {doc.fecha_sancion}
                    </p>
                  )}
                </div>
                {doc.pdf && (
                  <button
                    onClick={() => abrirPDF(doc.pdf)}
                    className="text-[#da1062] hover:brightness-110 text-sm font-medium flex items-center gap-1"
                  >
                    <FileText className="w-4 h-4" />
                    PDF
                  </button>
                )}
              </div>
              <p className="text-sm text-gray-600 leading-relaxed">
                {doc.fragmento}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default BuscadorIA;
