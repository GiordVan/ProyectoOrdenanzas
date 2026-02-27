import { useState, useRef, useEffect } from "react";
import { X, Send, Sparkles, FileText, ChevronDown, ChevronUp, Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";

let API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
if (API_URL && !API_URL.startsWith("http")) {
  API_URL = `https://${API_URL}`;
}
if (API_URL.endsWith("/")) {
  API_URL = API_URL.slice(0, -1);
}

interface Documento {
  nombre: string;
  numero_ordenanza: string;
  fecha_sancion: string;
  fragmento: string;
  pdf: string;
}

interface Mensaje {
  rol: "usuario" | "asistente";
  texto: string;
  documentos?: Documento[];
  cargando?: boolean;
}

interface ChatIAProps {
  abierto: boolean;
  onCerrar: () => void;
}

function ChatIA({ abierto, onCerrar }: ChatIAProps) {
  const [mensajes, setMensajes] = useState<Mensaje[]>([]);
  const [input, setInput] = useState("");
  const [enviando, setEnviando] = useState(false);
  const [docsExpandidos, setDocsExpandidos] = useState<Record<number, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll al fondo cuando hay nuevos mensajes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [mensajes]);

  // Focus en input cuando se abre
  useEffect(() => {
    if (abierto) {
      setTimeout(() => inputRef.current?.focus(), 350);
    }
  }, [abierto]);

  const abrirPDF = (nombrePDF: string) => {
    if (!nombrePDF) return;
    window.open(`/PDFs/${nombrePDF}`, "_blank");
  };

  const toggleDocs = (index: number) => {
    setDocsExpandidos((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  const enviarMensaje = async () => {
    if (!input.trim() || enviando) return;

    const pregunta = input.trim();
    setInput("");
    setEnviando(true);

    // Agregar mensaje del usuario
    setMensajes((prev) => [...prev, { rol: "usuario", texto: pregunta }]);

    // Agregar placeholder de carga para el asistente
    const idxAsistente = mensajes.length + 1;
    setMensajes((prev) => [
      ...prev,
      { rol: "asistente", texto: "", cargando: true },
    ]);

    try {
      const res = await fetch(`${API_URL}/ask-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta }),
      });

      if (!res.ok) throw new Error("Error al obtener respuesta");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("No se pudo leer la respuesta");

      let docsRecibidos: Documento[] = [];
      let respuestaTexto = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.substring(6);
            if (data === "[DONE]") break;

            try {
              const parsed = JSON.parse(data);

              if (parsed.tipo === "documentos") {
                docsRecibidos = parsed.documentos;
                // Actualizar documentos tan pronto llegan
                setMensajes((prev) => {
                  const updated = [...prev];
                  updated[idxAsistente] = {
                    ...updated[idxAsistente],
                    documentos: docsRecibidos,
                  };
                  return updated;
                });
              } else if (parsed.tipo === "chunk") {
                respuestaTexto += parsed.texto;
                // Actualizar texto progresivamente con cada chunk y quitar indicador de carga
                setMensajes((prev) => {
                  const updated = [...prev];
                  updated[idxAsistente] = {
                    ...updated[idxAsistente],
                    texto: respuestaTexto,
                    cargando: false
                  };
                  return updated;
                });
              }
            } catch {
              // Ignorar líneas malformadas
            }
          }
        }
      }

      // Actualizar el mensaje del asistente con la respuesta final
      setMensajes((prev) => {
        const updated = [...prev];
        updated[idxAsistente] = {
          rol: "asistente",
          texto: respuestaTexto || "No pude generar una respuesta.",
          documentos: docsRecibidos,
          cargando: false,
        };
        return updated;
      });
    } catch (err) {
      console.error(err);
      setMensajes((prev) => {
        const updated = [...prev];
        updated[idxAsistente] = {
          rol: "asistente",
          texto: "Hubo un problema al buscar. Intenta nuevamente.",
          cargando: false,
        };
        return updated;
      });
    } finally {
      setEnviando(false);
    }
  };

  return (
    <>
      {/* Overlay */}
      <div
        className={`fixed inset-0 bg-black/30 backdrop-blur-sm z-[998] transition-opacity duration-300 ${
          abierto ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={onCerrar}
      />

      {/* Panel deslizable */}
      <div
        className={`fixed top-0 right-0 h-full z-[999] transition-transform duration-300 ease-in-out
          w-full sm:w-[420px] md:w-[460px]
          ${abierto ? "translate-x-0" : "translate-x-full"}`}
      >
        <div className="h-full flex flex-col bg-gray-950 text-white shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#da1062] to-[#114380] flex items-center justify-center shadow-lg shadow-[#da1062]/10">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="font-semibold text-base text-white">Asistente de IA</h2>
                <p className="text-xs text-gray-400">Ordenanzas municipales</p>
              </div>
            </div>
            <button
              onClick={onCerrar}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition"
              aria-label="Cerrar chat"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Mensajes */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 chat-scrollbar">
            {mensajes.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center px-6">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#da1062]/20 to-[#114380]/20 flex items-center justify-center mb-4">
                  <Bot className="w-8 h-8 text-[#da1062]" />
                </div>
                <h3 className="text-lg font-medium text-gray-200 mb-2">
                  ¿En qué puedo ayudarte?
                </h3>
                <p className="text-sm text-gray-500 mb-6">
                  Preguntá sobre cualquier ordenanza municipal de Villa María.
                </p>
                <div className="space-y-2 w-full max-w-xs">
                  {[
                    "¿Qué dice la ordenanza 8000?",
                    "Ordenanzas sobre transporte",
                    "Normativa ambiental vigente",
                  ].map((sugerencia) => (
                    <button
                      key={sugerencia}
                      onClick={() => {
                        setInput(sugerencia);
                        setTimeout(() => inputRef.current?.focus(), 50);
                      }}
                      className="w-full text-left px-4 py-3 bg-gray-800/60 hover:bg-gray-800 border border-gray-700/50 rounded-xl text-sm text-gray-300 hover:text-white transition"
                    >
                      {sugerencia}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {mensajes.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.rol === "usuario" ? "justify-end" : "justify-start"}`}>
                {msg.rol === "asistente" && (
                  <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#da1062] to-[#114380] flex items-center justify-center flex-shrink-0 mt-1">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    msg.rol === "usuario"
                      ? "bg-[#114380] text-white rounded-br-md"
                      : "bg-gray-800/80 text-gray-100 rounded-bl-md"
                  }`}
                >
                  {msg.cargando ? (
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                      </div>
                      <span className="text-sm text-gray-400">Buscando...</span>
                    </div>
                  ) : (
                    <>
                      <div className="text-sm leading-relaxed prose prose-invert prose-p:leading-relaxed prose-pre:bg-gray-900 prose-pre:border prose-pre:border-gray-700 max-w-none">
                        <ReactMarkdown>{msg.texto}</ReactMarkdown>
                      </div>

                      {/* Documentos referenciados */}
                      {msg.documentos && msg.documentos.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-700/50">
                          <button
                            onClick={() => toggleDocs(i)}
                            className="flex items-center gap-2 text-xs text-[#da1062] hover:brightness-110 transition font-medium"
                          >
                            <FileText className="w-3.5 h-3.5" />
                            {msg.documentos.length} documento{msg.documentos.length > 1 ? "s" : ""} encontrado{msg.documentos.length > 1 ? "s" : ""}
                            {docsExpandidos[i] ? (
                              <ChevronUp className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronDown className="w-3.5 h-3.5" />
                            )}
                          </button>

                          {docsExpandidos[i] && (
                            <div className="mt-2 space-y-2">
                              {msg.documentos.map((doc, j) => (
                                <div
                                  key={j}
                                  className="bg-gray-900/60 rounded-lg p-3 border border-gray-700/30"
                                >
                                  <div className="flex items-center justify-between gap-2 mb-1">
                                    <span className="text-xs font-semibold text-gray-200 truncate">
                                      {doc.numero_ordenanza
                                        ? `Ord. N° ${doc.numero_ordenanza}`
                                        : doc.nombre}
                                    </span>
                                    {doc.pdf && (
                                      <button
                                        onClick={() => abrirPDF(doc.pdf)}
                                        className="text-[#da1062] hover:brightness-110 text-xs font-medium flex items-center gap-1 flex-shrink-0"
                                      >
                                        <FileText className="w-3 h-3" />
                                        PDF
                                      </button>
                                    )}
                                  </div>
                                  {doc.fecha_sancion && (
                                    <p className="text-xs text-gray-500 mb-1">
                                      📅 {doc.fecha_sancion}
                                    </p>
                                  )}
                                  <p className="text-xs text-gray-400 line-clamp-2">
                                    {doc.fragmento}
                                  </p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {msg.rol === "usuario" && (
                  <div className="w-7 h-7 rounded-lg bg-gray-700 flex items-center justify-center flex-shrink-0 mt-1">
                    <User className="w-4 h-4 text-gray-300" />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="px-4 py-4 border-t border-gray-800 bg-gray-900/80 backdrop-blur-sm">
            <div className="flex gap-2 items-center">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && enviarMensaje()}
                placeholder="Preguntá sobre ordenanzas..."
                className="flex-1 bg-gray-800 border border-gray-700 text-white placeholder-gray-500 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-[#da1062] focus:border-transparent transition"
                disabled={enviando}
              />
              <button
                onClick={enviarMensaje}
                disabled={enviando || !input.trim()}
                className="p-3 bg-[#da1062] hover:bg-[#c00d56] disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl transition shadow-lg shadow-[#da1062]/20 hover:shadow-[#da1062]/30"
                aria-label="Enviar mensaje"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default ChatIA;
