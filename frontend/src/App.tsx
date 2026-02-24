import { useState, lazy, Suspense } from "react";
import { Sparkles } from "lucide-react";
import pattern from "./assets/pattern.webp";
import consejo from "./assets/Consejo.webp";

const BuscadorManual = lazy(() => import("./components/BuscadorManual"));
const ChatIA = lazy(() => import("./components/ChatIA"));

function App() {
  const [chatAbierto, setChatAbierto] = useState(false);

  return (
    <div 
      className="flex flex-col min-h-[100dvh] w-full relative bg-repeat"
      style={{
        backgroundImage: `url(${pattern})`,
        backgroundSize: "auto",
        backgroundAttachment: "fixed",
      }}
    >
      {/* Imagen superior que se desvanece */}
      <div className="absolute top-16 left-0 w-full h-[45vh] md:h-[50vh] overflow-hidden z-0">
        <img
          src={consejo}
          alt="Consejo"
          className="w-full h-full object-cover mask-fade-bottom"
        />
      </div>

      {/* Header simplificado con Logo y Banner */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-4">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-[#da1062] to-[#114380] rounded-full blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
                <img 
                  src="/images/logo.webp" 
                  alt="Logo Municipalidad" 
                  className="relative w-12 h-12 object-contain bg-white rounded-full p-1 shadow-sm"
                />
              </div>
              <div className="flex flex-col">
                <h1 className="text-2xl md:text-3xl font-black tracking-tight leading-none bg-gradient-to-r from-[#da1062] to-[#114380] bg-clip-text text-transparent italic uppercase">
                  Digesto Digital
                </h1>
                <p className="text-xs uppercase tracking-[0.2em] text-[#114380] font-bold mt-1">
                  Villa María — Córdoba
                </p>
              </div>
            </div>
            
            
          </div>
        </div>
      </header>

      {/* Contenido principal: BuscadorManual */}
      <main className="flex-grow relative z-10 max-w-7xl mx-auto w-full px-4 py-8 pt-4 md:pt-4">
        <Suspense
          fallback={
            <div className="text-center py-16">
              <div className="animate-spin rounded-full h-12 w-12 border-3 border-blue-600 border-t-transparent mx-auto"></div>
              <p className="text-gray-500 mt-4">Cargando...</p>
            </div>
          }
        >
          <BuscadorManual />
        </Suspense>
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 border-t border-gray-200 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          Sistema de búsqueda de ordenanzas municipales
        </div>
      </footer>

      {/* Botón flotante "Asistente de IA" */}
      <button
        onClick={() => setChatAbierto(true)}
        className={`fixed bottom-6 right-6 z-[997] flex items-center gap-2.5 px-5 py-3.5
          bg-gradient-to-r from-[#da1062] to-[#114380] hover:brightness-110
          text-white font-medium rounded-2xl shadow-xl shadow-[#da1062]/20
          hover:shadow-2xl hover:shadow-[#da1062]/30 hover:scale-105
          active:scale-95 transition-all duration-200
          ${chatAbierto ? "opacity-0 pointer-events-none scale-90" : "opacity-100"}`}
        aria-label="Abrir Asistente de IA"
      >
        <Sparkles className="w-5 h-5" />
        <span className="hidden sm:inline">Asistente de IA</span>
      </button>

      {/* Panel de chat IA */}
      <Suspense fallback={null}>
        <ChatIA abierto={chatAbierto} onCerrar={() => setChatAbierto(false)} />
      </Suspense>
    </div>
  );
}

export default App;
