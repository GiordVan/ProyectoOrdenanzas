import { useState, lazy, Suspense } from "react";
import { Sparkles, Filter } from "lucide-react";
import pattern from "./assets/pattern.webp";
import consejo from "./assets/Consejo.webp";

const BuscadorIA = lazy(() => import("./components/BuscadorIA"));
const BuscadorManual = lazy(() => import("./components/BuscadorManual"));

// Componente de pestañas
function TabButton({ active, onClick, icon: Icon, children }: any) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-3 font-medium transition-all relative ${
        active
          ? "text-blue-600"
          : "text-gray-500 hover:text-gray-700"
      }`}
    >
      <Icon className="w-5 h-5" />
      {children}
      {active && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 rounded-full" />
      )}
    </button>
  );
}

function App() {
  const [tabActiva, setTabActiva] = useState("ia");

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
      <div className="absolute top-0 left-0 w-full h-[45vh] md:h-[50vh] overflow-hidden z-0">
        <img
          src={consejo}
          alt="Consejo"
          className="w-full h-full object-cover mask-fade-bottom"
        />
      </div>
      

      {/* Header o Navegación por pestañas */}
      <header className="sticky top-0 relative z-50 bg-white/80 backdrop-blur-sm border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-center gap-4">
            <TabButton
              active={tabActiva === "ia"}
              onClick={() => setTabActiva("ia")}
              icon={Sparkles}
            >
              Búsqueda Asistida por IA
            </TabButton>
            <TabButton
              active={tabActiva === "manual"}
              onClick={() => setTabActiva("manual")}
              icon={Filter}
            >
              Búsqueda Manual por Filtros
            </TabButton>
          </div>
        </div>
      </header>

      {/* Contenido principal envuelto en un flex-grow */}
      <main className="flex-grow relative z-10 max-w-7xl mx-auto w-full px-4 py-8 pt-4 md:pt-4">
        <Suspense
          fallback={
            <div className="text-center py-16">
              <div className="animate-spin rounded-full h-12 w-12 border-3 border-blue-600 border-t-transparent mx-auto"></div>
              <p className="text-gray-500 mt-4">Cargando...</p>
            </div>
          }
        >
          {tabActiva === "ia" ? <BuscadorIA /> : <BuscadorManual />}
        </Suspense>
      </main>

      {/* Footer fijo al final */}
      <footer className="relative z-10 py-6 border-t border-gray-200 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          Sistema de búsqueda de ordenanzas municipales
        </div>
      </footer>
    </div>
  );
}

export default App;
