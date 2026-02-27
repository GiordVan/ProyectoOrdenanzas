import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useDeferredValue,
  useMemo,
} from "react";
import {
  Filter,
  X,
  FileText,
  Calendar,
  AlertCircle,
  MoveUp,
  ChevronDown,
  ChevronUp
} from "lucide-react";

interface Metadato {
  nombre_archivo: string;
  numero_ordenanza: string;
  tipo_norma: string;
  fecha_sancion: string;
  fecha_sancion_iso: string;
  estado_vigencia: string;
  categoria: string;
  descripcion_categoria: string;
  temas: string[];
  palabras_clave: string[];
  autoridad_emisora: string;
  municipio: string;
  provincia: string;
  pais: string;
  "Art N°1": string;
  total_chunks?: number;
  chunk_indices?: number[];
}

let API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
// Asegurar que la URL tenga protocolo (evita que se trate como ruta relativa en Netlify/Railway)
if (API_URL && !API_URL.startsWith("http")) {
  API_URL = `https://${API_URL}`;
}
// Eliminar barra final si existe
if (API_URL.endsWith("/")) {
  API_URL = API_URL.slice(0, -1);
}
const ITEMS_POR_PAGINA = 20; // Renderizar 20 ordenanzas por vez

function BuscadorManual() {
  const [, setMetadatos] = useState<Metadato[]>([]);
  const [ordenanzas, setOrdenanzas] = useState<Metadato[]>([]);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [numeroOrdenanza, setNumeroOrdenanza] = useState("");
  const [fechaDesde, setFechaDesde] = useState("");
  const [fechaHasta, setFechaHasta] = useState("");
  const [categoriaSeleccionada, setCategoriaSeleccionada] = useState("");
  const [estadoVigencia, setEstadoVigencia] = useState("");
  const [palabraClave, setPalabraClave] = useState("");

  // ⚡ Debounce para filtros de texto - evita filtrar en cada tecla
  const deferredNumero = useDeferredValue(numeroOrdenanza);
  const deferredPalabraClave = useDeferredValue(palabraClave);

  const [totalOrdenanzas, setTotalOrdenanzas] = useState(0);
  const [usarPaginacion, setUsarPaginacion] = useState(false);

  // ⚡ Estados para lazy loading
  const [itemsVisibles, setItemsVisibles] = useState(ITEMS_POR_PAGINA);
  const [cargandoMas, setCargandoMas] = useState(false);
  const observerTarget = useRef<HTMLDivElement>(null);

  // 🚀 Estado para mostrar/ocultar botón de scroll to top
  const [mostrarScrollTop, setMostrarScrollTop] = useState(false);
  const [filtrosAbiertos, setFiltrosAbiertos] = useState(false); // Nuevo estado para colapsable en móvil
  const [expandida, setExpandida] = useState<string | null>(null); // Nuevo estado para expandir cards en móvil

  const cargarMetadatos = useCallback(async () => {
    try {
      setCargando(true);
      setError(null);

      const response = await fetch(`${API_URL}/metadatos`, {
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(
          `Error HTTP: ${response.status} ${response.statusText}`
        );
      }

      const contentLength = response.headers.get("content-length");
      const sizeInMB = contentLength
        ? parseInt(contentLength) / (1024 * 1024)
        : 0;

      console.log(`📦 Tamaño del archivo: ${sizeInMB.toFixed(2)} MB`);

      if (sizeInMB > 10) {
        console.warn("⚠️ Archivo grande detectado, activando modo paginación");
        setUsarPaginacion(true);
        await cargarConPaginacion();
        return;
      }

      const data: Metadato[] = await response.json();

      const ordenanzasUnicas = data.some((m: Metadato) => "chunk_id" in m)
        ? data.filter((m: Metadato) => (m as Metadato & { chunk_id?: number }).chunk_id === 0)
        : data;

      // Los datos ya vienen ordenados del backend (descendente por número)
      setMetadatos(data);
      setOrdenanzas(ordenanzasUnicas);
      setTotalOrdenanzas(ordenanzasUnicas.length);

      console.log(
        `✅ Cargadas ${ordenanzasUnicas.length} ordenanzas desde API`
      );
    } catch (error) {
      console.error("❌ Error al cargar metadatos:", error);
      setError(
        error instanceof Error
          ? `Error: ${error.message}. Verifica que el backend esté corriendo en ${API_URL}`
          : "Error desconocido al cargar datos"
      );
      setOrdenanzas([]);
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => {
    const timeout = setTimeout(() => {
      if ("requestIdleCallback" in window) {
        (window as unknown as { requestIdleCallback: (cb: () => void) => void }).requestIdleCallback(cargarMetadatos);
      } else {
        cargarMetadatos();
      }
    }, 1000);

    return () => clearTimeout(timeout);
  }, [cargarMetadatos]);

  const cargarConPaginacion = async () => {
    try {
      const response = await fetch(
        `${API_URL}/metadatos-paginado?pagina=1&por_pagina=100`
      );
      const data = await response.json();

      setTotalOrdenanzas(data.total);
      setOrdenanzas(data.metadatos);

      console.log(
        `✅ Modo paginación: ${data.metadatos.length} de ${data.total} ordenanzas cargadas`
      );
    } catch (error) {
      console.error("❌ Error en carga paginada:", error);
      throw error;
    }
  };

  const filtradas = useMemo(() => {
    let resultados = [...ordenanzas];

    if (deferredNumero.trim()) {
      resultados = resultados.filter((ord) =>
        ord.numero_ordenanza.includes(deferredNumero.trim())
      );
    }

    if (fechaDesde) {
      resultados = resultados.filter(
        (ord) => ord.fecha_sancion_iso >= fechaDesde
      );
    }

    if (fechaHasta) {
      resultados = resultados.filter(
        (ord) => ord.fecha_sancion_iso <= fechaHasta
      );
    }

    if (categoriaSeleccionada) {
      resultados = resultados.filter(
        (ord) => ord.descripcion_categoria === categoriaSeleccionada
      );
    }

    if (estadoVigencia) {
      resultados = resultados.filter(
        (ord) => ord.estado_vigencia === estadoVigencia
      );
    }

    if (deferredPalabraClave.trim()) {
      const palabraLower = deferredPalabraClave.toLowerCase();
      resultados = resultados.filter((ord) => {
        const temas = ord.temas.join(" ").toLowerCase();
        const palabras = ord.palabras_clave.join(" ").toLowerCase();
        const art1 = ord["Art N°1"]?.toLowerCase() || "";
        return (
          temas.includes(palabraLower) ||
          palabras.includes(palabraLower) ||
          art1.includes(palabraLower)
        );
      });
    }

    return resultados;
  }, [
    ordenanzas,
    deferredNumero,
    fechaDesde,
    fechaHasta,
    categoriaSeleccionada,
    estadoVigencia,
    deferredPalabraClave,
  ]);

  // Reset items visibles cuando cambian los filtros
  useEffect(() => {
    setItemsVisibles(ITEMS_POR_PAGINA);
  }, [filtradas]);

  const limpiarFiltros = () => {
    setNumeroOrdenanza("");
    setFechaDesde("");
    setFechaHasta("");
    setCategoriaSeleccionada("");
    setEstadoVigencia("");
    setPalabraClave("");
    // setItemsVisibles is resetted by the useEffect above
  };

  const abrirPDF = (nombrePDF: string) => {
    if (!nombrePDF) return;
    window.open(`/PDFs/${nombrePDF}`, "_blank");
  };

  const categoriasUnicas: string[] = Array.from(
    new Set(ordenanzas.map((o: Metadato) => o.descripcion_categoria))
  ).sort() as string[];

  // ⚡ Función para cargar más items
  const cargarMasItems = useCallback(() => {
    if (itemsVisibles >= filtradas.length) return;

    setCargandoMas(true);
    setTimeout(() => {
      setItemsVisibles((prev: number) =>
        Math.min(prev + ITEMS_POR_PAGINA, filtradas.length)
      );
      setCargandoMas(false);
    }, 300); // Pequeño delay para mejor UX
  }, [itemsVisibles, filtradas.length]);

  // ⚡ Intersection Observer para detectar cuando llegas al final
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (
          entries[0].isIntersecting &&
          !cargandoMas &&
          itemsVisibles < filtradas.length
        ) {
          cargarMasItems();
        }
      },
      { threshold: 0.1 }
    );

    const currentTarget = observerTarget.current;
    if (currentTarget) {
      observer.observe(currentTarget);
    }

    return () => {
      if (currentTarget) {
        observer.unobserve(currentTarget);
      }
    };
  }, [cargarMasItems, cargandoMas, itemsVisibles, filtradas.length]);

  // ⚡ Solo renderizar las primeras N ordenanzas
  const ordenanzasAMostrar = filtradas.slice(0, itemsVisibles);

  // 🚀 Detectar scroll para mostrar botón
  useEffect(() => {
    const handleScroll = () => {
      setMostrarScrollTop(window.scrollY > 300);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Panel de Filtros - Sticky a la izquierda en desktop, colapsable en mobile */}
        <div className="w-full lg:w-80 lg:flex-shrink-0">
          <div className="lg:sticky lg:top-24 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg transition-all duration-300">
            {/* Header / Toggle Button */}
            <button 
              onClick={() => setFiltrosAbiertos(!filtrosAbiertos)}
              className="w-full flex items-center justify-between p-6 lg:cursor-default"
              disabled={window.innerWidth >= 1024}
            >
              <div className="flex items-center gap-2">
                <Filter className="w-5 h-5 text-[#da1062]" />
                <h2 className="text-lg font-semibold text-gray-800">
                  Filtros de búsqueda
                </h2>
              </div>
              <div className="lg:hidden">
                {filtrosAbiertos ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </div>
            </button>

            {/* Contenido de Filtros (Siempre visible en LG, condicional en móvil) */}
            <div className={`px-6 pb-6 space-y-4 ${filtrosAbiertos ? 'block' : 'hidden lg:block'}`}>
              <div className="space-y-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    N° de Ordenanza
                  </label>
                  <input
                    type="text"
                    value={numeroOrdenanza}
                    onChange={(e) => setNumeroOrdenanza(e.target.value)}
                    placeholder="Ej: 6000"
                    className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Palabra Clave
                  </label>
                  <input
                    type="text"
                    value={palabraClave}
                    onChange={(e) => setPalabraClave(e.target.value)}
                    placeholder="Buscar..."
                    className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Fecha Desde
                  </label>
                  <input
                    type="date"
                    value={fechaDesde}
                    onChange={(e) => setFechaDesde(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Fecha Hasta
                  </label>
                  <input
                    type="date"
                    value={fechaHasta}
                    onChange={(e) => setFechaHasta(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Categoría
                  </label>
                  <select
                    value={categoriaSeleccionada}
                    onChange={(e) => setCategoriaSeleccionada(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Todas las categorías</option>
                    {categoriasUnicas.map((cat: string) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Estado de Vigencia
                  </label>
                  <select
                    value={estadoVigencia}
                    onChange={(e) => setEstadoVigencia(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Todos los estados</option>
                    <option value="vigente">Vigente</option>
                    <option value="derogada">Derogada</option>
                  </select>
                </div>
              </div>

              <button
                onClick={limpiarFiltros}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm text-gray-600 hover:text-gray-800 transition border border-gray-200 rounded-lg hover:bg-gray-50"
              >
                <X className="w-4 h-4" />
                Limpiar filtros
              </button>

              <div className="mt-4 text-xs text-gray-500 text-center">
                {cargando
                  ? "Cargando..."
                  : usarPaginacion
                  ? `${ordenanzas.length} primeras ordenanzas (Total: ${totalOrdenanzas})`
                  : `${Math.min(itemsVisibles, filtradas.length)} de ${
                      filtradas.length
                    } resultados`}
              </div>
            </div>
          </div>
        </div>

        {/* Contenedor principal de resultados */}
        <div className="flex-1 min-w-0">
          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-red-800 font-semibold mb-1">
                  Error al cargar datos
                </p>
                <p className="text-red-700 text-sm">{error}</p>
                <button
                  onClick={cargarMetadatos}
                  className="mt-3 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white text-sm font-medium transition"
                >
                  Reintentar
                </button>
              </div>
            </div>
          )}

          {/* Resultados */}
          <div className="space-y-3">
            {cargando ? (
              <div className="text-center py-16">
                <div className="animate-spin rounded-full h-12 w-12 border-3 border-blue-600 border-t-transparent mx-auto"></div>
                <p className="text-gray-500 mt-4">Cargando ordenanzas...</p>
              </div>
            ) : filtradas.length === 0 ? (
              <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg p-12 text-center">
                <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500 text-lg">
                  No se encontraron ordenanzas con los filtros aplicados
                </p>
              </div>
            ) : (
              <>
                {ordenanzasAMostrar.map((ord: Metadato) => (
                  <div
                    key={ord.numero_ordenanza}
                    onClick={() => setExpandida(expandida === ord.numero_ordenanza ? null : ord.numero_ordenanza)}
                    className="bg-white/95 backdrop-blur-sm rounded-xl shadow p-6 hover:shadow-md transition cursor-pointer lg:cursor-default group"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex flex-wrap items-center gap-3 mb-3">
                          <h3 className="text-xl font-bold text-gray-800">
                            Ordenanza N° {ord.numero_ordenanza}
                          </h3>
                          <div className="flex items-center gap-2 text-sm text-gray-500">
                            <Calendar className="w-4 h-4" />
                            {ord.fecha_sancion}
                          </div>
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-medium ${
                              ord.estado_vigencia === "vigente"
                                ? "bg-green-100 text-green-700"
                                : "bg-red-100 text-red-700"
                            }`}
                          >
                            {ord.estado_vigencia
                              ? ord.estado_vigencia.toUpperCase()
                              : "DESCONOCIDO"}
                          </span>
                        </div>

                        <div className="mb-3">
                          <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 rounded-lg text-sm font-medium">
                            {ord.descripcion_categoria}
                          </span>
                        </div>

                        <div className={`mb-3 overflow-hidden transition-all duration-300 ${expandida === ord.numero_ordenanza ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0 lg:max-h-[1000px] lg:opacity-100'}`}>
                          <p className="text-sm font-semibold text-gray-700 mb-1">
                            Artículo 1°:
                          </p>
                          <p className="text-sm text-gray-600 leading-relaxed">
                            {ord["Art N°1"] || "No disponible"}
                          </p>
                        </div>

                        <div className={`transition-all duration-300 ${expandida === ord.numero_ordenanza ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0 lg:max-h-[500px] lg:opacity-100 overflow-hidden'}`}>
                          {ord.temas && ord.temas.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-3">
                              {ord.temas.slice(0, 3).map((tema: string, i: number) => (
                                <span
                                  key={i}
                                  className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs"
                                >
                                  {tema}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>

                        {expandida !== ord.numero_ordenanza && (
                          <p className="text-[10px] text-[#da1062] font-semibold lg:hidden text-center mt-2 border-t border-dashed border-[#da1062]/20 pt-2">
                            Click para ver más detalles
                          </p>
                        )}
                      </div>
                      

                      <button
                        onClick={(event) => {
                          event.stopPropagation(); // Evitar que el click en el botón triggereé el colapsable
                          abrirPDF(ord.nombre_archivo);
                        }}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition text-white text-sm font-medium whitespace-nowrap shadow-md hover:shadow-lg"
                      >
                        <FileText className="w-4 h-4" />
                        Ver PDF
                      </button>
                    </div>
                  </div>
                ))}

                {/* ⚡ Observer target - se carga más al llegar aquí */}
                {itemsVisibles < filtradas.length && (
                  <div ref={observerTarget} className="py-8 text-center">
                    {cargandoMas && (
                      <div className="flex items-center justify-center gap-3">
                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-blue-600 border-t-transparent"></div>
                        <p className="text-gray-500">
                          Cargando más ordenanzas...
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* ⚡ Mensaje cuando se cargaron todas */}
                {itemsVisibles >= filtradas.length &&
                  filtradas.length > ITEMS_POR_PAGINA && (
                    <div className="py-8 text-center">
                      <p className="text-gray-500 text-sm">
                        ✓ Se cargaron todas las {filtradas.length} ordenanzas
                      </p>
                    </div>
                  )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* 🚀 Botón flotante para subir */}
      {mostrarScrollTop && (
        <button
          onClick={() => {
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
          className="fixed bottom-20 right-6 z-50 text-white bg-[#da1062] hover:brightness-110 rounded-full p-4 shadow-lg hover:scale-110 transition-all duration-300 shadow-[#da1062]/20"
          aria-label="Volver arriba"
        >
          <MoveUp className="w-6 h-6" />
        </button>
      )}
    </div>
  );
}

export default BuscadorManual;
