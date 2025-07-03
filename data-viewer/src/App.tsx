
import { useState, useMemo, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import './App.css';
import { Scene, imageData } from './components/Scene';
import { ImageViewer } from './components/ImageViewer';
import { FilterSidebar, type FilterState } from './components/FilterSidebar';
import { SimplePerformanceIndicator } from './components/SimplePerformanceIndicator';
import { LoadingAnimation } from './components/LoadingAnimation';

function App() {
  const [isFilterSidebarOpen, setFilterSidebarOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImageData, setSelectedImageData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState<FilterState>({
    plane: null,
    version: null,
    class: null,
    subset: null
  });

  // Handle filter changes with loading state
  const handleFiltersChange = (newFilters: FilterState) => {
    setIsLoading(true);
    setFilters(newFilters);
  };

  const handleImageSelect = (url: string | null, data?: any) => {
    setSelectedImage(url);
    setSelectedImageData(data);
  };


  // Calculate filtered image count and display count
  const { totalFiltered, displayCount } = useMemo(() => {
    let filtered = imageData;
    
    if (filters.plane) {
      filtered = filtered.filter(img => img.plane === filters.plane);
    }
    
    if (filters.version) {
      filtered = filtered.filter(img => img.version === filters.version);
    }
    
    if (filters.class) {
      filtered = filtered.filter(img => img.class === filters.class);
    }
    
    if (filters.subset) {
      filtered = filtered.filter(img => img.subset === filters.subset);
    }
    
    const totalFiltered = filtered.length;
    const displayCount = Math.min(totalFiltered, 500);
    
    return { totalFiltered, displayCount };
  }, [filters]);

  return (
    <main className="spatial-canvas">
      <button 
        onClick={() => setFilterSidebarOpen(!isFilterSidebarOpen)} 
        style={{position: 'absolute', zIndex: 11, top: 20, right: 20, background: 'rgba(0,0,0,0.8)', color: 'white', border: '1px solid rgba(255,255,255,0.3)', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer'}}
      >
        Filters
      </button>
      {isLoading && <LoadingAnimation />}
      <Canvas id="webgl-renderer">
        <Suspense fallback={null}>
          <Scene 
            onImageSelect={handleImageSelect} 
            filters={filters}
            onSceneReady={() => setIsLoading(false)}
          />
        </Suspense>
      </Canvas>
      <nav className="floating-controls">
        <div className="zoom-control"></div>
        <div className="view-switcher"></div>
      </nav>
      
      {/* Author Credit */}
      <div className="author-credit">
        Author: Zaky Ashari - izzat.zaky@gmail.com
      </div>
      <ImageViewer 
        imageUrl={selectedImage}
        imageData={selectedImageData}
        onClose={() => {
          setSelectedImage(null);
          setSelectedImageData(null);
        }} 
      />
      <FilterSidebar
        isOpen={isFilterSidebarOpen}
        onClose={() => setFilterSidebarOpen(false)}
        filters={filters}
        onFiltersChange={handleFiltersChange}
        totalImages={imageData.length}
        filteredImages={totalFiltered}
      />
    </main>
  );
}

export default App;
