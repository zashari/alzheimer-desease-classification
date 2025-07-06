
import { useMemo, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import './App.css';
import { Scene } from './components/viewer/Scene';
import { ImageViewer } from './components/ui/ImageViewer';
import { FilterSidebar } from './components/ui/FilterSidebar';
import { LoadingAnimation } from './components/ui/LoadingAnimation';
import { useViewerStore } from './store/viewerStore';
import { imageData, filterImages } from './utils/imageDataUtils';

function App() {
  const {
    filters,
    selectedImage,
    selectedImageData,
    isLoading,
    isFilterSidebarOpen,
    setFilters,
    setSelectedImage,
    toggleFilterSidebar,
    setFilterSidebarOpen,
    clearSelectedImage
  } = useViewerStore();

  // Handle filter changes with loading state
  const handleFiltersChange = (newFilters: typeof filters) => {
    setFilters(newFilters);
  };

  const handleImageSelect = (url: string | null, data?: any) => {
    setSelectedImage(url, data);
  };


  // Calculate filtered image count
  const totalFiltered = useMemo(() => {
    const filtered = filterImages(filters);
    return filtered.length;
  }, [filters]);

  return (
    <main className="spatial-canvas">
      <button 
        onClick={toggleFilterSidebar}
        className="filters-button"
      >
        Filters
      </button>
      {isLoading && <LoadingAnimation />}
      <Canvas id="webgl-renderer">
        <Suspense fallback={null}>
          <Scene 
            onImageSelect={handleImageSelect} 
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
        onClose={clearSelectedImage}
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
