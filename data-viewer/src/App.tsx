
import { useMemo, Suspense, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import './App.css';
import { Scene } from './components/viewer/Scene';
import { ImageViewer } from './components/ui/ImageViewer';
import { FilterSidebar } from './components/ui/FilterSidebar';
import { LoadingAnimation } from './components/ui/LoadingAnimation';
import { ServiceWorkerLoadingIndicator } from './components/ui/ServiceWorkerLoadingIndicator';
import { useViewerStore } from './store/viewerStore';
import { imageData, filterImages } from './utils/imageDataUtils';

function App() {
  const {
    filters,
    selectedImage,
    selectedImageData,
    isLoading,
    isServiceWorkerLoading,
    isFilterSidebarOpen,
    setFilters,
    setSelectedImage,
    setServiceWorkerLoading,
    toggleFilterSidebar,
    setFilterSidebarOpen,
    clearSelectedImage
  } = useViewerStore();

  // Handle filter changes with loading state
  const handleFiltersChange = (newFilters: typeof filters) => {
    setFilters(newFilters);
    
    // Notify Service Worker to start tracking
    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      const filtered = filterImages(newFilters);
      const displayCount = Math.min(filtered.length, 500);
      navigator.serviceWorker.controller.postMessage({
        type: 'START_TRACKING',
        total: displayCount
      });
    }
  };

  const handleImageClick = (url: string | null, data?: any) => {
    setSelectedImage(url, data);
  };

  const handleServiceWorkerLoadingComplete = () => {
    setServiceWorkerLoading(false);
  };

  // Initialize Service Worker tracking on first load
  useEffect(() => {
    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      const filtered = filterImages(filters);
      const displayCount = Math.min(filtered.length, 500);
      navigator.serviceWorker.controller.postMessage({
        type: 'START_TRACKING',
        total: displayCount
      });
    }
  }, []); // Only run once on mount

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
      
      {/* Service Worker Loading - shows during caching phase */}
      <ServiceWorkerLoadingIndicator 
        isVisible={isServiceWorkerLoading}
        totalImages={Math.min(totalFiltered, 500)}
        onLoadingComplete={handleServiceWorkerLoadingComplete}
      />
      
      {/* Regular loading animation - shows during batch rendering */}
      {isLoading && !isServiceWorkerLoading && <LoadingAnimation />}
      <Canvas id="webgl-renderer">
        <Suspense fallback={null}>
          <Scene 
            onImageClick={handleImageClick} 
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
