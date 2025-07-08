import { useViewerStore } from './viewerStore';
import { filterImages } from '../utils/imageDataUtils';

// Memoized selectors for better performance
export const useFilters = () => useViewerStore((state) => state.filters);

export const useSelectedImage = () => useViewerStore((state) => ({
  selectedImage: state.selectedImage,
  selectedImageData: state.selectedImageData
}));

export const useLoadingState = () => useViewerStore((state) => ({
  isLoading: state.isLoading,
  loadingProgress: state.loadingProgress
}));

export const useFilterSidebar = () => useViewerStore((state) => ({
  isFilterSidebarOpen: state.isFilterSidebarOpen,
  setFilterSidebarOpen: state.setFilterSidebarOpen,
  toggleFilterSidebar: state.toggleFilterSidebar
}));

// Computed selectors
export const useFilteredImageCount = () => useViewerStore((state) => {
  const filtered = filterImages(state.filters);
  return {
    totalFiltered: filtered.length,
    displayCount: Math.min(filtered.length, 500)
  };
});

// Actions selectors
export const useImageActions = () => useViewerStore((state) => ({
  setSelectedImage: state.setSelectedImage,
  clearSelectedImage: state.clearSelectedImage
}));

export const useFilterActions = () => useViewerStore((state) => ({
  setFilters: state.setFilters
}));

export const useLoadingActions = () => useViewerStore((state) => ({
  setIsLoading: state.setIsLoading,
  setLoadingProgress: state.setLoadingProgress
}));