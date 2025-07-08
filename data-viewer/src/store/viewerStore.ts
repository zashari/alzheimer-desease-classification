import { create } from 'zustand';
import { type FilterState } from '../types/index';

interface ViewerState {
  filters: FilterState;
  selectedImage: string | null;
  selectedImageData: any;
  isLoading: boolean;
  isFilterSidebarOpen: boolean;
  loadingProgress: { loaded: number; total: number };
  setFilters: (filters: FilterState) => void;
  setSelectedImage: (url: string | null, data?: any) => void;
  setIsLoading: (loading: boolean) => void;
  setFilterSidebarOpen: (open: boolean) => void;
  toggleFilterSidebar: () => void;
  clearSelectedImage: () => void;
  setLoadingProgress: (loaded: number, total: number) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  filters: {
    plane: 'axial', // Default to axial plane
    version: null,
    class: null,
    subset: null
  },
  selectedImage: null,
  selectedImageData: null,
  isLoading: true,
  isFilterSidebarOpen: false,
  loadingProgress: { loaded: 0, total: 0 },
  
  setFilters: (filters: FilterState) => set(() => ({ 
    filters, 
    isLoading: true 
  })),
  
  setSelectedImage: (url: string | null, data?: any) => set({ 
    selectedImage: url, 
    selectedImageData: data 
  }),
  
  setIsLoading: (loading: boolean) => set({ isLoading: loading }),
  
  setFilterSidebarOpen: (open: boolean) => set({ isFilterSidebarOpen: open }),
  
  toggleFilterSidebar: () => set((state) => ({ 
    isFilterSidebarOpen: !state.isFilterSidebarOpen 
  })),
  
  clearSelectedImage: () => set({ 
    selectedImage: null, 
    selectedImageData: null 
  }),
  
  setLoadingProgress: (loaded: number, total: number) => set({ 
    loadingProgress: { loaded, total } 
  })
}));