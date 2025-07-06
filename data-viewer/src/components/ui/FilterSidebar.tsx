import './FilterSidebar.css';
import { type FilterState } from '../../types/index';

interface FilterSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  totalImages: number;
  filteredImages: number;
}

export function FilterSidebar({ 
  isOpen, 
  onClose, 
  filters, 
  onFiltersChange, 
  totalImages, 
  filteredImages 
}: FilterSidebarProps) {
  
  const updateFilter = <K extends keyof FilterState>(
    key: K, 
    value: FilterState[K]
  ) => {
    onFiltersChange({
      ...filters,
      [key]: value
    });
  };

  const clearAllFilters = () => {
    onFiltersChange({
      plane: null,
      version: null,
      class: null,
      subset: null
    });
  };

  return (
    <div className={`filter-sidebar ${isOpen ? 'open' : ''}`}>
      <div className="filter-sidebar-header">
        <h2>Image Filters</h2>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      
      <div className="filter-stats">
        <p>Showing {filteredImages} of {totalImages} images</p>
      </div>

      <div className="filter-section">
        <h3>Brain Plane</h3>
        <div className="filter-options">
          <button 
            className={filters.plane === null ? 'active' : ''}
            onClick={() => updateFilter('plane', null)}
          >
            All Planes
          </button>
          <button 
            className={filters.plane === 'axial' ? 'active' : ''}
            onClick={() => updateFilter('plane', 'axial')}
          >
            Axial
          </button>
          <button 
            className={filters.plane === 'coronal' ? 'active' : ''}
            onClick={() => updateFilter('plane', 'coronal')}
          >
            Coronal
          </button>
          <button 
            className={filters.plane === 'sagittal' ? 'active' : ''}
            onClick={() => updateFilter('plane', 'sagittal')}
          >
            Sagittal
          </button>
        </div>
      </div>

      <div className="filter-section">
        <h3>Image Version</h3>
        <div className="filter-options">
          <button 
            className={filters.version === null ? 'active' : ''}
            onClick={() => updateFilter('version', null)}
          >
            All Versions
          </button>
          <button 
            className={filters.version === 'original-images' ? 'active' : ''}
            onClick={() => updateFilter('version', 'original-images')}
          >
            Original
          </button>
          <button 
            className={filters.version === 'enhanced-images' ? 'active' : ''}
            onClick={() => updateFilter('version', 'enhanced-images')}
          >
            Enhanced
          </button>
        </div>
      </div>

      <div className="filter-section">
        <h3>Class</h3>
        <div className="filter-options">
          <button 
            className={filters.class === null ? 'active' : ''}
            onClick={() => updateFilter('class', null)}
          >
            All Classes
          </button>
          <button 
            className={filters.class === 'CN' ? 'active' : ''}
            onClick={() => updateFilter('class', 'CN')}
          >
            CN (Normal)
          </button>
          <button 
            className={filters.class === 'AD' ? 'active' : ''}
            onClick={() => updateFilter('class', 'AD')}
          >
            AD (Alzheimer's)
          </button>
        </div>
      </div>

      <div className="filter-section">
        <h3>Dataset Split</h3>
        <div className="filter-options">
          <button 
            className={filters.subset === null ? 'active' : ''}
            onClick={() => updateFilter('subset', null)}
          >
            All Splits
          </button>
          <button 
            className={filters.subset === 'train' ? 'active' : ''}
            onClick={() => updateFilter('subset', 'train')}
          >
            Training
          </button>
          <button 
            className={filters.subset === 'test' ? 'active' : ''}
            onClick={() => updateFilter('subset', 'test')}
          >
            Test
          </button>
          <button 
            className={filters.subset === 'val' ? 'active' : ''}
            onClick={() => updateFilter('subset', 'val')}
          >
            Validation
          </button>
        </div>
      </div>

      <div className="filter-actions">
        <button className="clear-filters-button" onClick={clearAllFilters}>
          Clear All Filters
        </button>
      </div>
    </div>
  );
}