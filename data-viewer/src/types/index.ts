export interface FilterState {
  plane: 'axial' | 'coronal' | 'sagittal' | null;
  version: 'original-images' | 'enhanced-images' | null;
  class: 'CN' | 'AD' | null;
  subset: 'train' | 'test' | 'val' | null;
}

export interface ImageMetadata {
  url: string;
  filename: string;
  version: 'original-images' | 'enhanced-images';
  plane: 'axial' | 'coronal' | 'sagittal';
  subset: 'train' | 'test' | 'val';
  class: 'CN' | 'AD';
  fullPath: string;
}