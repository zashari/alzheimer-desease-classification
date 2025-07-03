import { File } from './File';
import { ErrorBoundary } from './ErrorBoundary';
import { Stars } from '@react-three/drei';
import { useMemo, useEffect, useState, useRef } from 'react';
import { FreeCameraControls } from './FreeCameraControls';
import { S3_IMAGE_PATHS, parseS3Path, TOTAL_IMAGES } from '../data/s3-actual-images';
import { Title3D } from './Title3D';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';

// Use environment variable for S3 URL (can be CloudFront URL later)
const IMAGE_BASE_URL = import.meta.env.VITE_IMAGE_BASE_URL || 'https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com';

// Performance optimization: randomly display up to 500 images per filter combination

// Create image mapping from S3 paths
const allImages: Record<string, string> = {};
S3_IMAGE_PATHS.forEach(path => {
  const url = `${IMAGE_BASE_URL}/${path}`;
  allImages[path] = url;
});

console.log("Loaded S3 images:", TOTAL_IMAGES);
console.log("Using CloudFront CDN:", IMAGE_BASE_URL.includes('cloudfront'));

// Parse image paths to extract metadata
export interface ImageMetadata {
  url: string;
  filename: string;
  version: 'original-images' | 'enhanced-images';
  plane: 'axial' | 'coronal' | 'sagittal';
  subset: 'train' | 'test' | 'val';
  class: 'CN' | 'AD';
  fullPath: string;
}

// Function to parse image path and extract metadata
function parseImagePath(path: string, url: string): ImageMetadata {
  const metadata = parseS3Path(path);
  
  if (metadata) {
    return {
      url,
      filename: metadata.filename,
      version: metadata.version,
      plane: metadata.plane,
      subset: metadata.subset,
      class: metadata.class,
      fullPath: path
    };
  }
  
  // Fallback parsing for unexpected paths
  const pathParts = path.split('/');
  const filename = pathParts[pathParts.length - 1];
  
  return {
    url,
    filename: filename.replace('.png', ''),
    version: 'enhanced-images',
    plane: 'axial',
    subset: 'train',
    class: 'CN',
    fullPath: path
  };
}

// Parse all images and create metadata
export const imageData: ImageMetadata[] = Object.entries(allImages).map(([path, url]) => 
  parseImagePath(path, url as string)
);

// Function to find the corresponding original/enhanced pair
export function findImagePair(selectedImage: ImageMetadata): { original: ImageMetadata | null, enhanced: ImageMetadata | null } {
  // Find the pair by matching plane, class, subset and filename (without extension)
  const baseFilename = selectedImage.filename;
  
  const original = imageData.find(img => 
    img.version === 'original-images' &&
    img.plane === selectedImage.plane &&
    img.class === selectedImage.class &&
    img.subset === selectedImage.subset &&
    img.filename === baseFilename
  );
  
  const enhanced = imageData.find(img => 
    img.version === 'enhanced-images' &&
    img.plane === selectedImage.plane &&
    img.class === selectedImage.class &&
    img.subset === selectedImage.subset &&
    img.filename === baseFilename
  );
  
  return { original: original || null, enhanced: enhanced || null };
}

console.log("Parsed image metadata:", imageData.length);

// Batch loading component to prevent browser overload
function BatchLoadedImages({ images, positions, onImageSelect, onLoadComplete }: { 
  images: ImageMetadata[], 
  positions: any[],
  onImageSelect: (url: string | null, data?: any) => void,
  onLoadComplete?: () => void
}) {
  const [loadedCount, setLoadedCount] = useState(0);
  const batchSize = 25; // Load 25 images at a time
  
  useEffect(() => {
    // Load images in batches
    const timer = setInterval(() => {
      setLoadedCount(prev => {
        const next = Math.min(prev + batchSize, images.length);
        if (next >= images.length) {
          clearInterval(timer);
          // Notify when loading is complete
          if (onLoadComplete) {
            setTimeout(onLoadComplete, 300); // Small delay for last batch to render
          }
        }
        return next;
      });
    }, 200); // 200ms delay between batches
    
    return () => clearInterval(timer);
  }, [images.length, batchSize, onLoadComplete]);
  
  // Reset when images change
  useEffect(() => {
    setLoadedCount(0);
  }, [images]);
  
  // Log progress
  useEffect(() => {
    if (loadedCount > 0 && loadedCount < images.length) {
      console.log(`Loading batch: ${loadedCount}/${images.length} images`);
    }
  }, [loadedCount, images.length]);
  
  return (
    <>
      {images.slice(0, loadedCount).map((image, index) => (
        <ErrorBoundary key={image.url} resetKey={image.url}>
          <File 
            imageUrl={image.url} 
            position={[positions[index].x, positions[index].y, positions[index].z]}
            onImageClick={(url) => onImageSelect(url, image)}
          />
        </ErrorBoundary>
      ))}
    </>
  );
}

export function Scene({ 
  onImageSelect,
  filters = {},
  onSceneReady
}: { 
  onImageSelect: (url: string | null, data?: any) => void;
  onSceneReady?: () => void;
  filters?: {
    plane?: 'axial' | 'coronal' | 'sagittal' | null;
    version?: 'original-images' | 'enhanced-images' | null;
    class?: 'CN' | 'AD' | null;
    subset?: 'train' | 'test' | 'val' | null;
  };
}) {
  
  const hasLoggedPerformance = useRef(false);
  
  // Reset logging flag when filters change
  useEffect(() => {
    hasLoggedPerformance.current = false;
  }, [filters]);
  
  // Filter images and randomly select up to 500 for display
  const filteredImages = useMemo(() => {
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
    
    // Randomly sample up to 500 images for better performance
    const maxDisplayImages = 500;
    let displayImages = filtered;
    
    if (totalFiltered > maxDisplayImages) {
      // Create a copy and shuffle it
      const shuffled = [...filtered];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      displayImages = shuffled.slice(0, maxDisplayImages);
    }
    
    // Log the actual count being loaded (only once per filter change)
    if (!hasLoggedPerformance.current) {
      console.log(`Displaying ${displayImages.length} of ${totalFiltered} filtered images from CloudFront`);
      hasLoggedPerformance.current = true;
    }
    
    return displayImages;
  }, [filters]);

  // Calculate filtered counts for title
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
    
    return {
      totalFiltered: filtered.length,
      displayCount: Math.min(filtered.length, 500)
    };
  }, [filters]);

  const filePositions = useMemo(() => {
    // Spherical arrangement
    const radius = 10; // Base radius of the sphere
    const radiusVariation = 5; // Random variation in radius for organic feel
    
    const positions = filteredImages.map((_, index) => {
      // Use golden angle for even distribution on sphere
      const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // ~137.5 degrees
      const theta = goldenAngle * index; // Rotation around Y axis
      
      // Map index to sphere surface using Fibonacci sphere algorithm
      const y = 1 - (index / (filteredImages.length - 1)) * 2; // -1 to 1
      const radiusAtY = Math.sqrt(1 - y * y); // Circle radius at this Y level
      
      // Add some randomness to radius for organic feel
      const r = radius + (Math.random() - 0.5) * radiusVariation;
      
      return {
        x: Math.cos(theta) * radiusAtY * r,
        y: y * r,
        z: Math.sin(theta) * radiusAtY * r,
      };
    });

    
    return {
      positions,
      maxDistance: 100000, // Very large max distance for full scene overview
    };
  }, [filteredImages.length]); // Recalculate if number of images changes

  return (
    <>
      <FreeCameraControls />
      <ambientLight intensity={0.5} /> {/* Soft ambient light */}
      <directionalLight position={[5, 5, 5]} intensity={1} /> {/* Main light source */}
      <pointLight position={[-5, -5, -5]} intensity={0.8} /> {/* Additional light source */}
      <Stars
        radius={100}
        depth={50}
        count={10000}
        factor={5}
        saturation={0}
        fade={false}
      />
      <Title3D 
        filters={filters}
        totalFiltered={totalFiltered}
        displayCount={displayCount}
      />
      <BatchLoadedImages 
        images={filteredImages} 
        positions={filePositions.positions}
        onImageSelect={onImageSelect}
        onLoadComplete={onSceneReady}
      />
    </>
  );
}