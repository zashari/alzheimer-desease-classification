import { useEffect, useState } from 'react';
import { File } from './File';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { type ImageMetadata } from '../../types/index';
import { useViewerStore } from '../../store/viewerStore';

interface BatchLoadedImagesProps {
  images: ImageMetadata[];
  positions: { x: number; y: number; z: number }[];
  onImageClick: (url: string | null, data?: any) => void;
}

export function BatchLoadedImages({ 
  images, 
  positions, 
  onImageClick
}: BatchLoadedImagesProps) {
  const [loadedCount, setLoadedCount] = useState(0);
  const batchSize = 25; // Load 25 images at a time
  const { setLoadingProgress, setIsLoading } = useViewerStore();
  
  useEffect(() => {
    // Load images in batches
    const timer = setInterval(() => {
      setLoadedCount(prev => {
        const next = Math.min(prev + batchSize, images.length);
        
        // Hide loading screen when first batch is loaded (not before)
        if (prev === 0 && next > 0) {
          setTimeout(() => {
            setIsLoading(false);
          }, 100); // Small delay to ensure first batch renders
        }
        
        if (next >= images.length) {
          clearInterval(timer);
        }
        return next;
      });
    }, 200); // 200ms delay between batches
    
    return () => clearInterval(timer);
  }, [images.length, batchSize, setIsLoading]);

  // Reset when images change - keep loading state true
  useEffect(() => {
    setLoadedCount(0);
    setLoadingProgress(0, images.length);
    // Don't modify isLoading here - let it be controlled by setFilters and first batch
  }, [images, setLoadingProgress]);

  // Update loading progress when loadedCount changes
  useEffect(() => {
    setLoadingProgress(loadedCount, images.length);
  }, [loadedCount, images.length, setLoadingProgress]);
  
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
            onImageClick={(url) => onImageClick(url, image)}
          />
        </ErrorBoundary>
      ))}
    </>
  );
}