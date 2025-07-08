import { useEffect, useState, useRef } from 'react';
import { File } from './File';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { type ImageMetadata } from '../../types/index';
import { useViewerStore } from '../../store/viewerStore';

// Global flag to prevent multiple completion messages
let isCompletionSent = false;

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
  const instanceId = useRef(Math.random().toString(36).substr(2, 9));
  
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
    // Reset completion flag when new images load
    isCompletionSent = false;
    console.log(`BatchLoadedImages instance ${instanceId.current} initialized with ${images.length} images`);
  }, [images, setLoadingProgress]);

  // Update loading progress when loadedCount changes
  useEffect(() => {
    setLoadingProgress(loadedCount, images.length);
  }, [loadedCount, images.length, setLoadingProgress]);
  
  // Log progress and notify Service Worker
  useEffect(() => {
    if (loadedCount > 0 && loadedCount <= images.length) {
      console.log(`[${instanceId.current}] Loading batch: ${loadedCount}/${images.length} images`);
      
      // Send batch progress to Service Worker for forwarding
      if (navigator.serviceWorker && navigator.serviceWorker.controller) {
        navigator.serviceWorker.controller.postMessage({
          type: 'BATCH_PROGRESS',
          batched: loadedCount,
          total: images.length,
          instanceId: instanceId.current
        });
        
        // If all images are loaded, notify completion (only once)
        if (loadedCount === images.length && !isCompletionSent) {
          isCompletionSent = true;
          console.log(`[${instanceId.current}] Sending completion signal`);
          setTimeout(() => {
            navigator.serviceWorker?.controller?.postMessage({
              type: 'LOADING_COMPLETE',
              instanceId: instanceId.current
            });
          }, 500);
        }
      }
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