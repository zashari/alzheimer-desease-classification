import { Stars } from '@react-three/drei';
import { useMemo, useEffect, useRef } from 'react';
import { FreeCameraControls } from './FreeCameraControls';
import { Title3D } from './Title3D';
import { useViewerStore } from '../../store/viewerStore';
import { filterImages, sampleImages } from '../../utils/imageDataUtils';
import { BatchLoadedImages } from './BatchLoadedImages';

export function Scene({ 
  onImageClick
}: { 
  onImageClick: (url: string | null, data?: any) => void;
}) {
  const { filters } = useViewerStore();
  
  const hasLoggedPerformance = useRef(false);
  
  // Reset logging flag when filters change
  useEffect(() => {
    hasLoggedPerformance.current = false;
  }, [filters]);
  
  // Filter images and randomly select up to 500 for display
  const filteredImages = useMemo(() => {
    const filtered = filterImages(filters);
    const displayImages = sampleImages(filtered, 500);
    
    // Log the actual count being loaded (only once per filter change)
    if (!hasLoggedPerformance.current) {
      console.log(`Displaying ${displayImages.length} of ${filtered.length} filtered images from CloudFront`);
      hasLoggedPerformance.current = true;
    }
    
    return displayImages;
  }, [filters]);

  // Calculate filtered counts for title
  const { totalFiltered, displayCount } = useMemo(() => {
    const filtered = filterImages(filters);
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
        onImageClick={onImageClick}
      />
    </>
  );
}