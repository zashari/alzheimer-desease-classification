import { useMemo } from 'react';
import * as THREE from 'three';
import { type ImageMetadata } from '../../types/index';

interface InstantPointsProps {
  images: ImageMetadata[];
  positions: { x: number; y: number; z: number }[];
  loadedCount?: number;
}

// Instant visualization using points - no texture loading required
export function InstantPoints({ images, positions, loadedCount = 0 }: InstantPointsProps) {
  const points = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const positionsArray = new Float32Array(positions.length * 3);
    const colors = new Float32Array(positions.length * 3);
    
    positions.forEach((pos, i) => {
      positionsArray[i * 3] = pos.x;
      positionsArray[i * 3 + 1] = pos.y;
      positionsArray[i * 3 + 2] = pos.z;
      
      // Color based on image class
      const image = images[i];
      if (image?.class === 'AD') {
        colors[i * 3] = 1;     // Red for AD
        colors[i * 3 + 1] = 0.2;
        colors[i * 3 + 2] = 0.2;
      } else {
        colors[i * 3] = 0.2;   // Blue for CN
        colors[i * 3 + 1] = 0.2;
        colors[i * 3 + 2] = 1;
      }
    });
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positionsArray, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    return geometry;
  }, [images, positions]);
  
  return (
    <points>
      <bufferGeometry attach="geometry" {...points} />
      <pointsMaterial 
        attach="material" 
        size={0.5} 
        sizeAttenuation={true}
        vertexColors={true}
        transparent={true}
        opacity={Math.max(0, 0.8 - (loadedCount / images.length) * 0.8)}
      />
    </points>
  );
}