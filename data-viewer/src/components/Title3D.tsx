import { Text, Billboard } from '@react-three/drei';
import { useMemo } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { FilterState } from './FilterSidebar';

interface Title3DProps {
  filters: FilterState;
  totalFiltered: number;
  displayCount: number;
}

export function Title3D({ filters, totalFiltered, displayCount }: Title3DProps) {

  // Generate dynamic title based on current filters
  const titleText = useMemo(() => {
    const parts = [];
    
    // Plane
    if (filters.plane) {
      parts.push(`${filters.plane.charAt(0).toUpperCase() + filters.plane.slice(1)} plane`);
    } else {
      parts.push("All planes");
    }
    
    // Version
    if (filters.version) {
      const versionText = filters.version === 'original-images' ? 'Original' : 'Enhanced';
      parts.push(`${versionText} version`);
    } else {
      parts.push("All versions");
    }
    
    // Class
    if (filters.class) {
      const classText = filters.class === 'CN' ? 'CN (Normal)' : 'AD (Alzheimer\'s)';
      parts.push(`Class ${classText}`);
    } else {
      parts.push("All classes");
    }
    
    // Dataset split
    if (filters.subset) {
      const splitText = filters.subset === 'train' ? 'Training' : 
                       filters.subset === 'test' ? 'Test' : 'Validation';
      parts.push(`${splitText} split`);
    } else {
      parts.push("All splits");
    }
    
    return `Showing ${parts.join(' - ')}`;
  }, [filters]);

  const countText = useMemo(() => {
    return totalFiltered > 500 
      ? `Displaying ${displayCount} of ${totalFiltered} images` 
      : `${displayCount} images`;
  }, [totalFiltered, displayCount]);

  return (
    <Billboard
      follow={true}
      lockX={false}
      lockY={false}
      lockZ={false}
      position={[0, 13, 5]} // Fixed position: top (Y=18) and in front (Z=5)
    >
      <group>
        {/* Main title */}
        <Text
          position={[0, 1, 0]}
          fontSize={1.5}
          color="white"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.4}
          outlineColor="black"
        >
          {titleText}
        </Text>
        
        {/* Image count subtitle */}
        <Text
          position={[0, -0.2, 0]}
          fontSize={0.8}
          color="#cccccc"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.4}
          outlineColor="black"
        >
          {countText}
        </Text>
      </group>
    </Billboard>
  );
}