
import { useTexture, Billboard } from '@react-three/drei';
import { useRef, useState } from 'react';
import { useSpring, animated } from '@react-spring/three';
import * as THREE from 'three';

interface FileProps {
  imageUrl: string;
  position: [number, number, number];
  onImageClick?: (imageUrl: string, imageData?: any) => void;
}

export function File({ imageUrl, position, onImageClick, ...props }: FileProps) {
  const ref = useRef<THREE.Mesh>(null!);
  const texture = useTexture(imageUrl);
  const [hovered, hover] = useState(false);

  // Spring animation for hover effect
  const springs = useSpring({
    scale: hovered ? 1.1 : 1,
    positionZ: hovered ? 0.2 : 0,
    config: { tension: 300, friction: 20 }
  });

  const handleClick = () => {
    if (onImageClick) {
      // Pass image data along with URL
      const imageData = {
        name: imageUrl.split('/').pop()?.split('?')[0] || 'Unknown',
        width: texture?.image?.width || 0,
        height: texture?.image?.height || 0,
        url: imageUrl
      };
      onImageClick(imageUrl, imageData);
    }
  };

  return (
    <Billboard
      {...props}
      position={position}
      follow={true} // Always face the camera
      lockX={false} // Allow rotation around X axis
      lockY={false} // Allow rotation around Y axis
      lockZ={false} // Allow rotation around Z axis
    >
      <animated.mesh
        ref={ref}
        scale={springs.scale}
        position-z={springs.positionZ}
        onClick={handleClick}
        onPointerOver={() => hover(true)}
        onPointerOut={() => hover(false)}
      >
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial map={texture} color={texture ? 'white' : 'gray'} />
      </animated.mesh>
    </Billboard>
  );
}
