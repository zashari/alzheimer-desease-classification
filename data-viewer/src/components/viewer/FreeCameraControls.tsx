import { useRef, useEffect, useState } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export function FreeCameraControls() {
  const { camera, gl } = useThree();
  
  // Set initial camera position to max zoom out
  useEffect(() => {
    camera.position.set(0, 0, 25); // Max zoom out distance
    camera.lookAt(0, 0, 0); // Look at center
  }, [camera]);
  const panDelta = useRef({ x: 0, y: 0 });
  const isMouseDown = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });
  const [isMoving, setIsMoving] = useState(false);
  const [motionIntensity, setMotionIntensity] = useState(0);
  const movementTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const velocityRef = useRef({ x: 0, y: 0, zoom: 0 });
  const momentumRef = useRef({ x: 0, y: 0, zoom: 0 });
  const dampingFactor = 0.85; // How quickly momentum decays
  
  // Zoom limits and sphere center
  const maxZoomOutDistance = 25; // Maximum distance from origin
  const sphereCenter = new THREE.Vector3(0, 0, 0); // Center of the sphere

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      
      // Check if this is a trackpad pan gesture
      // Include both horizontal (deltaX) and vertical (deltaY) movement
      const isTrackpadPan = (Math.abs(e.deltaX) > 0 || (Math.abs(e.deltaY) > 0 && e.deltaMode === 0)) && !e.ctrlKey;
      
      if (isTrackpadPan) {
        // Two-finger swipe on trackpad for panning
        // Make trackpad feel like natural scrolling (opposite of deltas)
        panDelta.current.x += e.deltaX * 0.01;  // Natural direction for trackpad
        panDelta.current.y -= e.deltaY * 0.01;  // Natural direction for trackpad
      } else {
        // Scroll wheel or pinch zoom
        const rect = gl.domElement.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
        
        const zoomDirection = raycaster.ray.direction.normalize();
        const speed = 0.3; // Slower, smoother zoom speed
        
        // Calculate zoom velocity and add momentum
        const zoomVelocity = Math.abs(e.deltaY) * 0.01;
        velocityRef.current.zoom = zoomVelocity;
        
        // Add momentum to zoom
        const zoomMomentum = e.deltaY > 0 ? -speed : speed;
        momentumRef.current.zoom += zoomMomentum * 0.3; // Add momentum
        
        if (e.deltaY < 0) {
          // Zoom in with momentum - no restrictions
          const totalZoom = speed + momentumRef.current.zoom * 0.5;
          camera.position.add(zoomDirection.multiplyScalar(totalZoom));
        } else {
          // Zoom out with momentum - check distance limit
          const totalZoom = speed + Math.abs(momentumRef.current.zoom) * 0.5;
          const newPosition = camera.position.clone().add(zoomDirection.multiplyScalar(-totalZoom));
          const distanceFromCenter = newPosition.distanceTo(sphereCenter);
          
          if (distanceFromCenter <= maxZoomOutDistance) {
            camera.position.copy(newPosition);
          } else {
            // At max zoom out, smoothly center the camera view towards sphere center
            const maxPosition = sphereCenter.clone().add(
              camera.position.clone().sub(sphereCenter).normalize().multiplyScalar(maxZoomOutDistance)
            );
            camera.position.copy(maxPosition);
            camera.lookAt(sphereCenter);
          }
        }
        
        // Trigger enhanced motion blur for zoom
        triggerMotionBlur(zoomVelocity * 3);
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button === 0) {
        isMouseDown.current = true;
        lastMousePos.current = { x: e.clientX, y: e.clientY };
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (isMouseDown.current) {
        const deltaX = e.clientX - lastMousePos.current.x;
        const deltaY = e.clientY - lastMousePos.current.y;
        
        // Calculate pan velocity and add momentum
        const panVelocity = Math.sqrt(deltaX * deltaX + deltaY * deltaY) * 0.001;
        velocityRef.current.x = Math.abs(deltaX) * 0.001;
        velocityRef.current.y = Math.abs(deltaY) * 0.001;
        
        // Add momentum to panning
        momentumRef.current.x += deltaX * 0.003;
        momentumRef.current.y += deltaY * 0.003;
        
        // Invert the movement so dragging feels like grabbing the scene
        // When you drag right, the scene moves left (camera moves left to see content on the right)
        panDelta.current.x -= (deltaX * 0.008) + (momentumRef.current.x * 0.2); // Add momentum
        panDelta.current.y += (deltaY * 0.008) + (momentumRef.current.y * 0.2);
        
        lastMousePos.current = { x: e.clientX, y: e.clientY };
        
        // Trigger enhanced motion blur for panning
        triggerMotionBlur(panVelocity * 5);
      }
    };

    const handleMouseUp = () => {
      isMouseDown.current = false;
    };

    gl.domElement.addEventListener('wheel', handleWheel, { passive: false });
    gl.domElement.addEventListener('mousedown', handleMouseDown);
    gl.domElement.addEventListener('mousemove', handleMouseMove);
    gl.domElement.addEventListener('mouseup', handleMouseUp);
    gl.domElement.addEventListener('mouseleave', handleMouseUp);

    return () => {
      gl.domElement.removeEventListener('wheel', handleWheel);
      gl.domElement.removeEventListener('mousedown', handleMouseDown);
      gl.domElement.removeEventListener('mousemove', handleMouseMove);
      gl.domElement.removeEventListener('mouseup', handleMouseUp);
      gl.domElement.removeEventListener('mouseleave', handleMouseUp);
    };
  }, [camera, gl]);

  // Enhanced helper function to trigger motion blur with intensity
  const triggerMotionBlur = (intensity = 1) => {
    setIsMoving(true);
    setMotionIntensity(Math.min(intensity, 3)); // Cap intensity at 3
    
    // Clear existing timeout
    if (movementTimeout.current) {
      clearTimeout(movementTimeout.current);
    }
    
    // Set new timeout to remove blur after movement stops
    movementTimeout.current = setTimeout(() => {
      setIsMoving(false);
      setMotionIntensity(0);
    }, 150); // Slightly longer duration for smoother fade
  };

  useFrame(() => {
    // Apply momentum decay for smoother movement
    momentumRef.current.x *= dampingFactor;
    momentumRef.current.y *= dampingFactor;
    momentumRef.current.zoom *= dampingFactor;
    
    // Check current camera distance
    const currentDistance = camera.position.distanceTo(sphereCenter);
    
    // Apply accumulated pan movement
    if (panDelta.current.x !== 0 || panDelta.current.y !== 0) {
      // If at max zoom out distance, restrict panning and keep centered
      if (currentDistance >= maxZoomOutDistance) {
        // Force camera to look at center and maintain position
        camera.lookAt(sphereCenter);
      } else {
        // Normal panning when not at max distance
        // Get camera's right and up vectors
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        
        // Calculate right vector (perpendicular to forward and up)
        camera.getWorldDirection(right);
        right.cross(camera.up).normalize();
        
        // Use world up vector
        up.copy(camera.up);
        
        // Calculate new position
        const newPosition = camera.position.clone()
          .add(right.multiplyScalar(panDelta.current.x))
          .add(up.multiplyScalar(panDelta.current.y));
        
        // Check if new position would exceed max distance
        const newDistance = newPosition.distanceTo(sphereCenter);
        
        if (newDistance <= maxZoomOutDistance) {
          // Apply pan movement normally
          camera.position.copy(newPosition);
        } else {
          // Constrain movement to stay within max distance
          const directionFromCenter = newPosition.clone().sub(sphereCenter).normalize();
          const constrainedPosition = sphereCenter.clone().add(
            directionFromCenter.multiplyScalar(maxZoomOutDistance)
          );
          camera.position.copy(constrainedPosition);
          camera.lookAt(sphereCenter);
        }
        
        // Calculate total velocity for enhanced motion blur
        const totalVelocity = Math.sqrt(
          velocityRef.current.x * velocityRef.current.x + 
          velocityRef.current.y * velocityRef.current.y
        );
        
        // Trigger enhanced motion blur
        triggerMotionBlur(totalVelocity * 4);
      }
      
      // Reset deltas after applying
      panDelta.current = { x: 0, y: 0 };
    }
  });

  // Apply enhanced motion blur with intensity-based effects
  useEffect(() => {
    if (gl.domElement) {
      if (isMoving) {
        // Enhanced motion blur with intensity and multiple effects
        const blurAmount = Math.max(1.5, motionIntensity * 1.2); // Minimum 1.5px, scales with intensity
        const brightness = Math.max(0.95, 1 - motionIntensity * 0.05); // Slight darkening for speed effect
        
        gl.domElement.style.filter = `blur(${blurAmount}px) brightness(${brightness})`;
        gl.domElement.style.transition = 'filter 0.03s ease-out';
        gl.domElement.style.transform = 'translateZ(0)'; // Hardware acceleration
      } else {
        gl.domElement.style.filter = 'blur(0px) brightness(1)';
        gl.domElement.style.transition = 'filter 0.25s ease-out';
        gl.domElement.style.transform = 'translateZ(0)';
      }
    }
  }, [isMoving, motionIntensity, gl]);

  return null;
}