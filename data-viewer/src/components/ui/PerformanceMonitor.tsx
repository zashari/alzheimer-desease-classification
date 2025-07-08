import { useEffect, useState, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { createPortal } from 'react-dom';

export function PerformanceMonitor() {
  const { gl } = useThree();
  const [fps, setFps] = useState(60);
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [drawCalls, setDrawCalls] = useState(0);
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  
  useFrame(() => {
    frameCount.current++;
    const currentTime = performance.now();
    
    if (currentTime - lastTime.current >= 1000) {
      setFps(Math.round((frameCount.current * 1000) / (currentTime - lastTime.current)));
      frameCount.current = 0;
      lastTime.current = currentTime;
      
      // Get render info
      const info = gl.info;
      setDrawCalls(info.render.calls);
      
      // Get memory usage if available
      if ((performance as any).memory) {
        setMemoryUsage(Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024));
      }
    }
  });

  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .performance-monitor {
        position: fixed;
        top: 80px;
        left: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 10px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
        z-index: 1000;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      .performance-monitor div {
        margin: 2px 0;
      }
      .fps-good { color: #4ade80; }
      .fps-medium { color: #fbbf24; }
      .fps-bad { color: #f87171; }
    `;
    document.head.appendChild(style);
    
    return () => {
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    };
  }, []);

  const fpsClass = fps >= 50 ? 'fps-good' : fps >= 30 ? 'fps-medium' : 'fps-bad';

  // Use portal to render outside of Canvas
  return createPortal(
    <div className="performance-monitor">
      <div className={fpsClass}>FPS: {fps}</div>
      <div>Draw Calls: {drawCalls}</div>
      <div>Memory: {memoryUsage}MB</div>
      <div>Renderer: {gl.capabilities.isWebGL2 ? 'WebGL2' : 'WebGL1'}</div>
    </div>,
    document.body
  );
}