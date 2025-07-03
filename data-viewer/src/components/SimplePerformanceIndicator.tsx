import { useEffect, useState } from 'react';

export function SimplePerformanceIndicator() {
  const [status, setStatus] = useState<'loading' | 'good' | 'warning'>('loading');
  const [imageCount, setImageCount] = useState(0);

  useEffect(() => {
    // Simple performance check based on browser capabilities
    const checkPerformance = () => {
      const isWebGL2 = !!document.createElement('canvas').getContext('webgl2');
      const memoryInfo = (performance as any).memory;
      const hasGoodGPU = isWebGL2;
      
      if (hasGoodGPU && (!memoryInfo || memoryInfo.usedJSHeapSize < 200 * 1024 * 1024)) {
        setStatus('good');
      } else {
        setStatus('warning');
      }
    };

    checkPerformance();
    
    // Listen for image count from console logs
    const originalLog = console.log;
    let hasIntercepted = false;
    
    console.log = (...args) => {
      const message = args.join(' ');
      if (message.includes('Displaying') && message.includes('images from CloudFront') && !hasIntercepted) {
        const match = message.match(/Displaying (\d+) of (\d+)/);
        if (match) {
          setImageCount(parseInt(match[1]));
          hasIntercepted = true;
          // Reset after a delay to allow new filter changes
          setTimeout(() => { hasIntercepted = false; }, 1000);
        }
      }
      originalLog(...args);
    };

    return () => {
      console.log = originalLog;
    };
  }, []);

  return (
    <div style={{
      position: 'fixed',
      top: '80px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: status === 'good' ? '#4ade80' : status === 'warning' ? '#fbbf24' : '#f87171',
      padding: '8px 12px',
      borderRadius: '4px',
      fontSize: '12px',
      fontFamily: 'monospace',
      zIndex: 1000,
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255, 255, 255, 0.1)'
    }}>
      {imageCount > 0 ? `${imageCount} images loaded` : 'Loading images...'}
      <div style={{ fontSize: '10px', opacity: 0.7 }}>
        {status === 'good' ? '✓ Good performance' : 
         status === 'warning' ? '⚠ Limited performance' : '⟳ Initializing...'}
      </div>
    </div>
  );
}