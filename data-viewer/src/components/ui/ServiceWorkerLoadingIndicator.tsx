import { useEffect, useState } from 'react';
import './ServiceWorkerLoadingIndicator.css';

interface ServiceWorkerLoadingIndicatorProps {
  isVisible: boolean;
  totalImages: number;
  onLoadingComplete: () => void;
}

interface LoadingState {
  phase: 'preparing' | 'caching' | 'batching' | 'complete';
  cached: number;
  batched: number;
  total: number;
}

export function ServiceWorkerLoadingIndicator({ 
  isVisible, 
  totalImages,
  onLoadingComplete 
}: ServiceWorkerLoadingIndicatorProps) {
  const [loadingState, setLoadingState] = useState<LoadingState>({
    phase: 'preparing',
    cached: 0,
    batched: 0,
    total: totalImages
  });

  useEffect(() => {
    if (!isVisible) return;

    // Reset state when becoming visible
    setLoadingState({
      phase: 'preparing',
      cached: 0,
      batched: 0,
      total: totalImages
    });

    // Listen for Service Worker messages
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'IMAGE_CACHE_PROGRESS') {
        setLoadingState(prev => ({
          ...prev,
          phase: 'caching',
          cached: event.data.cached,
          total: event.data.total
        }));
      } else if (event.data?.type === 'BATCH_PROGRESS') {
        setLoadingState(prev => ({
          ...prev,
          phase: 'batching',
          batched: event.data.batched
        }));
        
        // Auto-complete when batch reaches total
        if (event.data.batched >= totalImages) {
          setTimeout(() => {
            setLoadingState(prevState => ({
              ...prevState,
              phase: 'complete'
            }));
            
            // Notify parent that loading is complete
            setTimeout(() => {
              onLoadingComplete();
            }, 500);
          }, 1000); // Give time for images to render
        }
      } else if (event.data?.type === 'LOADING_COMPLETE') {
        setLoadingState(prev => ({
          ...prev,
          phase: 'complete'
        }));
        
        // Notify parent that loading is complete
        setTimeout(() => {
          onLoadingComplete();
        }, 500); // Brief delay to show completion
      }
    };

    // Register service worker message listener
    navigator.serviceWorker?.addEventListener('message', handleMessage);

    // Start preparation phase
    setTimeout(() => {
      setLoadingState(prev => ({ ...prev, phase: 'caching' }));
    }, 300);

    return () => {
      navigator.serviceWorker?.removeEventListener('message', handleMessage);
    };
  }, [isVisible, totalImages, onLoadingComplete]);

  const getProgressPercentage = () => {
    switch (loadingState.phase) {
      case 'preparing':
        return 5;
      case 'caching':
        return 10 + (loadingState.cached / loadingState.total) * 60; // 10-70%
      case 'batching':
        return 70 + (loadingState.batched / loadingState.total) * 25; // 70-95%
      case 'complete':
        return 100;
      default:
        return 0;
    }
  };

  const getPhaseText = () => {
    switch (loadingState.phase) {
      case 'preparing':
        return 'Preparing images...';
      case 'caching':
        return `Caching images... (${loadingState.cached}/${loadingState.total})`;
      case 'batching':
        return `Loading images... (${loadingState.batched}/${loadingState.total})`;
      case 'complete':
        return 'Complete!';
      default:
        return 'Loading...';
    }
  };

  if (!isVisible) return null;

  return (
    <div className="sw-loading-overlay">
      <div className="sw-loading-container">
        <div className="sw-loading-header">
          <h3>Alzheimer's Disease Image Viewer</h3>
          <p>Loading brain scan images...</p>
        </div>
        
        <div className="sw-progress-container">
          <div className="sw-progress-bar">
            <div 
              className="sw-progress-fill"
              style={{ width: `${getProgressPercentage()}%` }}
            />
          </div>
          <div className="sw-progress-text">
            {Math.round(getProgressPercentage())}%
          </div>
        </div>
        
        <div className="sw-phase-indicator">
          <div className={`sw-phase ${loadingState.phase === 'preparing' ? 'active' : loadingState.phase !== 'preparing' ? 'complete' : ''}`}>
            <div className="sw-phase-dot"></div>
            <span>Preparing</span>
          </div>
          <div className={`sw-phase ${loadingState.phase === 'caching' ? 'active' : loadingState.phase === 'batching' || loadingState.phase === 'complete' ? 'complete' : ''}`}>
            <div className="sw-phase-dot"></div>
            <span>Caching</span>
          </div>
          <div className={`sw-phase ${loadingState.phase === 'batching' ? 'active' : loadingState.phase === 'complete' ? 'complete' : ''}`}>
            <div className="sw-phase-dot"></div>
            <span>Rendering</span>
          </div>
        </div>
        
        <div className="sw-status-text">
          {getPhaseText()}
        </div>
        
        <div className="sw-loading-tips">
          <p>💡 Images are being cached for faster future access</p>
          <p>🧠 Analyzing {totalImages} brain scan images</p>
        </div>
      </div>
    </div>
  );
}