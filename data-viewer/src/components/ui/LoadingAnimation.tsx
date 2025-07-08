import './LoadingAnimation.css';
import { useViewerStore } from '../../store/viewerStore';

export function LoadingAnimation() {
  const { loadingProgress } = useViewerStore();
  
  const progressPercentage = loadingProgress.total > 0 
    ? Math.round((loadingProgress.loaded / loadingProgress.total) * 100)
    : 0;
  
  return (
    <div className="loading-overlay">
      <div className="loading-container">
        <div className="loading-spinner">
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
        </div>
        <p className="loading-text">
          {loadingProgress.total > 0 
            ? `Loading images... ${loadingProgress.loaded}/${loadingProgress.total} (${progressPercentage}%)`
            : 'Loading images...'
          }
        </p>
        {loadingProgress.total > 0 && (
          <div className="loading-progress-bar">
            <div 
              className="loading-progress-fill" 
              style={{ width: `${progressPercentage}%` }}
            ></div>
          </div>
        )}
      </div>
    </div>
  );
}