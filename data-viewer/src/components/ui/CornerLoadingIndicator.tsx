import { useViewerStore } from '../../store/viewerStore';
import './CornerLoadingIndicator.css';

export function CornerLoadingIndicator() {
  const { loadingProgress } = useViewerStore();
  
  const progressPercentage = loadingProgress.total > 0 
    ? Math.round((loadingProgress.loaded / loadingProgress.total) * 100)
    : 0;

  if (progressPercentage >= 100 || loadingProgress.total === 0) {
    return null;
  }

  return (
    <div className="corner-loading-indicator">
      <div className="mini-spinner"></div>
      <span className="loading-text-mini">
        Loading {loadingProgress.loaded}/{loadingProgress.total}
      </span>
    </div>
  );
}