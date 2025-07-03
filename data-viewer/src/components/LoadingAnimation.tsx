import './LoadingAnimation.css';

export function LoadingAnimation() {
  return (
    <div className="loading-overlay">
      <div className="loading-container">
        <div className="loading-spinner">
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
        </div>
        <p className="loading-text">Loading images...</p>
      </div>
    </div>
  );
}