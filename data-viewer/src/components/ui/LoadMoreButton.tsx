import './LoadMoreButton.css';

interface LoadMoreButtonProps {
  currentCount: number;
  totalCount: number;
  onLoadMore: () => void;
  increment?: number;
}

export function LoadMoreButton({ 
  currentCount, 
  totalCount, 
  onLoadMore, 
  increment = 50 
}: LoadMoreButtonProps) {
  if (currentCount >= totalCount) {
    return null;
  }
  
  const remaining = totalCount - currentCount;
  const loadCount = Math.min(increment, remaining);
  
  return (
    <div className="load-more-container">
      <button 
        className="load-more-button"
        onClick={onLoadMore}
      >
        Load {loadCount} more images
        <span className="load-more-info">
          ({currentCount}/{totalCount} loaded)
        </span>
      </button>
    </div>
  );
}