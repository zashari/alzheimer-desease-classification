import { useEffect, useMemo } from 'react';
import './ImageViewer.css';
import type { ImageMetadata } from '../../types/index';
import { findImagePair } from '../../utils/imageDataUtils';

interface ImageViewerProps {
  imageUrl: string | null;
  imageData?: ImageMetadata | null;
  onClose: () => void;
}

export function ImageViewer({ imageUrl, imageData, onClose }: ImageViewerProps) {
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  // Find the original/enhanced pair
  const imagePair = useMemo(() => {
    if (!imageData) return { original: null, enhanced: null };
    return findImagePair(imageData);
  }, [imageData]);

  if (!imageUrl) return null;

  return (
    <div className="image-viewer-overlay" onClick={onClose}>
      <div className="image-viewer-container">
        <div className="image-viewer-content">
          {/* Image Comparison Section */}
          <div className="image-comparison-container" onClick={(e) => e.stopPropagation()}>
            {/* Original Image */}
            {imagePair.original && (
              <div className="image-comparison-panel">
                <h4 className="image-panel-title">Original</h4>
                <img 
                  src={imagePair.original.url} 
                  alt="Original version" 
                  className="image-comparison-image"
                />
              </div>
            )}
            
            {/* Enhanced Image */}
            {imagePair.enhanced && (
              <div className="image-comparison-panel">
                <h4 className="image-panel-title">Enhanced</h4>
                <img 
                  src={imagePair.enhanced.url} 
                  alt="Enhanced version" 
                  className="image-comparison-image"
                />
              </div>
            )}
            
            {/* If only one version exists, show it centered */}
            {(!imagePair.original || !imagePair.enhanced) && (
              <div className="image-comparison-panel single-image">
                <h4 className="image-panel-title">
                  {imageData?.version === 'original-images' ? 'Original' : 'Enhanced'}
                  {(!imagePair.original || !imagePair.enhanced) && ' (Only version available)'}
                </h4>
                <img 
                  src={imageUrl} 
                  alt="Single version view" 
                  className="image-comparison-image"
                />
              </div>
            )}
          </div>
          
          {/* Image Details Panel */}
          <div className="image-details-panel" onClick={(e) => e.stopPropagation()}>
            <h3 className="image-details-title">Image Details</h3>
            
            <div className="image-detail-item">
              <span className="detail-label">Filename:</span>
              <span className="detail-value">{imageData?.filename || 'Unknown'}</span>
            </div>
            
            <div className="image-detail-item">
              <span className="detail-label">Brain Plane:</span>
              <span className="detail-value">
                {imageData?.plane ? imageData.plane.charAt(0).toUpperCase() + imageData.plane.slice(1) : 'Unknown'}
              </span>
            </div>
            
            <div className="image-detail-item">
              <span className="detail-label">Versions Available:</span>
              <span className="detail-value">
                {imagePair.original && imagePair.enhanced ? 'Original & Enhanced' :
                 imagePair.original ? 'Original only' :
                 imagePair.enhanced ? 'Enhanced only' : 'Unknown'}
              </span>
            </div>
            
            <div className="image-detail-item">
              <span className="detail-label">Class:</span>
              <span className="detail-value">
                {imageData?.class === 'CN' ? 'CN (Normal)' : 
                 imageData?.class === 'AD' ? 'AD (Alzheimer\'s)' : 'Unknown'}
              </span>
            </div>
            
            <div className="image-detail-item">
              <span className="detail-label">Dataset Split:</span>
              <span className="detail-value">
                {imageData?.subset === 'train' ? 'Training' :
                 imageData?.subset === 'test' ? 'Test' :
                 imageData?.subset === 'val' ? 'Validation' : 'Unknown'}
              </span>
            </div>
            
            <div className="image-detail-item">
              <span className="detail-label">File Type:</span>
              <span className="detail-value">PNG</span>
            </div>
          </div>
        </div>
        
        <button className="image-viewer-close" onClick={onClose}>
          ×
        </button>
      </div>
    </div>
  );
}