import { S3_IMAGE_PATHS, parseS3Path, TOTAL_IMAGES } from '../data/s3-actual-images';
import { type ImageMetadata, type FilterState } from '../types/index';

// Get image base URL from environment variables
const getImageBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_IMAGE_BASE_URL;
  
  if (!envUrl) {
    console.error('VITE_IMAGE_BASE_URL environment variable is not set. Please configure it in .env file.');
    throw new Error('Image base URL is not configured. Please set VITE_IMAGE_BASE_URL in your .env file.');
  }
  
  return envUrl;
};

const IMAGE_BASE_URL = getImageBaseUrl();

// Create image mapping from S3 paths
const allImages: Record<string, string> = {};
S3_IMAGE_PATHS.forEach(path => {
  const url = `${IMAGE_BASE_URL}/${path}`;
  allImages[path] = url;
});

console.log("Loaded S3 images:", TOTAL_IMAGES);
console.log("Using CloudFront CDN:", IMAGE_BASE_URL.includes('cloudfront'));

// Function to parse image path and extract metadata
function parseImagePath(path: string, url: string): ImageMetadata {
  const metadata = parseS3Path(path);
  
  if (metadata) {
    return {
      url,
      filename: metadata.filename,
      version: metadata.version,
      plane: metadata.plane,
      subset: metadata.subset,
      class: metadata.class,
      fullPath: path
    };
  }
  
  // Fallback parsing for unexpected paths
  const pathParts = path.split('/');
  const filename = pathParts[pathParts.length - 1];
  
  return {
    url,
    filename: filename.replace('.png', ''),
    version: 'enhanced-images',
    plane: 'axial',
    subset: 'train',
    class: 'CN',
    fullPath: path
  };
}

// Parse all images and create metadata
export const imageData: ImageMetadata[] = Object.entries(allImages).map(([path, url]) => 
  parseImagePath(path, url as string)
);

// Function to find the corresponding original/enhanced pair
export function findImagePair(selectedImage: ImageMetadata): { original: ImageMetadata | null, enhanced: ImageMetadata | null } {
  // Find the pair by matching plane, class, subset and filename (without extension)
  const baseFilename = selectedImage.filename;
  
  const original = imageData.find(img => 
    img.version === 'original-images' &&
    img.plane === selectedImage.plane &&
    img.class === selectedImage.class &&
    img.subset === selectedImage.subset &&
    img.filename === baseFilename
  );
  
  const enhanced = imageData.find(img => 
    img.version === 'enhanced-images' &&
    img.plane === selectedImage.plane &&
    img.class === selectedImage.class &&
    img.subset === selectedImage.subset &&
    img.filename === baseFilename
  );
  
  return { original: original || null, enhanced: enhanced || null };
}

// Function to filter images based on filter state
export function filterImages(filters: FilterState): ImageMetadata[] {
  let filtered = imageData;
  
  // Plane filter is always applied (never null)
  filtered = filtered.filter(img => img.plane === filters.plane);
  
  if (filters.version) {
    filtered = filtered.filter(img => img.version === filters.version);
  }
  
  if (filters.class) {
    filtered = filtered.filter(img => img.class === filters.class);
  }
  
  if (filters.subset) {
    filtered = filtered.filter(img => img.subset === filters.subset);
  }
  
  return filtered;
}

// Function to randomly sample images for performance
export function sampleImages(images: ImageMetadata[], maxCount: number = 500): ImageMetadata[] {
  if (images.length <= maxCount) {
    return images;
  }
  
  // Create a copy and shuffle it
  const shuffled = [...images];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  
  return shuffled.slice(0, maxCount);
}

console.log("Parsed image metadata:", imageData.length);