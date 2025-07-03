import * as THREE from 'three';

export interface ProgressiveImageConfig {
  lowResUrl: string;
  highResUrl: string;
  onProgress?: (progress: number) => void;
}

export class ProgressiveImageLoader {
  private loader = new THREE.TextureLoader();
  private cache = new Map<string, THREE.Texture>();
  private loadingQueue = new Set<string>();

  // Create low-resolution placeholder (blur effect)
  private createPlaceholder(width = 64, height = 64): THREE.Texture {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d')!;
    
    // Create a simple gradient placeholder
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#e5e7eb'); // Light gray
    gradient.addColorStop(1, '#9ca3af'); // Darker gray
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Add subtle pattern
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    for (let i = 0; i < width; i += 8) {
      for (let j = 0; j < height; j += 8) {
        if ((i + j) % 16 === 0) {
          ctx.fillRect(i, j, 4, 4);
        }
      }
    }
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    
    return texture;
  }

  // Load image with WebP fallback to PNG and retry logic
  async loadWithFallback(basePath: string, baseUrl: string = '', retries: number = 2): Promise<THREE.Texture> {
    const cacheKey = basePath;
    
    // Check cache first
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    // Prevent duplicate requests
    if (this.loadingQueue.has(cacheKey)) {
      // Wait for existing request
      return new Promise((resolve) => {
        const checkCache = () => {
          if (this.cache.has(cacheKey)) {
            resolve(this.cache.get(cacheKey)!);
          } else if (!this.loadingQueue.has(cacheKey)) {
            // Request failed, return placeholder
            resolve(this.createPlaceholder());
          } else {
            setTimeout(checkCache, 50);
          }
        };
        checkCache();
      });
    }

    this.loadingQueue.add(cacheKey);

    // Construct full URLs - ensure absolute URLs
    const webpPath = 'webp/' + basePath.replace('.png', '.webp');
    const webpUrl = `${baseUrl}/${webpPath}`;
    const pngUrl = `${baseUrl}/${basePath}`;
    
    // Debug only occasionally
    if (Math.random() < 0.01) {
      console.log('URL Construction:', { basePath, webpPath, webpUrl, pngUrl });
    }

    try {
      // Try WebP first
      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          const texture = await this.loader.loadAsync(webpUrl);
          this.cache.set(cacheKey, texture);
          this.loadingQueue.delete(cacheKey);
          return texture;
        } catch (error) {
          if (attempt === retries) {
            // WebP failed after retries, try PNG fallback
            break;
          }
          // Wait briefly before retry
          await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)));
        }
      }

      // Try PNG fallback with retries
      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          const texture = await this.loader.loadAsync(pngUrl);
          this.cache.set(cacheKey, texture);
          this.loadingQueue.delete(cacheKey);
          return texture;
        } catch (pngError) {
          if (attempt === retries) {
            // Both formats failed after retries
            console.warn(`Image loading failed after retries: ${basePath}`);
            break;
          }
          // Wait briefly before retry
          await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)));
        }
      }

      // All attempts failed
      this.loadingQueue.delete(cacheKey);
      const placeholder = this.createPlaceholder();
      return placeholder;
    } catch (error) {
      // Unexpected error
      this.loadingQueue.delete(cacheKey);
      const placeholder = this.createPlaceholder();
      return placeholder;
    }
  }

  // Progressive loading with low-res placeholder
  async loadProgressive(
    basePath: string,
    onProgress?: (stage: 'placeholder' | 'lowres' | 'highres', texture: THREE.Texture) => void,
    baseUrl: string = ''
  ): Promise<THREE.Texture> {
    // 1. Start with placeholder
    const placeholder = this.createPlaceholder();
    onProgress?.('placeholder', placeholder);

    try {
      // 2. Try to load a low-res version (if available)
      const lowResPath = 'webp/thumbs/' + basePath.replace('.png', '_thumb.webp');
      const lowResUrl = `${baseUrl}/${lowResPath}`;
      
      // Debug occasionally
      if (Math.random() < 0.01) {
        console.log('Thumbnail URL:', { basePath, lowResPath, lowResUrl });
      }
      
      try {
        const lowResTexture = await this.loader.loadAsync(lowResUrl);
        lowResTexture.minFilter = THREE.LinearFilter;
        onProgress?.('lowres', lowResTexture);
      } catch {
        // Low-res not available, skip
      }

      // 3. Load high-res version
      const highResTexture = await this.loadWithFallback(basePath, baseUrl);
      onProgress?.('highres', highResTexture);
      
      return highResTexture;
    } catch (error) {
      console.error('Progressive loading failed:', error);
      return placeholder;
    }
  }

  // Batch load images with progress tracking
  async loadBatch(
    paths: string[], 
    batchSize = 10,
    onProgress?: (loaded: number, total: number) => void
  ): Promise<THREE.Texture[]> {
    const results: THREE.Texture[] = [];
    
    for (let i = 0; i < paths.length; i += batchSize) {
      const batch = paths.slice(i, i + batchSize);
      
      const batchPromises = batch.map(path => this.loadWithFallback(path));
      const batchResults = await Promise.allSettled(batchPromises);
      
      batchResults.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        } else {
          console.error(`Failed to load ${batch[index]}:`, result.reason);
          results.push(this.createPlaceholder());
        }
      });
      
      onProgress?.(Math.min(i + batchSize, paths.length), paths.length);
    }
    
    return results;
  }

  // Clear cache to free memory
  clearCache(): void {
    this.cache.forEach(texture => texture.dispose());
    this.cache.clear();
  }

  // Get cache size
  getCacheSize(): number {
    return this.cache.size;
  }
}