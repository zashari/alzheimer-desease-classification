# Performance Optimization Guide

## Current Optimizations

### 🚀 **Performance Improvements Applied:**

1. **Level of Detail (LOD) System**
   - Maximum 500 images rendered at once (down from 5,815)
   - Batch loading: 50 images per batch
   - Distance-based culling: Only render images within 30 units

2. **CloudFront CDN**
   - Global edge locations for faster delivery
   - Automatic compression and caching
   - HTTPS by default

3. **Reduced Scene Complexity**
   - Stars reduced from 5,000 to 1,000
   - Performance monitor to track FPS and draw calls

4. **Smart Filtering**
   - Warns when showing more than 500 images
   - Progressive loading with visual feedback

## Performance Metrics

### **Expected Performance:**
- **FPS**: 30-60 (depending on device)
- **Load Time**: 2-5 seconds for first 50 images
- **Memory Usage**: 100-300MB (down from 1GB+)

### **Performance Monitor**
Check the top-left corner for real-time metrics:
- **Green FPS (50+)**: Excellent performance
- **Yellow FPS (30-49)**: Good performance  
- **Red FPS (<30)**: Consider filtering images

## CloudFront vs S3 Performance

Based on your Singapore location:

| Metric | S3 Direct | CloudFront | Notes |
|--------|-----------|------------|-------|
| Initial Load | ~170ms | ~1.3s | Cold cache penalty |
| Cached Load | ~170ms | ~670ms | Warming up |
| Global Users | Slow | Fast | CloudFront wins globally |

### **Recommendation:**
- **For Singapore users**: S3 direct might be faster initially
- **For global users**: CloudFront is significantly faster
- **Production**: Use CloudFront for global reach

## Switching Between S3 and CloudFront

Update `.env` file:

```bash
# For S3 direct (Singapore optimized):
VITE_IMAGE_BASE_URL=https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com

# For CloudFront (Global optimized):
VITE_IMAGE_BASE_URL=https://d2iiwoaj8v8tqz.cloudfront.net
```

## Performance Tips

### **For Better Performance:**

1. **Use Filters**: Instead of showing all 5,815 images
   - Filter by plane (axial/coronal/sagittal)
   - Filter by class (CN/AD)
   - Filter by subset (train/test/val)

2. **Hardware Recommendations**:
   - GPU: Dedicated graphics preferred
   - RAM: 8GB+ recommended
   - Browser: Chrome/Firefox with WebGL2 support

3. **Network Optimization**:
   - Use filters to reduce image count
   - Let images load progressively
   - CloudFront cache warms up over time

### **Performance Troubleshooting:**

| Issue | Solution |
|-------|----------|
| Low FPS (<30) | Apply filters to show fewer images |
| High memory usage | Refresh page, use specific filters |
| Slow image loading | Check network connection |
| Images not loading | Verify S3/CloudFront accessibility |

## Advanced Performance Settings

You can adjust these constants in `Scene.tsx`:

```typescript
const MAX_VISIBLE_IMAGES = 500;  // Reduce for better performance
const LOAD_BATCH_SIZE = 50;      // Reduce for smoother loading
const VISIBILITY_DISTANCE = 30;  // Reduce to show fewer images
```

## Monitoring

### **Real-time Metrics:**
- Performance monitor shows FPS, draw calls, memory
- Browser DevTools Network tab shows image loading
- Console logs show filter performance warnings

### **Optimization Success:**
✅ 5,815 → 500 max images (90% reduction)  
✅ Progressive loading prevents browser freeze  
✅ Distance culling improves framerates  
✅ CloudFront provides global scalability