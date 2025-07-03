# S3 and CloudFront Setup Guide

## Overview

This application loads medical images from AWS S3 through CloudFront CDN for optimal performance.

## Current Configuration

### S3 Bucket
- **Bucket Name**: `ad-public-storage-data-viewer-ap-southeast-1-836322468413`
- **Region**: `ap-southeast-1` (Singapore)
- **Direct URL**: `https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com`

### CloudFront Distribution
- **Distribution ID**: `EN2U5K6XUPERH`
- **Domain**: `d2iiwoaj8v8tqz.cloudfront.net`
- **Status**: Active (may take 15-30 minutes to fully deploy)

## Image Structure

```
assets/images/
├── original-images/
│   ├── axial/
│   │   ├── train/
│   │   │   ├── AD/
│   │   │   └── CN/
│   │   ├── test/
│   │   │   ├── AD/
│   │   │   └── CN/
│   │   └── val/
│   │       ├── AD/
│   │       └── CN/
│   ├── coronal/
│   │   └── ... (same structure)
│   └── sagittal/
│       └── ... (same structure)
└── enhanced-images/
    └── ... (same structure as original-images)
```

## Configuration

### Environment Variables

Set the image base URL in `.env`:

```bash
# For CloudFront (recommended - faster):
VITE_IMAGE_BASE_URL=https://<your-cloudfront-domain>

# For S3 direct access (fallback):
# VITE_IMAGE_BASE_URL=https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com
```

## AWS CLI Commands

### Upload new images to S3:
```bash
aws s3 cp local-image.png s3://ad-public-storage-data-viewer-ap-southeast-1-836322468413/assets/images/path/to/image.png
```

### Sync entire directory:
```bash
aws s3 sync ./local-images s3://ad-public-storage-data-viewer-ap-southeast-1-836322468413/assets/images/
```

### List bucket contents:
```bash
aws s3 ls s3://ad-public-storage-data-viewer-ap-southeast-1-836322468413/assets/images/ --recursive
```

## Updating Image List

The application uses a generated list of image paths in `src/data/s3-image-list.ts`.

To update with actual S3 contents:

1. Run the file list generator (requires AWS credentials):
   ```bash
   node scripts/generate-s3-file-list.js
   ```

2. Update `src/data/s3-image-list.ts` with the actual file paths

## CloudFront Benefits

1. **Global Edge Locations**: Images cached at 400+ edge locations worldwide
2. **Reduced Latency**: ~50-80% faster load times compared to S3 direct
3. **Cost Effective**: Reduced S3 bandwidth costs
4. **HTTPS by Default**: Secure delivery included
5. **Compression**: Automatic gzip compression for faster transfers

## Monitoring

### CloudFront Metrics
```bash
aws cloudfront get-distribution --id EN2U5K6XUPERH
```

### Invalidate Cache (if needed)
```bash
aws cloudfront create-invalidation --distribution-id EN2U5K6XUPERH --paths "/*"
```

## Troubleshooting

1. **Images not loading**: Check browser console for CORS errors
2. **Slow initial load**: CloudFront may take 15-30 minutes to fully deploy
3. **404 errors**: Verify the image exists in S3 and path is correct
4. **CORS issues**: Run `./scripts/setup-s3-cors.sh` to update CORS configuration

## Security Notes

- S3 bucket has CORS enabled for web access
- CloudFront provides HTTPS encryption
- Consider adding custom domain with SSL certificate for production
- Review IAM permissions for bucket access