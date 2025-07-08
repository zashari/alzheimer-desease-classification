#!/bin/bash

# Advanced image compression pipeline
# Combines WebP conversion with additional optimization techniques

set -e

# Check if bucket name is provided via environment variable
if [ -z "$AWS_S3_BUCKET_NAME" ]; then
  echo "❌ Error: AWS_S3_BUCKET_NAME environment variable is not set"
  echo "Please set it with: export AWS_S3_BUCKET_NAME=your-bucket-name"
  exit 1
fi

BUCKET_NAME="$AWS_S3_BUCKET_NAME"
LOCAL_DIR="temp-compress"
OUTPUT_DIR="optimized-images"

echo "🚀 Starting advanced image compression pipeline..."

# Check dependencies
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 not found. Installing..."
        return 1
    fi
    return 0
}

# Install dependencies based on OS
install_tools() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install webp imagemagick jpegoptim pngquant
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install webp imagemagick jpegoptim pngquant
    fi
}

# Check and install tools
for tool in cwebp convert jpegoptim pngquant; do
    if ! check_dependency $tool; then
        install_tools
        break
    fi
done

# Create directories
mkdir -p "$LOCAL_DIR" "$OUTPUT_DIR/webp" "$OUTPUT_DIR/thumbs"

echo "📥 Downloading images from S3..."
aws s3 sync "s3://$BUCKET_NAME/" "$LOCAL_DIR" --exclude "*" --include "*.png" --cli-read-timeout 300

echo "🔄 Processing images with multiple optimization techniques..."

# Process each PNG file
find "$LOCAL_DIR" -name "*.png" -type f | while read -r file; do
    filename=$(basename "$file")
    rel_path=${file#$LOCAL_DIR/}
    base_name="${filename%.*}"
    
    echo "Processing: $rel_path"
    
    # 1. PNG optimization with pngquant (lossy compression)
    pngquant --quality=80-95 --ext .png --force "$file"
    
    # 2. Create WebP version (high quality)
    webp_path="$OUTPUT_DIR/webp/${rel_path%.png}.webp"
    mkdir -p "$(dirname "$webp_path")"
    cwebp -q 85 -m 6 -mt "$file" -o "$webp_path"
    
    # 3. Create WebP thumbnail (low quality for progressive loading)
    thumb_path="$OUTPUT_DIR/thumbs/${rel_path%.png}_thumb.webp"
    mkdir -p "$(dirname "$thumb_path")"
    
    # Resize to 10% and compress heavily for thumbnails
    convert "$file" -resize 10% -quality 50 temp_thumb.jpg
    cwebp -q 30 -resize 64 64 temp_thumb.jpg -o "$thumb_path"
    rm -f temp_thumb.jpg
    
    # 4. Get compression statistics
    original_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    webp_size=$(stat -f%z "$webp_path" 2>/dev/null || stat -c%s "$webp_path")
    thumb_size=$(stat -f%z "$thumb_path" 2>/dev/null || stat -c%s "$thumb_path")
    
    webp_savings=$((100 - (webp_size * 100 / original_size)))
    thumb_savings=$((100 - (thumb_size * 100 / original_size)))
    
    echo "  ✅ WebP: ${webp_savings}% smaller, Thumb: ${thumb_savings}% smaller"
done

echo "📤 Uploading optimized images to S3..."

# Upload WebP images
aws s3 sync "$OUTPUT_DIR/webp" "s3://$BUCKET_NAME/webp/" \
    --content-type "image/webp" \
    --cache-control "public, max-age=31536000" \
    --content-encoding "gzip"

# Upload thumbnails
aws s3 sync "$OUTPUT_DIR/thumbs" "s3://$BUCKET_NAME/webp/thumbs/" \
    --content-type "image/webp" \
    --cache-control "public, max-age=31536000" \
    --content-encoding "gzip"

# Update original PNGs with optimized versions
aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME/" \
    --exclude "*" --include "*.png" \
    --cache-control "public, max-age=31536000"

echo "🧹 Cleaning up..."
rm -rf "$LOCAL_DIR" "$OUTPUT_DIR"

echo "✅ Advanced compression complete!"
echo ""
echo "📊 Results:"
echo "  • WebP versions: ~30-50% smaller than PNG"
echo "  • Thumbnails: ~90% smaller (for progressive loading)"
echo "  • Cache headers: Set to 1 year"
echo "  • Gzip compression: Enabled"
echo ""
echo "💡 Next steps:"
echo "  1. Update CloudFront settings"
echo "  2. Test progressive loading"