#!/bin/bash

# Script to convert PNG images to WebP format for better performance
# Requires: AWS CLI, cwebp (WebP encoder)

set -e

BUCKET_NAME="<your-bucket-name>"
LOCAL_DIR="temp-images"
WEBP_DIR="webp-images"

echo "🚀 Starting PNG to WebP conversion process..."

# Check if cwebp is installed
if ! command -v cwebp &> /dev/null; then
    echo "❌ cwebp not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install webp
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install webp
    else
        echo "Please install webp tools manually"
        exit 1
    fi
fi

# Create directories
mkdir -p "$LOCAL_DIR" "$WEBP_DIR"

echo "📥 Downloading sample images from S3..."
# Download first 100 images for testing
aws s3 sync "s3://$BUCKET_NAME/" "$LOCAL_DIR" --exclude "*" --include "*.png" --cli-read-timeout 300 | head -100

echo "🔄 Converting PNG to WebP..."
find "$LOCAL_DIR" -name "*.png" -type f | while read -r file; do
    # Get relative path
    rel_path=${file#$LOCAL_DIR/}
    webp_path="$WEBP_DIR/${rel_path%.png}.webp"
    
    # Create directory structure
    mkdir -p "$(dirname "$webp_path")"
    
    # Convert to WebP with quality 85 (good balance of size/quality)
    cwebp -q 85 "$file" -o "$webp_path"
    
    # Get file sizes for comparison
    original_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    webp_size=$(stat -f%z "$webp_path" 2>/dev/null || stat -c%s "$webp_path")
    savings=$((100 - (webp_size * 100 / original_size)))
    
    echo "✅ $rel_path: ${savings}% smaller"
done

echo "📤 Uploading WebP images to S3..."
aws s3 sync "$WEBP_DIR" "s3://$BUCKET_NAME/webp/" --content-type "image/webp"

echo "🧹 Cleaning up temporary files..."
rm -rf "$LOCAL_DIR" "$WEBP_DIR"

echo "✅ WebP conversion complete!"
echo "💡 Update your code to use .webp extensions and fallback to .png"