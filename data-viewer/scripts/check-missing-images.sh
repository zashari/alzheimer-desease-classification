#!/bin/bash

# Script to check for missing images

set -e

# Check if required environment variables are set
if [ -z "$AWS_S3_BUCKET_NAME" ]; then
  echo "❌ Error: AWS_S3_BUCKET_NAME environment variable is not set"
  echo "Please set it with: export AWS_S3_BUCKET_NAME=your-bucket-name"
  exit 1
fi

if [ -z "$AWS_CLOUDFRONT_DOMAIN" ]; then
  echo "❌ Error: AWS_CLOUDFRONT_DOMAIN environment variable is not set"
  echo "Please set it with: export AWS_CLOUDFRONT_DOMAIN=your-cloudfront-domain.cloudfront.net"
  exit 1
fi

BUCKET_NAME="$AWS_S3_BUCKET_NAME"
CLOUDFRONT_URL="https://$AWS_CLOUDFRONT_DOMAIN"

echo "🔍 Checking for missing images..."

# Sample of images that were failing
FAILING_IMAGES=(
  "assets/images/enhanced-images/axial/train/CN/052_S_1251_sc_axial_x102.png"
  "assets/images/enhanced-images/axial/val/CN/009_S_0862_sc_axial_x105.png"
  "assets/images/enhanced-images/axial/train/CN/037_S_0303_m12_axial_x123.png"
  "assets/images/enhanced-images/axial/train/CN/057_S_0779_sc_axial_x69.png"
  "assets/images/enhanced-images/axial/val/CN/010_S_0067_m06_axial_x99.png"
)

echo "Testing sample of previously failing images..."

for image in "${FAILING_IMAGES[@]}"; do
  echo "Testing: $image"
  
  # Test PNG
  png_status=$(curl -s -o /dev/null -w "%{http_code}" "$CLOUDFRONT_URL/$image")
  
  # Test WebP
  webp_path="webp/${image%.png}.webp"
  webp_status=$(curl -s -o /dev/null -w "%{http_code}" "$CLOUDFRONT_URL/$webp_path")
  
  echo "  PNG: $png_status | WebP: $webp_status"
  
  if [ "$png_status" != "200" ] && [ "$webp_status" != "200" ]; then
    echo "  ❌ Both formats missing!"
  elif [ "$png_status" != "200" ]; then
    echo "  ⚠️  PNG missing"
  elif [ "$webp_status" != "200" ]; then
    echo "  ⚠️  WebP missing"
  else
    echo "  ✅ Both formats available"
  fi
  echo
done

echo "✅ Image check complete!"