#!/bin/bash

# Test script to verify CloudFront is working and warm the cache

# Check if required environment variables are set
if [ -z "$AWS_CLOUDFRONT_DOMAIN" ]; then
  echo "❌ Error: AWS_CLOUDFRONT_DOMAIN environment variable is not set"
  echo "Please set it with: export AWS_CLOUDFRONT_DOMAIN=your-cloudfront-domain.cloudfront.net"
  exit 1
fi

if [ -z "$AWS_S3_BUCKET_NAME" ]; then
  echo "❌ Error: AWS_S3_BUCKET_NAME environment variable is not set"
  echo "Please set it with: export AWS_S3_BUCKET_NAME=your-bucket-name"
  exit 1
fi

CLOUDFRONT_URL="https://$AWS_CLOUDFRONT_DOMAIN"
S3_URL="https://${AWS_S3_BUCKET_NAME}.s3.${AWS_REGION:-ap-southeast-1}.amazonaws.com"

# Test a sample image from both URLs
SAMPLE_IMAGE="assets/images/enhanced-images/axial/test/AD/002_S_1018_m06_axial_x110.png"

echo "Testing CloudFront vs S3 performance..."
echo "==========================================="

echo ""
echo "Testing S3 direct:"
time curl -s -w "Response: %{http_code}, Time: %{time_total}s, Size: %{size_download} bytes\n" \
  -o /dev/null "${S3_URL}/${SAMPLE_IMAGE}"

echo ""
echo "Testing CloudFront:"
time curl -s -w "Response: %{http_code}, Time: %{time_total}s, Size: %{size_download} bytes\n" \
  -o /dev/null "${CLOUDFRONT_URL}/${SAMPLE_IMAGE}"

echo ""
echo "Testing CloudFront again (should be faster - cached):"
time curl -s -w "Response: %{http_code}, Time: %{time_total}s, Size: %{size_download} bytes\n" \
  -o /dev/null "${CLOUDFRONT_URL}/${SAMPLE_IMAGE}"

echo ""
echo "==========================================="
if [ -n "$AWS_CLOUDFRONT_DISTRIBUTION_ID" ]; then
  echo "CloudFront Distribution Status:"
  aws cloudfront get-distribution --id "$AWS_CLOUDFRONT_DISTRIBUTION_ID" --query 'Distribution.Status' --output text
  
  echo ""
  echo "If status is 'Deployed', CloudFront is ready."
  echo "If status is 'InProgress', wait 15-30 minutes for full deployment."
else
  echo "Set AWS_CLOUDFRONT_DISTRIBUTION_ID to check distribution status"
fi