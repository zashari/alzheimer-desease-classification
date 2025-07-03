#!/bin/bash

# Test script to verify CloudFront is working and warm the cache

CLOUDFRONT_URL="https://d2iiwoaj8v8tqz.cloudfront.net"
S3_URL="https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com"

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
echo "CloudFront Distribution Status:"
aws cloudfront get-distribution --id EN2U5K6XUPERH --query 'Distribution.Status' --output text

echo ""
echo "If status is 'Deployed', CloudFront is ready."
echo "If status is 'InProgress', wait 15-30 minutes for full deployment."