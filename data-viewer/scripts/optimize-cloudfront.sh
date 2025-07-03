#!/bin/bash

# CloudFront optimization script
# Configures compression, caching, and Origin Shield

set -e

DISTRIBUTION_ID="<your-distribution-id>"
ORIGIN_REGION="<your-origin-region>"

echo "🚀 Optimizing CloudFront distribution: $DISTRIBUTION_ID"

# Get current distribution config
echo "📥 Getting current CloudFront configuration..."
aws cloudfront get-distribution-config --id $DISTRIBUTION_ID > current-config.json

# Extract the current config and ETag
cat current-config.json | jq '.DistributionConfig' > distribution-config.json
ETAG=$(cat current-config.json | jq -r '.ETag')

echo "Current ETag: $ETAG"

# Create optimized configuration
echo "🔧 Creating optimized configuration..."

# Update the distribution config with optimizations
cat distribution-config.json | jq '
.DefaultCacheBehavior.Compress = true |
.DefaultCacheBehavior.DefaultTTL = 31536000 |
.DefaultCacheBehavior.MaxTTL = 31536000 |
.DefaultCacheBehavior.MinTTL = 0 |
.Comment = "Optimized for fast image delivery with compression and long caching" |
.Origins.Items[0].OriginShield = {
  "Enabled": true,
  "OriginShieldRegion": "'$ORIGIN_REGION'"
}
' > optimized-config.json

echo "📤 Updating CloudFront distribution..."
aws cloudfront update-distribution \
    --id $DISTRIBUTION_ID \
    --distribution-config file://optimized-config.json \
    --if-match $ETAG

echo "⏳ Waiting for deployment to complete..."
aws cloudfront wait distribution-deployed --id $DISTRIBUTION_ID

# Create additional cache behaviors for different file types
echo "🎯 Creating cache behavior for WebP images..."

# Get the updated config for adding cache behaviors
aws cloudfront get-distribution-config --id $DISTRIBUTION_ID > updated-config.json
ETAG=$(cat updated-config.json | jq -r '.ETag')

# Add cache behavior for WebP images
cat updated-config.json | jq '.DistributionConfig.CacheBehaviors.Items += [{
  "PathPattern": "webp/*",
  "TargetOriginId": .DistributionConfig.Origins.Items[0].Id,
  "ViewerProtocolPolicy": "redirect-to-https",
  "Compress": true,
  "DefaultTTL": 31536000,
  "MaxTTL": 31536000,
  "MinTTL": 31536000,
  "ForwardedValues": {
    "QueryString": false,
    "Cookies": {"Forward": "none"},
    "Headers": {
      "Quantity": 1,
      "Items": ["Accept"]
    }
  },
  "AllowedMethods": {
    "Quantity": 2,
    "Items": ["GET", "HEAD"]
  }
}] | .DistributionConfig.CacheBehaviors.Quantity += 1' > webp-config.json

echo "📤 Adding WebP cache behavior..."
aws cloudfront update-distribution \
    --id $DISTRIBUTION_ID \
    --distribution-config file://webp-config.json \
    --if-match $ETAG

echo "⏳ Waiting for final deployment..."
aws cloudfront wait distribution-deployed --id $DISTRIBUTION_ID

# Clean up temporary files
rm -f current-config.json distribution-config.json optimized-config.json updated-config.json webp-config.json

echo "✅ CloudFront optimization complete!"
echo ""
echo "📊 Applied optimizations:"
echo "  ✅ Gzip/Brotli compression enabled"
echo "  ✅ Cache headers set to 1 year (31,536,000 seconds)"
echo "  ✅ Origin Shield enabled in $ORIGIN_REGION"
echo "  ✅ WebP-specific cache behavior added"
echo "  ✅ Accept header forwarding for content negotiation"
echo ""
echo "🔗 Distribution URL: https://d2iiwoaj8v8tqz.cloudfront.net"
echo ""
echo "💡 Testing recommendations:"
echo "  • Test WebP fallback: https://d2iiwoaj8v8tqz.cloudfront.net/webp/[image-path].webp"
echo "  • Check compression: curl -H \"Accept-Encoding: gzip\" [url] -v"
echo "  • Verify caching: Check Cache-Control headers"