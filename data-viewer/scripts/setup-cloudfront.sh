#!/bin/bash

# CloudFront setup script for S3 bucket
# This script creates a CloudFront distribution for faster image delivery

# Check if required environment variables are set
if [ -z "$AWS_S3_BUCKET_NAME" ]; then
  echo "❌ Error: AWS_S3_BUCKET_NAME environment variable is not set"
  echo "Please set it with: export AWS_S3_BUCKET_NAME=your-bucket-name"
  exit 1
fi

BUCKET_NAME="$AWS_S3_BUCKET_NAME"
REGION="${AWS_REGION:-ap-southeast-1}"
DISTRIBUTION_COMMENT="AD Medical Images CDN"
ORIGIN_ID="S3-${BUCKET_NAME}"

echo "Creating CloudFront distribution for bucket: ${BUCKET_NAME}"

# Create the CloudFront distribution
DISTRIBUTION_CONFIG=$(cat <<EOF
{
  "CallerReference": "$(date +%s)",
  "Comment": "${DISTRIBUTION_COMMENT}",
  "DefaultRootObject": "",
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "${ORIGIN_ID}",
        "DomainName": "${BUCKET_NAME}.s3.${REGION}.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        },
        "ConnectionAttempts": 3,
        "ConnectionTimeout": 10
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "${ORIGIN_ID}",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {
      "Quantity": 2,
      "Items": ["GET", "HEAD"],
      "CachedMethods": {
        "Quantity": 2,
        "Items": ["GET", "HEAD"]
      }
    },
    "ForwardedValues": {
      "QueryString": false,
      "Cookies": {
        "Forward": "none"
      },
      "Headers": {
        "Quantity": 0
      }
    },
    "TrustedSigners": {
      "Enabled": false,
      "Quantity": 0
    },
    "MinTTL": 0,
    "DefaultTTL": 86400,
    "MaxTTL": 31536000,
    "Compress": true
  },
  "CacheBehaviors": {
    "Quantity": 0
  },
  "CustomErrorResponses": {
    "Quantity": 0
  },
  "Enabled": true,
  "PriceClass": "PriceClass_100",
  "HttpVersion": "http2"
}
EOF
)

# Create the distribution
echo "Creating CloudFront distribution..."
DISTRIBUTION_OUTPUT=$(aws cloudfront create-distribution \
  --distribution-config "$DISTRIBUTION_CONFIG" \
  --output json)

if [ $? -eq 0 ]; then
  DISTRIBUTION_ID=$(echo "$DISTRIBUTION_OUTPUT" | jq -r '.Distribution.Id')
  DOMAIN_NAME=$(echo "$DISTRIBUTION_OUTPUT" | jq -r '.Distribution.DomainName')
  
  echo "CloudFront distribution created successfully!"
  echo "Distribution ID: $DISTRIBUTION_ID"
  echo "CloudFront Domain: $DOMAIN_NAME"
  
  # Save the CloudFront configuration
  cat > cloudfront-config.json <<EOF
{
  "distributionId": "$DISTRIBUTION_ID",
  "domainName": "$DOMAIN_NAME",
  "bucketName": "$BUCKET_NAME",
  "region": "$REGION",
  "createdAt": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
  
  echo "Configuration saved to cloudfront-config.json"
  echo ""
  echo "⚠️  Note: It may take 15-30 minutes for the distribution to be fully deployed."
  echo ""
  echo "Next steps:"
  echo "1. Update your application to use CloudFront URL: https://${DOMAIN_NAME}"
  echo "2. Make sure your S3 bucket has proper CORS configuration"
  echo "3. Consider adding custom domain name for production use"
  
else
  echo "Error creating CloudFront distribution"
  exit 1
fi