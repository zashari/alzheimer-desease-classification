#!/bin/bash

# Script to configure CORS on S3 bucket for web access

# Check if bucket name is provided via environment variable
if [ -z "$AWS_S3_BUCKET_NAME" ]; then
  echo "❌ Error: AWS_S3_BUCKET_NAME environment variable is not set"
  echo "Please set it with: export AWS_S3_BUCKET_NAME=your-bucket-name"
  exit 1
fi

BUCKET_NAME="$AWS_S3_BUCKET_NAME"

echo "Setting up CORS configuration for bucket: ${BUCKET_NAME}"

# Create CORS configuration
cat > cors-config.json <<EOF
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedOrigins": ["*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 3000
    }
  ]
}
EOF

# Apply CORS configuration to bucket
echo "Applying CORS configuration..."
aws s3api put-bucket-cors \
  --bucket "${BUCKET_NAME}" \
  --cors-configuration file://cors-config.json

if [ $? -eq 0 ]; then
  echo "✅ CORS configuration applied successfully!"
  
  # Clean up
  rm cors-config.json
  
  echo ""
  echo "Your S3 bucket is now configured to allow web access from any origin."
  echo "Images can be accessed directly from: https://${BUCKET_NAME}.s3.${AWS_REGION:-ap-southeast-1}.amazonaws.com"
else
  echo "❌ Error applying CORS configuration"
  exit 1
fi