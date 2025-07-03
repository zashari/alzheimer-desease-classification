#!/bin/bash

# Script to configure CORS on S3 bucket for web access

BUCKET_NAME="<your-bucket-name>"

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
  echo "Images can be accessed directly from: https://${BUCKET_NAME}.s3.ap-southeast-1.amazonaws.com"
else
  echo "❌ Error applying CORS configuration"
  exit 1
fi