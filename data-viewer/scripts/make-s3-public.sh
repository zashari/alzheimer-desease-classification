#!/bin/bash

# Script to make S3 bucket objects publicly readable

# Check if bucket name is provided via environment variable
if [ -z "$AWS_S3_BUCKET_NAME" ]; then
  echo "❌ Error: AWS_S3_BUCKET_NAME environment variable is not set"
  echo "Please set it with: export AWS_S3_BUCKET_NAME=your-bucket-name"
  exit 1
fi

BUCKET_NAME="$AWS_S3_BUCKET_NAME"

echo "Making S3 bucket objects publicly accessible..."

# First, create a bucket policy to allow public read access
cat > bucket-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"
        }
    ]
}
EOF

# Apply the bucket policy
echo "Applying bucket policy for public read access..."
aws s3api put-bucket-policy \
    --bucket "${BUCKET_NAME}" \
    --policy file://bucket-policy.json

if [ $? -eq 0 ]; then
    echo "✅ Bucket policy applied successfully!"
    
    # Clean up
    rm bucket-policy.json
    
    # Also update the bucket's public access block settings
    echo "Updating public access block settings..."
    aws s3api put-public-access-block \
        --bucket "${BUCKET_NAME}" \
        --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
    
    if [ $? -eq 0 ]; then
        echo "✅ Public access settings updated!"
        echo ""
        echo "Your S3 bucket objects are now publicly accessible."
        echo "CloudFront should now be able to serve the images."
    else
        echo "⚠️  Warning: Could not update public access block settings"
        echo "You may need to do this manually in the AWS Console"
    fi
else
    echo "❌ Error applying bucket policy"
    exit 1
fi