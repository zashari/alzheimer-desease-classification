#!/bin/bash

# Environment Setup Helper Script
# This script helps you configure environment variables for the data-viewer application

set -e

echo "🔧 Data Viewer Environment Setup"
echo "=================================="
echo ""
echo "This script will help you set up the required environment variables."
echo "You can either:"
echo "1. Set them in your shell profile (persistent)"
echo "2. Create a .env file (project-specific)"
echo ""

# Check if .env already exists
if [ -f "../.env" ]; then
  echo "⚠️  Warning: .env file already exists"
  echo "Do you want to:"
  echo "1. Update existing .env file"
  echo "2. Create a backup and start fresh"
  echo "3. Cancel"
  read -p "Choose option (1-3): " env_choice
  
  case $env_choice in
    1) echo "Updating existing .env file..." ;;
    2) 
      cp ../.env ../.env.backup.$(date +%Y%m%d_%H%M%S)
      echo "Backup created: .env.backup.$(date +%Y%m%d_%H%M%S)"
      ;;
    3) echo "Setup cancelled."; exit 0 ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
  esac
fi

echo ""
echo "📝 Please provide the following information:"
echo ""

# Collect AWS S3 Bucket Name
read -p "AWS S3 Bucket Name: " bucket_name
if [ -z "$bucket_name" ]; then
  echo "❌ Bucket name is required"
  exit 1
fi

# Collect AWS Region
read -p "AWS Region (default: ap-southeast-1): " aws_region
aws_region=${aws_region:-ap-southeast-1}

# Collect CloudFront Domain
read -p "CloudFront Domain (e.g., d1234567890.cloudfront.net): " cloudfront_domain
if [ -z "$cloudfront_domain" ]; then
  echo "⚠️  CloudFront domain not provided. Using S3 direct URL (slower)."
  image_base_url="https://${bucket_name}.s3.${aws_region}.amazonaws.com"
else
  image_base_url="https://${cloudfront_domain}"
fi

# Optional: CloudFront Distribution ID
read -p "CloudFront Distribution ID (optional): " distribution_id

echo ""
echo "🎯 Configuration Summary:"
echo "========================"
echo "S3 Bucket: $bucket_name"
echo "AWS Region: $aws_region"
echo "Image Base URL: $image_base_url"
[ -n "$cloudfront_domain" ] && echo "CloudFront Domain: $cloudfront_domain"
[ -n "$distribution_id" ] && echo "Distribution ID: $distribution_id"
echo ""

read -p "Is this correct? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
  echo "Setup cancelled."
  exit 0
fi

# Create .env file
echo ""
echo "📄 Creating .env file..."

cat > ../.env <<EOF
# Data Viewer Environment Configuration
# Generated on: $(date)

# Application Configuration
VITE_IMAGE_BASE_URL=$image_base_url

# AWS Configuration
AWS_S3_BUCKET_NAME=$bucket_name
AWS_REGION=$aws_region
EOF

# Add CloudFront specific variables if provided
if [ -n "$cloudfront_domain" ]; then
  cat >> ../.env <<EOF
AWS_CLOUDFRONT_DOMAIN=$cloudfront_domain
EOF
fi

if [ -n "$distribution_id" ]; then
  cat >> ../.env <<EOF
AWS_CLOUDFRONT_DISTRIBUTION_ID=$distribution_id
EOF
fi

echo ""
echo "✅ Environment configuration complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure AWS CLI: aws configure"
echo "2. Test the application: cd .. && npm run dev"
echo "3. Run deployment scripts from this directory as needed"
echo ""
echo "🔒 Security reminder:"
echo "- The .env file is excluded from git commits"
echo "- Never share your .env file or commit it to version control"
echo "- Use AWS IAM roles with minimal required permissions"
echo ""
echo "For more information, see .env.example"