#!/usr/bin/env node

// This script generates a list of all image files in the S3 bucket
// Run this locally with AWS credentials to generate the file list

const AWS = require('aws-sdk');
const fs = require('fs');
const path = require('path');

// Configure AWS
const s3 = new AWS.S3({
  region: 'ap-southeast-1'
});

const BUCKET_NAME = 'ad-public-storage-data-viewer-ap-southeast-1-836322468413';
const OUTPUT_FILE = path.join(__dirname, '..', 'src', 'data', 's3-images.json');

async function listAllObjects() {
  const allObjects = [];
  let continuationToken = null;
  
  do {
    const params = {
      Bucket: BUCKET_NAME,
      Prefix: 'assets/images/',
      ContinuationToken: continuationToken
    };
    
    try {
      const data = await s3.listObjectsV2(params).promise();
      
      // Filter for PNG files only
      const pngFiles = data.Contents
        .filter(obj => obj.Key.endsWith('.png'))
        .map(obj => ({
          key: obj.Key,
          size: obj.Size,
          lastModified: obj.LastModified
        }));
      
      allObjects.push(...pngFiles);
      continuationToken = data.NextContinuationToken;
    } catch (error) {
      console.error('Error listing objects:', error);
      break;
    }
  } while (continuationToken);
  
  return allObjects;
}

async function generateFileList() {
  console.log('Fetching file list from S3...');
  const files = await listAllObjects();
  
  console.log(`Found ${files.length} PNG files in S3`);
  
  // Create the data directory if it doesn't exist
  const dataDir = path.dirname(OUTPUT_FILE);
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  
  // Write the file list
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(files, null, 2));
  console.log(`File list saved to ${OUTPUT_FILE}`);
}

// Run the script
generateFileList().catch(console.error);