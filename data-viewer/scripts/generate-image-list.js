#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Read the S3 image list
const s3ImagesPath = '/tmp/s3-images.txt';
const outputPath = path.join(__dirname, '..', 'src', 'data', 's3-actual-images.ts');

if (!fs.existsSync(s3ImagesPath)) {
  console.error('Please run: aws s3 ls s3://<your-bucket-name>/assets/images/ --recursive | grep "\\.png$" | awk \'{print $4}\' > /tmp/s3-images.txt');
  process.exit(1);
}

const images = fs.readFileSync(s3ImagesPath, 'utf-8')
  .split('\n')
  .filter(line => line.trim())
  .map(line => line.trim());

console.log(`Found ${images.length} images in S3`);

// Generate TypeScript file
const tsContent = `// Auto-generated list of actual S3 images
// Generated on: ${new Date().toISOString()}

export const S3_BUCKET_URL = 'https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com';

// Actual image paths from S3
export const S3_IMAGE_PATHS = [
${images.map(img => `  '${img}'`).join(',\n')}
];

// Helper function to get full S3 URL
export function getS3ImageUrl(path: string): string {
  return \`\${S3_BUCKET_URL}/\${path}\`;
}

// Helper function to parse S3 path to metadata
export function parseS3Path(path: string) {
  // Example path: assets/images/enhanced-images/axial/test/AD/002_S_1018_m06_axial_x110.png
  const parts = path.split('/');
  
  if (parts.length >= 7) {
    const filename = parts[parts.length - 1];
    return {
      version: parts[2] as 'original-images' | 'enhanced-images',
      plane: parts[3] as 'axial' | 'coronal' | 'sagittal',
      subset: parts[4] as 'train' | 'test' | 'val',
      class: parts[5] as 'CN' | 'AD',
      filename: filename.replace('.png', ''),
      fullPath: path
    };
  }
  
  return null;
}

export const TOTAL_IMAGES = ${images.length};
`;

// Create data directory if it doesn't exist
const dataDir = path.dirname(outputPath);
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

// Write the file
fs.writeFileSync(outputPath, tsContent);
console.log(`Generated ${outputPath}`);
console.log(`Total images: ${images.length}`);