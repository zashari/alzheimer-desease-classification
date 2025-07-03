// Static list of S3 image paths
// This is generated from the S3 bucket structure you provided
// In production, you might want to generate this dynamically

export const S3_BUCKET_URL = 'https://ad-public-storage-data-viewer-ap-southeast-1-836322468413.s3.ap-southeast-1.amazonaws.com';

// Helper function to generate image file names based on patterns
function generateImagePaths(): string[] {
  const paths: string[] = [];
  
  // Pattern for image files: PatientID_VisitID_ScanType_Plane_SliceNumber.png
  // Example: 002_S_1018_m06_axial_x110.png
  
  // Common patient IDs and visit IDs from your data
  const patientPatterns = [
    '002_S_1018', '002_S_0295', '002_S_0685', '002_S_0816', '002_S_0938',
    '005_S_0221', '005_S_1341', '005_S_0814', '005_S_0223', '005_S_0553',
    '007_S_1206', '007_S_0316', '007_S_1339', '007_S_0070',
    '010_S_0829', '010_S_0786', '010_S_0419', '010_S_0472',
    '011_S_0010', '011_S_0023', '011_S_0003', '011_S_0053', '011_S_0183',
    '012_S_0637', '012_S_0689', '012_S_0712', '012_S_0720', '012_S_0803',
    '014_S_0328', '014_S_1095', '014_S_0519', '014_S_0520',
    '016_S_0359', '016_S_0538', '016_S_0991',
    '018_S_0682', '018_S_0286', '018_S_0633',
    '020_S_0213', '021_S_0343', '021_S_0642', '021_S_0753', '021_S_1109',
    '022_S_0129', '023_S_0084', '023_S_0083', '023_S_0139', '023_S_0916',
    '024_S_0985', '024_S_1171', '024_S_1307',
    '027_S_0118', '027_S_0120', '027_S_0404', '027_S_1082', '027_S_1254',
    '029_S_0836', '029_S_0999', '029_S_1056',
    '031_S_0321', '031_S_0554', '031_S_1209',
    '032_S_0400', '032_S_0095', '032_S_1101',
    '033_S_0733', '033_S_1285', '033_S_0724', '033_S_0739', '033_S_0889',
    '035_S_0341', '035_S_0048', '036_S_0577', '036_S_0759', '036_S_0760',
    '037_S_0627', '037_S_0467', '041_S_1368', '041_S_0125',
    '053_S_1044', '057_S_1373', '057_S_0474', '057_S_0818', '057_S_1371',
    '062_S_0768', '062_S_0690', '062_S_0730', '062_S_0793',
    '067_S_0177', '067_S_0029', '068_S_0210', '073_S_0565',
    '082_S_1377', '094_S_1027', '094_S_1090', '094_S_1164', '094_S_1402',
    '099_S_1144', '099_S_0040', '099_S_0352', '099_S_0372', '099_S_0470',
    '100_S_0035', '109_S_1157', '114_S_0374',
    '116_S_0657', '116_S_0370', '116_S_0392', '116_S_0487',
    '123_S_0088', '123_S_0091', '123_S_0094', '123_S_0162',
    '126_S_0680', '126_S_0784', '126_S_0891', '126_S_1221',
    '127_S_0684', '127_S_0431', '127_S_0754', '127_S_0844', '127_S_1382',
    '130_S_1201', '130_S_0956', '130_S_1290',
    '131_S_1301', '131_S_0457', '136_S_0194', '136_S_0086', '136_S_0300', '136_S_0426',
    '137_S_1041', '137_S_0366', '137_S_0796',
    '141_S_0726', '141_S_0810', '141_S_1094', '141_S_0852', '141_S_0853', '141_S_1137',
    '941_S_1202'
  ];
  
  const scanTypes = ['m06', 'm12', 'sc'];
  const planes = ['axial', 'coronal', 'sagittal'];
  const versions = ['original-images', 'enhanced-images'];
  const subsets = ['train', 'test', 'val'];
  const classes = ['AD', 'CN'];
  
  // Generate paths based on the pattern
  versions.forEach(version => {
    planes.forEach(plane => {
      subsets.forEach(subset => {
        classes.forEach(cls => {
          // For each combination, add some sample files
          // In reality, you'd have the actual file names
          patientPatterns.slice(0, 20).forEach(patient => {
            scanTypes.forEach(scan => {
              // Generate a slice number (varies by plane)
              const sliceNum = Math.floor(Math.random() * 150) + 50;
              const axis = plane === 'axial' ? 'x' : plane === 'coronal' ? 'y' : 'z';
              const filename = `${patient}_${scan}_${plane}_${axis}${sliceNum}.png`;
              const path = `assets/images/${version}/${plane}/${subset}/${cls}/${filename}`;
              paths.push(path);
            });
          });
        });
      });
    });
  });
  
  return paths;
}

// Export the generated paths
export const S3_IMAGE_PATHS = generateImagePaths();

// Helper function to get full S3 URL
export function getS3ImageUrl(path: string): string {
  return `${S3_BUCKET_URL}/${path}`;
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