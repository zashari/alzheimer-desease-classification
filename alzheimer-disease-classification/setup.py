from setuptools import setup, find_packages

setup(
    name="processing_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scikit-image>=0.21.0",
        "nibabel>=5.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "imgaug==0.4.0",
        "torch",
        "torchvision",
        "timm"
    ],
) 