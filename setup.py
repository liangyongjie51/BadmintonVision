from setuptools import setup, find_packages

setup(
    name="badmintonvision",
    version="1.0.0",
    description="Deep-learning framework for automated tactical analysis in elite badminton",
    packages=find_packages(include=["badmintonvision", "badmintonvision.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24", "scipy>=1.10", "pandas>=2.0", "statsmodels>=0.14",
        "matplotlib>=3.7", "PyYAML>=6.0", "opencv-python>=4.8", "Pillow>=10.0",
    ],
    extras_require={
        "dl": ["torch>=2.1", "torchvision>=0.16", "timm>=0.9.12", "ultralytics>=8.3"],
        "dev": ["pytest>=7.4"],
    },
)
