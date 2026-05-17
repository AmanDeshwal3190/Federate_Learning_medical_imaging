from setuptools import setup, find_packages

setup(
    name="federated_medical_imaging",
    version="1.0.0",
    description="Federated Learning for Disease Detection in Medical Imaging",
    author="Shrikant, Aman Kumar, Yash Vardhan S. Parmar",
    author_email="shrikantyadav9024@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.15.0",
        "flwr>=1.7.0",
        "flask>=3.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "nibabel>=5.2.0",
        "matplotlib>=3.8.0",
        "PyYAML>=6.0.1",
    ],
)
