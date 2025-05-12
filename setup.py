from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='ifri_mini_ml_lib',
    version='0.1.1',
    description='A lightweight machine learning library built from scratch by IFRI IA students',
    author='IFRI IA Students',
    packages=find_packages(),  
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'cvxpy'
    ],
    python_requires='>=3.9',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/IFRI-AI-Classes/ifri_mini_ml_lib',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="machine-learning education from-scratch ml-library ifri ai"
)
