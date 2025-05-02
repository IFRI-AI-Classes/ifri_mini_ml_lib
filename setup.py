from setuptools import setup, find_packages

setup(
    name='ifri_mini_ml_lib',
    version='0.1.0',
    description='A lightweight machine learning library built from scratch by IFRI IA students',
    author='IFRI IA Students',
    packages=find_packages(),  
    install_requires=[
        'numpy>=2.2.4',
        'pandas>=2.2.3',
        'scipy>=1.15.0',
        'matplotlib>=3.9.1'
    ],
    python_requires='>=3.7',
)
