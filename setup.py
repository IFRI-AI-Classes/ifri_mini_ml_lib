from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name='ifri_mini_ml_lib',
    version='0.1.0',
    description='A lightweight machine learning library built from scratch by IFRI IA students',
    author='IFRI IA Team',
    packages=find_packages(),  
    install_requires=[
        'numpy',
        'pandas'
    ],
    python_requires='>=3.7',
)
