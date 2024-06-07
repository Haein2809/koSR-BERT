
from setuptools import setup, find_packages

setup(
    name='koSR-BERT',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'pandas'
    ],
    author='Haein2809',
    author_email='1995khi@gmail.com',
    description='Korean SR-BERT model implementation',
    url='https://github.com/Haein2809/koSR-BERT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
