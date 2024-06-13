from setuptools import setup, find_packages

setup(
    
    name='STProfiler',
    version='0.0.2',
    long_description='Spatial Transcriptomics Profiler (STProfiler) is a python package for researchers to interpret spatial transcriptomics in single-cell level, \
                      by extracting spatial transcriptomic features from smFISH images. \
                      Package includes image analysis pipeline, spatial transcriptomic profile extraction pipeline and machine learning pipeline.',

    author='Garrick Chang',
    author_email='changg@iu.edu',

    # packages=['STProfiler'],
    packages=find_packages(),

    install_requires=[
                     'numpy==1.24.4',
                     'scipy==1.10.1',
                     'seaborn==0.13.2',
                     'matplotlib==3.7.5',
                     'pandas==2.0.3',
                     'scikit-learn==1.3.2',
                     'scikit-image==0.21.0',
                     'imbalanced-learn==0.12.3',
                     'tqdm==4.66.4',
                     'segment-anything==1.0',
                     'pillow==10.3.0'
                     ],
)