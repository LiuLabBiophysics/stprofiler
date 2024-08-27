# STProfiler

Spatial transcriptomics profiler (STProfiler) is a package dedicated for extracting subcellular spatial transcriptomics feautres from smFISH images. 

## Description

STProfiler includes image analysis, feature extraction and cell classification.

STProfiler is currently under development

## Getting Started

### Dependencies

STProfiler requires Python 3.8 orr newer. Addtional package dependencies include:

* numpy (>= 1.24.4)
* scipy (>= 1.10.1)
* matplotlib (>= 3.7.5)
* seaborn (>= 0.13.2)
* pandas (>= 2.0.3)
* scikit-learn (>= 1.3.2)
* scikit-image (>= 0.21.0)
* imbalanced-learn (>= 0.12.3)
* tqdm (>= 4.66.4)
* pillow (>= 10.3.0)

For cell segmentation purposes, [Segment-Anything](https://github.com/facebookresearch/segment-anything) is required. Please refer to its github page for additional package dependencies.

### Installing

Create a new conda environment

```
conda create -n stp_env python=3.8
conda activate stp_env
```

STProfiler is currently available on Test-PyPI


## License

This project is licensed under the MIT License - see the LICENSE.md file for details

