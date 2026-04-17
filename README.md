# ZX-1

ZX-1 is a Year 3 Bioengineering project repository for sensor-based and image-based throat-motion analysis. The repository contains code for data acquisition, signal processing, imaging analysis, machine-learning experiments, and a separate pre-research package used to explore PCA/LDA analysis and simulation workflows.

## Project Overview

The repository combines several connected strands of work:

- acquisition of raw ZX-1 sensor data through serial logging and controlled collection scripts
- MATLAB-based signal processing, demodulation, inspection, and map-position analysis
- image and video analysis of throat-region recordings, including dimensionality reduction and clip-level correlation analysis
- machine-learning notebooks for classification and exploratory modeling
- an archived pre-research package containing a complete preprocessing and PCA/LDA analysis pipeline, together with a synthetic-data simulation workflow

## Repository Structure

### `data_acquisition/`
MATLAB scripts for recording raw data from the ZX-1 setup.

- `Control_data_collection.m`: controlled multi-condition data collection workflow
- `Serial_logging.m`: general serial logging script
- `Serial_logging_RedIR.m`: alternating red/IR logging workflow
- `Serial_logging_single.m`: single-channel DC acquisition script

### `signal_analysis/`
MATLAB scripts for post-acquisition signal processing and exploratory analysis.

- demodulation and threshold-based signal checks
- moving-average and plotting utilities for recorded signals
- ICA-based exploratory analysis
- map-position analysis and file-level feature extraction

### `imaging_analysis/`
Notebooks and scripts for throat-image and video-based analysis.

- `Image_analysis.ipynb`: image/video inspection workflow
- `t-SNE&UMAP analysis.ipynb`: dimensionality reduction and visualization experiments
- `video_correlation.m`: clip-level or mode-level correlation analysis

### `ml_experiments/`
Machine-learning notebooks for modeling and classification.

- `ZX-1MLmodels.ipynb`: notebook for video classification model development and related experiments

### `pre_research/`
A self-contained pre-research package preserved as a separate module.

- `PCA_LDA_pipeline/`: preprocessing and PCA/LDA analysis scripts for throat-video datasets
- `Simulation_pipeline/`: synthetic-data generation, training, aggregation, and report-generation scripts
- `README.md`: overview of the packaged pre-research code

### `docs/`
Supporting reference material used during the project.

- article highlights PDFs
- project mind maps and related reading material

## Recommended Reading Order

If you are new to the repository, a good reading order is:

1. Read this `README.md` for the overall project structure.
2. Start with `data_acquisition/` to understand how raw ZX-1 data were recorded.
3. Continue with `signal_analysis/` for the MATLAB analysis workflow.
4. Read `imaging_analysis/` for the image/video analysis branch.
5. Open `ml_experiments/` for classification and modeling work.
6. Use `pre_research/` as a separate, deeper reference for the PCA/LDA and simulation pipelines.

## Software Environment

This repository mixes multiple environments:

- **MATLAB**: most acquisition and signal-analysis scripts
- **Python / Jupyter notebooks**: imaging analysis, modeling, and pre-research workflows
- **Mixed project structure**: some subprojects are exploratory notebooks, while others are more pipeline-oriented scripts

Because of this, the repository is organized by workflow stage rather than by file type.

## Notes

- The root-level folders are intended to separate acquisition, signal processing, imaging analysis, modeling, and archived pre-research code.
- The `pre_research/` folder should be treated as a standalone reference package rather than as the main execution entrypoint for the whole repository.
- Supporting PDFs in `docs/` are included for project context and literature tracking, not as executable components.
