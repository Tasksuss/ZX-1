# Packup

This folder collects the main workflow code from the `PCA&LDA` and `Simulation` projects in a compact, portable structure.

## Folder Structure

### `PCA_LDA_pipeline/`
Core scripts for the throat-video preprocessing and PCA/LDA analysis workflow.

- `01_build_folder_references.py`
  Build folder-level grayscale reference images and compute dataset-level brightness statistics.
- `02_pick_folder_centers.py`
  Manually annotate the throat center on each reference image.
- `03_preprocess_dataset.py`
  Apply illumination correction, brightness normalization, and ROI cropping/resizing.
- `04_split_full_crop.py`
  Split processed videos into `full` and `crop` branches for downstream analysis.
- `05_analyze_processed_split2.py`
  Extract standardized feature representations, run PCA/LDA, generate plots, and evaluate linear-SVM separability.
- `LDA_method_notes.md`
  Method note summarizing the LDA formulation and interpretation used in the project.

### `Simulation_pipeline/`
Core scripts for synthetic-data generation, dataset profiling, model training, result aggregation, and report generation.

- `config.json`
  Main configuration for the size-study pipeline.
- `config_phase2.json`
  Configuration for the phase-2 search / refinement pipeline.
- `pipeline_utils.py`
  Shared utilities for config loading, dataset iteration, frame reading, metadata extraction, and file export.
- `generate_real_style_synthetic.py`
  Generate synthetic throat-video data with appearance and motion patterns inspired by the real dataset.
- `analyze_real_dataset.py`
  Profile the real dataset and compute summary statistics and reference outputs.
- `run_size_study.py`
  Main experiment runner for the synthetic size-study workflow.
- `run_phase2_search.py`
  Run the second-stage model search / refinement experiments.
- `aggregate_study_results.py`
  Aggregate repeated experiment outputs into summary metrics and plots.
- `generate_paper_report.py`
  Build paper-ready summary tables and figures from phase-1 experiments.
- `generate_integrated_final_report.py`
  Combine outputs from multiple stages into a final integrated report.

## Suggested Reading Order

If you want to understand the pipelines quickly:

1. Read this README.
2. For the throat-image analysis workflow, follow `PCA_LDA_pipeline/01 -> 05`.
3. For the simulation workflow, start with `Simulation_pipeline/config*.json`, then read `pipeline_utils.py`, `generate_real_style_synthetic.py`, `run_size_study.py`, and finally the aggregation/report scripts.

## Notes

- This packup intentionally keeps only the main workflow code and key configuration files.
- Large outputs, intermediate results, dashboard scripts, notebooks, and figure-only helper scripts are not included here.
