#!/bin/bash

set -e

PROJECT_NAME="v1"

mkdir -p ${PROJECT_NAME}
cd ${PROJECT_NAME}

# Root files
touch README.md
touch LICENSE
touch pyproject.toml
touch requirements.txt
touch .gitignore
touch .env.example

# Configs
mkdir -p configs/datasets
mkdir -p configs/models
mkdir -p configs/methods
mkdir -p configs/experiments

touch configs/default.yaml

touch configs/datasets/videomme.yaml
touch configs/datasets/longvideobench.yaml

touch configs/models/qwen2_vl_7b.yaml
touch configs/models/qwen25_vl_7b.yaml
touch configs/models/clip_vit_l14.yaml

touch configs/methods/uniform.yaml
touch configs/methods/random.yaml
touch configs/methods/global_clip_topk.yaml
touch configs/methods/global_clip_diverse.yaml
touch configs/methods/event_equal.yaml
touch configs/methods/event_duration.yaml
touch configs/methods/event_relevance.yaml
touch configs/methods/event_complexity.yaml
touch configs/methods/event_complexity_relevance.yaml
touch configs/methods/event_sparse.yaml
touch configs/methods/event_minimum.yaml

touch configs/experiments/videomme_main.yaml
touch configs/experiments/videomme_ablation_scoring.yaml
touch configs/experiments/videomme_ablation_allocation.yaml
touch configs/experiments/videomme_ablation_selection.yaml
touch configs/experiments/granularity_analysis.yaml

# Data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/annotations
mkdir -p data/cache/features
mkdir -p data/cache/events
mkdir -p data/cache/selections
mkdir -p data/cache/predictions

touch data/README.md

# Scripts
mkdir -p scripts

touch scripts/prepare_videomme.py
touch scripts/extract_frames.py
touch scripts/extract_features.py
touch scripts/run_experiment.py
touch scripts/run_batch.py
touch scripts/evaluate.py
touch scripts/analyze_results.py
touch scripts/export_tables.py

# Source package
mkdir -p src/evalloc

touch src/evalloc/__init__.py

# Config package
mkdir -p src/evalloc/config
touch src/evalloc/config/__init__.py
touch src/evalloc/config/schema.py
touch src/evalloc/config/loader.py

# Data package
mkdir -p src/evalloc/data
touch src/evalloc/data/__init__.py
touch src/evalloc/data/base.py
touch src/evalloc/data/videomme.py
touch src/evalloc/data/longvideobench.py
touch src/evalloc/data/video_reader.py
touch src/evalloc/data/collator.py

# Feature extraction package
mkdir -p src/evalloc/features
touch src/evalloc/features/__init__.py
touch src/evalloc/features/base.py
touch src/evalloc/features/clip_extractor.py
touch src/evalloc/features/siglip_extractor.py
touch src/evalloc/features/video_encoder.py
touch src/evalloc/features/cache.py

# Segmentation package
mkdir -p src/evalloc/segmentation
touch src/evalloc/segmentation/__init__.py
touch src/evalloc/segmentation/base.py
touch src/evalloc/segmentation/no_segmenter.py
touch src/evalloc/segmentation/fixed_segmenter.py
touch src/evalloc/segmentation/semantic_segmenter.py
touch src/evalloc/segmentation/oracle_segmenter.py
touch src/evalloc/segmentation/boundary_utils.py

# Representation package
mkdir -p src/evalloc/representation
touch src/evalloc/representation/__init__.py
touch src/evalloc/representation/base.py
touch src/evalloc/representation/mean_pool.py
touch src/evalloc/representation/max_relevance.py
touch src/evalloc/representation/start_middle_end.py
touch src/evalloc/representation/multi_rep.py

# Scoring package
mkdir -p src/evalloc/scoring
touch src/evalloc/scoring/__init__.py
touch src/evalloc/scoring/base.py
touch src/evalloc/scoring/random.py
touch src/evalloc/scoring/duration.py
touch src/evalloc/scoring/relevance.py
touch src/evalloc/scoring/complexity.py
touch src/evalloc/scoring/combined.py
touch src/evalloc/scoring/normalization.py

# Allocation package
mkdir -p src/evalloc/allocation
touch src/evalloc/allocation/__init__.py
touch src/evalloc/allocation/base.py
touch src/evalloc/allocation/equal.py
touch src/evalloc/allocation/duration.py
touch src/evalloc/allocation/proportional.py
touch src/evalloc/allocation/softmax.py
touch src/evalloc/allocation/minimum.py
touch src/evalloc/allocation/sparse.py

# Frame selection package
mkdir -p src/evalloc/selection
touch src/evalloc/selection/__init__.py
touch src/evalloc/selection/base.py
touch src/evalloc/selection/uniform.py
touch src/evalloc/selection/random.py
touch src/evalloc/selection/relevance_topk.py
touch src/evalloc/selection/diverse.py
touch src/evalloc/selection/temporal_coverage.py

# Inference package
mkdir -p src/evalloc/inference
touch src/evalloc/inference/__init__.py
touch src/evalloc/inference/base.py
touch src/evalloc/inference/qwen2_vl.py
touch src/evalloc/inference/qwen25_vl.py
touch src/evalloc/inference/prompt_builder.py
touch src/evalloc/inference/answer_parser.py

# Evaluation package
mkdir -p src/evalloc/evaluation
touch src/evalloc/evaluation/__init__.py
touch src/evalloc/evaluation/qa_metrics.py
touch src/evalloc/evaluation/efficiency.py
touch src/evalloc/evaluation/selection_metrics.py
touch src/evalloc/evaluation/boundary_metrics.py
touch src/evalloc/evaluation/correlation.py

# Pipeline package
mkdir -p src/evalloc/pipeline
touch src/evalloc/pipeline/__init__.py
touch src/evalloc/pipeline/frame_sampler.py
touch src/evalloc/pipeline/event_pipeline.py
touch src/evalloc/pipeline/baseline_pipeline.py
touch src/evalloc/pipeline/runner.py

# Utils package
mkdir -p src/evalloc/utils
touch src/evalloc/utils/__init__.py
touch src/evalloc/utils/logging.py
touch src/evalloc/utils/io.py
touch src/evalloc/utils/seed.py
touch src/evalloc/utils/timer.py
touch src/evalloc/utils/distributed.py
touch src/evalloc/utils/registry.py

# Outputs
mkdir -p outputs/runs
mkdir -p outputs/tables
mkdir -p outputs/figures
mkdir -p outputs/logs

# Notebooks
mkdir -p notebooks
touch notebooks/result_analysis.ipynb
touch notebooks/event_visualization.ipynb
touch notebooks/failure_case_analysis.ipynb

# Tests
mkdir -p tests
touch tests/test_allocation.py
touch tests/test_selection.py
touch tests/test_segmentation.py
touch tests/test_metrics.py
touch tests/test_config.py

echo "Project structure created successfully: ${PROJECT_NAME}"
