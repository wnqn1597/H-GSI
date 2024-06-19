# Horizon-wise Gene Splicing Identification (H-GSI)

This repository provides the inference code for paper "Horizon-wise Learning Paradigm Promotes Gene Splicing Identification".

## Requirements

- `scikit-learn>=1.0.2`
- `torch>=2.0.0`
- `transformers>=4.37.2`
- `datasets>=2.18.0`
- `safetensors>=0.4.2`

## Dataset

The source files of the dataset are uploaded to the [huggingface repository](https://huggingface.co/datasets/beqjal/Human-splicing-variants). Download these files and place them in `human` folder.

## Model Files

The model parameter files are uploaded to the [huggingface repository](https://huggingface.co/beqjal/H-GSI). Download these files and place them in `models` folder.

## Usage

Inference code can be run via the command:

```bash
python [-m torch.distributed.run] [--nproc_per_node 8] inference.py \
	--model {gru|lstm|fcn|tcn|transformer} \
	--split {test|train}
```

Required parameters:

- `-m` and `--nproc_per_node`: Options for Distributed Data Parallel (DDP).
- `--model`: Architecture of the model.
- `--split`: Training test or test set for inference.
