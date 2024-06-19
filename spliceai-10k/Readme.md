# Baseline: SpliceAI-10k

SpliceAI is proposed in [Jaganathan *et al*, Cell 2019 in press](https://doi.org/10.1016/j.cell.2018.12.015). This is our implement in PyTorch.

## Usage

```bash
python [-m torch.distributed.run] [--nproc_per_node 8] spliceai_baseline.py --split {test|train}
```