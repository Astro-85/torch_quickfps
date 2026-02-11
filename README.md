# PyTorch fpsample

PyTorch efficient farthest point sampling (FPS) implementation, adopted from [fpsample](https://github.com/leonardodalinky/fpsample).

**This project provides bucket-based farthest point sampling (FPS) on both CPU and CUDA GPU.**

## Installation

```bash
# Install from github
pip install git+https://github.com/leonardodalinky/pytorch_fpsample

# Build locally
pip install .
```

## Usage

```python
import torch_fpsample

x = torch.rand(64, 2048, 3)
# random sample
sampled_points, indices = torch_fpsample.sample(x, 1024)
# random sample with specific tree height
sampled_points, indices = torch_fpsample.sample(x, 1024, h=3)
# random sample with start point index (int)
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)

# For high-dimensional embeddings on CUDA, set low_d (e.g. 3 or 8) for faster bucketing
sampled_points, indices = torch_fpsample.sample(x, 1024, h=8, low_d=8)

# indices-only (avoids gather; fastest path)
indices = torch_fpsample.sample(x, 1024, return_points=False)
# or equivalently:
indices = torch_fpsample.sample_idx(x, 1024)

# masked sample: only sample from valid points (mask shape [B, N])
mask = torch.ones(x.shape[:-1], dtype=torch.bool)
mask[:, 1000:] = False  # e.g. padding
sampled_points, indices = torch_fpsample.sample(x, 512, mask=mask)

> sampled_points.size(), indices.size()
Size([64, 1024, 3]), Size([64, 1024])
```

### CUDA support

This package builds a CUDA extension when a CUDA toolchain is available (i.e., `nvcc` / `CUDA_HOME`).

- Force CUDA build: `WITH_CUDA=1 pip install .`
- Force CPU-only build: `WITH_CUDA=0 pip install .`

On GPU, the sampler uses **bucket-based pruning** with an **adaptive KD-tree bucketing** (height `h`) in a bucketing space of dimension `p`:

- If `low_d` is not set, `p = D` (full input dimension) and bucketing is done directly on `x` (cast to float32).
- If `low_d` is set, `p = min(D, low_d)` and a cheap Rademacher projection is used to build a non-expansive bucketing space. Two independent projections are computed to get a tighter (still safe) pruning bound. Current CUDA limit: `p <= 64`.

GPU pipeline overview:

1. **Bucketing space**: build `coords_f32` (and optional `coords2_f32`).
2. **Adaptive KD-tree build on GPU** (level-by-level): compute per-node bbox + sums, choose split dim by widest extent, split at mean, and update node ids until leaves.
3. **Bucket CSR**: compute `bucket_count`, then `bucket_offsets` + `bucket_indices` for contiguous bucket membership.
4. **FPS loop**:
  - Initialize min distances and per-bucket best candidates.
  - Each iteration selects the next farthest point and marks active buckets via bbox bounds.
  - Per-active-leaf update uses warp-coalesced L2 with early-exit to minimize unnecessary work.

This avoids GPU↔CPU syncs and keeps the update path fully parallel on GPU.

## Reference
Bucket-based farthest point sampling (QuickFPS) is proposed in the following paper. The implementation is based on the author's Repo ([CPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU) & [GPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_GPU)).
```bibtex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

Thanks to the authors for their great works.

