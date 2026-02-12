# PyTorch fpsample

PyTorch efficient farthest point sampling (FPS) implementation, adopted from [fpsample](https://github.com/leonardodalinky/fpsample).

**This project provides bucket-based farthest point sampling (FPS) on both CPU and GPU.** The GPU implementation is optimized for high dimension sampling (eg. model feature spaces).

## Installation

```bash
# Install from github
pip install git+https://github.com/Astro-85/pytorch_fpsample

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

# For high-dimensional embeddings on CUDA, set low_d  for faster bucketing
sampled_points, indices = torch_fpsample.sample(x, 1024, h=8, low_d=8)

# indices-only
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


## Performance comparison (CPU vs GPU vs vanilla GPU FPS)

Comparison includes CPU, our bucketed GPU implementation, and a vanilla FPS GPU baseline. Here, N = number of input points, D = point dimension, and K = number of sampled points.

```
  N      D      K     CPU ms  GPU baseline ms  GPU bucketed ms    CPU vs GPU   GPU baseline vs bucketed
--------------------------------------------------------------------------------------
  1000      8    250     0.271          0.404           2.671       0.10x          0.15x
  1000     64    250     2.718          6.117           2.908       0.93x          2.10x
  1000    256    250    13.945         23.676           4.005       3.48x          5.91x
  1000   1024    250    69.697         94.144           4.867      14.32x         19.34x
  1000   4096    250   248.521        378.458          10.614      23.41x         35.65x
  2000      8    500     1.578          1.299           5.432       0.29x          0.24x
  2000     64    500     9.644         25.658           6.555       1.47x          3.91x
  2000    256    500    39.424        100.612           8.612       4.58x         11.68x
  2000   1024    500   213.804        399.292          11.018      19.41x         36.24x
  2000   4096    500   869.318       1585.913          33.974      25.59x         46.68x
  5000      8   1250     6.151          7.156          16.970       0.36x          0.42x
  5000     64   1250    47.877        157.348          24.252       1.97x          6.49x
  5000    256   1250   318.593        626.325          29.906      10.65x         20.94x
  5000   1024   1250  1075.742       2483.299          47.459      22.67x         52.33x
  5000   4096   1250  4547.318      10027.665         154.874      29.36x         64.75x
 10000      8   2500    22.135         26.152          43.379       0.51x          0.60x
 10000     64   2500   187.709        626.442          65.269       2.88x          9.60x
 10000    256   2500  1172.482       2503.120          94.904      12.35x         26.38x
 10000   1024   2500  4503.257       9959.041         186.622      24.13x         53.36x
 10000   4096   2500 21699.598      40439.047         645.883      33.60x         62.61x
```

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

