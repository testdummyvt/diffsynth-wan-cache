# Introduction

I tried to implement the first block cache optimization for the Wan2.1 models. The optimization is based [ParaAttention](https://github.com/chengzeyi/ParaAttention) repository.

# Installation

Follow installation instructions from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

# Usage

```python
from fbcache import fbcache_forward
# ...
# load models and initialize the pipeline
# ...

fb_cache_thr = 0.06
#Add FB cache to the pipeline
enable_fbcache = True
pipe.dit.__class__.enable_fbcache = enable_fbcache
pipe.dit.__class__.fb_cache_thr = fb_cache_thr
pipe.dit.__class__.first_block_cache = None
pipe.dit.__class__.last_block_cache = None
pipe.dit.__class__.forward = fbcache_forward
```

# Thoughts

Threshold is different for each model. I tried to find a good threshold for each model by trial and error.

For Wan2.1 T2V 14B, I found that the threshold is `0.06` or `0.07` works well.

For Wan2.1 I2V 1.3B, I found that the threshold is `0.03` or `0.04` works well.

# Results

## Wan 2.1 T2V 14B

| No FBCache | 0.04 FBCache | 0.06 FBCache | 0.09 FBCache |
|---|---|---|---|
| 1216.17 sec | 1090.77 sec (1.11X) | 657.64 sec (1.85X) | 359.47 sec (Not usable) |
Videos are available at [assets](https://github.com/testdummyvt/diffsynth-wan-cache/tree/refs/heads/main/assets).


## Wan 2.1 I2V 1.3B

| No FBCache | 0.03 FBCache | 0.04 FBCache | 0.06 FBCache |
|---|---|---|---|
| 227.14 sec | 200.35 sec (1.13X) | 167.06 sec (1.36X) | 80.41 sec (Not usable) |
Videos are available at [assets](https://github.com/testdummyvt/diffsynth-wan-cache/tree/refs/heads/main/assets).

## Wan 2.1 I2V 14B (480P)

I was not able to run it on my setup. Need testing.