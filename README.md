# perlin

## Overview
Ken Perlin's eponymous [noise function](https://en.wikipedia.org/wiki/Perlin_noise)&mdash;whose original C implementation may be found [here](https://mrl.nyu.edu/~perlin/doc/oscar.html#noise)&mdash;is a means of procedurally generating textures *apropos* CGI (realtime or otherwise). The essence of his algorithm can be distilled as follows: a pseudorandom gradient field is defined over an *n*-dimensional lattice, from which a surface may be interpolated in *R^n*. Sampling this surface yields a type of [gradient noise](https://en.wikipedia.org/wiki/Gradient_noise). 

My implementation of *Perlin noise* is not so much an exercise in computer graphics as it is in parallel programming: this sort of algorithm qualifies as [embarassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) in the sense that any pixel of a given texture may be sampled (read: computed) independently of every other pixel. Here, I've ported Perlin noise to [CUDA](https://en.wikipedia.org/wiki/CUDA) in the hope that even a naive parallelization should easily amortize the cost of *cudaMemcpy()*&mdash;note in particular that the data only go one way (DeviceToHost).

My work on the University of Rochester's [BlueHive cluster](https://www.circ.rochester.edu/resources.html) has so far involved very little usage of its NVIDIA K80s. This project, with thanks to the [Center for Integrated Research Computing](https://www.circ.rochester.edu/), really amounts to a contrived exercise in learning how to run workloads on a GPU.

## Design
*perlin* implements a two-dimensional version of Perlin's ['improved'](https://mrl.nyu.edu/~perlin/paper445.pdf) noise function (the relevant improvement boils down to better interpolation). The noise generation is completely GPU-accelerated, parallelized over the entire frame buffer using a grid-stride loop *a la* NVIDIA's [Mark Harris](https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/). As in Perlin's original implementation, the pseudorandom vector field is generated by hashing the integral part of every sample coordinate into a permutation table (this part of the code should be attributed to Dave Mount and Roger Eastman's [lecture on Perlin noise](https://www.cs.umd.edu/class/spring2018/cmsc425/Lects/lect13-2d-perlin.pdf) at the University of Maryland). My *grad()* function is a two-dimensional simplification of Riven's optimized [gradient/dot-product](http://riven8192.blogspot.com/2010/08/calculate-perlinnoise-twice-as-fast.html). Finally, I consulted Adrian Biagioli's very helpful [writeup](https://adrianb.io/2014/08/09/perlinnoise.html) to support the layering of multiple 'octaves' of noise. 

The CUDA kernel is followed by a *cudaMemcpy()* to the CPU, where the data are encoded to a PNG using Lode Vandevenne's [LodePNG](https://lodev.org/lodepng/) as in my [previous project](https://github.com/brekekekex/seam).

## Usage 
*perlin* is more a vular research script than a standalone program. With that said, one may build and run it as follows:

Clone the repository:
```
git clone https://github.com/brekekekex/perlin.git
```

Download *lodepng.c* and *lodepng.h* from [https://github.com/lvandeve/lodepng](https://github.com/lvandeve/lodepng) and place them in *~/perlin/src*. Rename *lodepng.c* to *lodepng.cu*. A crude SLURM script is provided, which may be run with 
```
sbatch make.sh
```

Otherwise, directly compile the source with NVCC:
```
nvcc -o perlin main.cu lodepng.cu
```








