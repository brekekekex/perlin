# perlin

## Overview
Ken Perlin's eponymous [noise function](https://en.wikipedia.org/wiki/Perlin_noise)&mdash;whose original C implementation may be found [here](https://mrl.nyu.edu/~perlin/doc/oscar.html#noise)&mdash;is a means of procedurally generating textures apropos CGI (realtime or otherwise). The essence of his algorithm can be distilled as follows: a pseudorandom gradient field is defined over an *n*-dimensional lattice, from which a surface may be interpolated in *R^n*. Sampling this surface yields a type of [gradient noise](https://en.wikipedia.org/wiki/Gradient_noise). 

My implementation of *Perlin noise* is not so much an exercise in computer graphics as it is in parallel programming: this sort of work qualifies as [embarassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) in the sense that any pixel of a given texture may be sampled (read: computed) independently of every other pixel.     

