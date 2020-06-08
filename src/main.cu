#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "lodepng.h"

#define WIDTH_4K 3840
#define HEIGHT_4K 2160

struct perlin_noise {
    	double *noise_map;
	unsigned int width;
	unsigned int height;
	unsigned int samples; // per integral value
};

struct perlin_noise *perlin_noise_new(unsigned int w, unsigned int h, unsigned int s)
{
	struct perlin_noise *p = (struct perlin_noise *)malloc(sizeof(struct perlin_noise));
   	assert(p != NULL);
   	p->width = w;
   	p->height = h;
   	p->samples = s;
   	p->noise_map = (double *)malloc(sizeof(double) * p->width * p->height);
	assert(p->noise_map != NULL);
    	return p;
}

void perlin_noise_destroy(struct perlin_noise **p)
{
    	assert(*p != NULL);
    	free((*p)->noise_map);
   	(*p)->noise_map = NULL;
   	(*p)->width = 0;
    	(*p)->height = 0;
    	(*p)->samples = 0;
    	free(*p);
    	(*p) = NULL;
}

unsigned char *perlin_noise_render(const struct perlin_noise *p)
{
	assert(p != NULL);
	unsigned char *im = (unsigned char *)malloc(sizeof(char) * p->width * p->height * 4);
	assert(im != NULL);
	// find min and max values
	double min, max;
	min = (p->noise_map)[0];
	max = (p->noise_map)[0];
	for (int i = 0; i < p->width; i++) {
		for (int j = 0; j < p->height; j++) {
			if ((p->noise_map)[i + j * (p->width)] < min) {
				min = (p->noise_map)[i + j * (p->width)];
			}
			if ((p->noise_map)[i + j * (p->width)] > max) {
				max = (p->noise_map)[i + j * (p->width)];
			}
		}
	} 
	// write rgba 
	for (int i = 0; i < p->width; i++) {
		for (int j = 0; j < p->height; j++) {
			char val = (char)(255. * (((p->noise_map)[i + j * (p->width)] - min) / (max - min)));
			im[4 * i + (4 * j * (p->width)) + 0] = val;
			im[4 * i + (4 * j * (p->width)) + 1] = val;
			im[4 * i + (4 * j * (p->width)) + 2] = val;
			im[4 * i + (4 * j * (p->width)) + 3] = 255;
		}
	}
	return im;
}

__device__
double fade(double t)
{
	return 6 * (t * t * t * t * t) - 15 * (t * t * t * t) + 10 * (t * t * t);
}

__device__
double linterp(double t, double a, double b)
{
	return a + t * (b - a);
}

__device__
double grad(int hash, double a, double b)
{
    switch (hash & 0x7) {
        case 0x0: return a + b;
		case 0x1: return -a + b;
		case 0x2: return a - b;
		case 0x3: return -a - b;
		case 0x4: return a;
		case 0x5: return -a;
		case 0x6: return b;
		case 0x7: return -b;
		default:
			return 0;
    }
}

__global__
void perlin_noise_fill(double *noise_map, unsigned int w, unsigned int h, unsigned int s, unsigned int oct, double pers)
{
	int num_elems = w * h; 
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	double x, y;
   	int X, Y;
   	double x_fade, y_fade;
	int p[512] = {151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163, 70,221,153,101,155,167,43,172,9, 129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228, 251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254, 138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};
	int A, AA, AB, B, BA, BB;
	double noise, freq, amp;
	// grid-stride loop
	for (int i = index; i < num_elems; i += stride) {
		noise = 0.;
		freq = 1.;
		amp = 1.;
		for (int o = 0; o < oct; o++) {
			x = ((i % w) * (1. / (double)s)) * freq;
			y = (((i - (i % w)) / w) * (1. / (double)s)) * freq;
			X = (int)floor(x);
			Y = (int)floor(y);
			x -= X;
			y -= Y;
			X &= 255;
			Y &= 255;
			x_fade = fade(x);
			y_fade = fade(y);
			A = p[X] + Y;
			AA = p[A];
			AB = p[A + 1];
			B = p[X + 1] + Y;
			BA = p[B];
			BB = p[B + 1];
			noise += amp * linterp(y_fade, linterp(x_fade, grad(p[AA], x, y), grad(p[BA], x - 1, y)), linterp(x_fade, grad(p[AB], x, y - 1), grad(p[BB], x - 1, y - 1)));
			amp *= pers;
			freq *= 2;
		}
		noise_map[i] = noise;
	}
}

int main(void)
{
	// allocate host memory
	struct perlin_noise *p = perlin_noise_new(WIDTH_4K, HEIGHT_4K, 100);

	// allocate device memory
	double *d_noise;
	cudaMalloc((void **)&d_noise, sizeof(double) * p->width * p->height);
	
	// run kernel
	perlin_noise_fill<<<1024, 1024>>>(d_noise, p->width, p->height, p->samples, 5, 0.8);	
	
 	// transfer map to host
	cudaMemcpy(p->noise_map, d_noise, sizeof(double) * p->width * p->height, cudaMemcpyDeviceToHost);

	// render
	unsigned char *render = perlin_noise_render(p);
	unsigned char *png = 0;
	size_t pngsize;
	unsigned int err = lodepng_encode32(&png, &pngsize, render, p->width, p->height);
	lodepng_save_file(png, pngsize, "test.png");

	// deallocate device memory
	cudaFree(d_noise);
	
	// deallocate host memory
	perlin_noise_destroy(&p);
	
	
	return 0;
}

