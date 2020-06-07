#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "lodepng.h"

#define CELL_X 600
#define CELL_Y 600
#define GRAIN 10

struct perlin_map{
	double *heights;
	unsigned int cells_x;
	unsigned int cells_y;
	unsigned int grain;
};

struct perlin_map *perlin_map_new(unsigned int c_x, unsigned int c_y, unsigned int g)
{
	struct perlin_map *p = (struct perlin_map *)malloc(sizeof(struct perlin_map));
	assert(p != NULL);
	p->cells_x = c_x;
	p->cells_y = c_y;
	p->grain = g;
	p->heights = (double *)malloc(sizeof(double) * (p->cells_x * p->grain) * (p->cells_y * p->grain));
	return p;
}

void perlin_map_destroy(struct perlin_map **p) 
{
	assert(*p != NULL);
	free((*p)->heights);
	(*p)->cells_x = 0;
	(*p)->cells_y = 0;
	(*p)->grain = 0;
	free(*p);
	(*p) = NULL;
}

unsigned char *render_grayscale(const struct perlin_map *p)
{
	assert(p != NULL);
	unsigned char *im = (unsigned char *)malloc(sizeof(char) * (p->cells_x * p->grain) * (p->cells_y * p->grain) * 4);
	assert(im != NULL);
	// find min and max heights
	double min, max;
	min = (p->heights)[0];
	max = (p->heights)[0];
	for (int i = 0; i < (p->cells_x * p->grain); i++) {
		for (int j = 0; j < (p->cells_y * p->grain); j++) {
			if ((p->heights)[i + j * (p->cells_x * p->grain)] < min) {
				min = (p->heights)[i + j * (p->cells_x * p->grain)];
			}
			if ((p->heights)[i + j * (p->cells_x * p->grain)] > max) {
				max = (p->heights)[i + j * (p->cells_x * p->grain)];
			}
		}
	} 
	// write rgba (use alpha to encode normalized heights)
	for (int i = 0; i < (p->cells_x * p->grain); i++) {
		for (int j = 0; j < (p->cells_y * p->grain); j++) {
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 0] = 0;
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 1] = 0;
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 2] = 0;
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 3] = (char)(255. * (((p->heights)[i + j * (p->cells_x * p->grain)] - min) / (max - min)));
		}
	}
	return im;
}

__device__
double fade(int d)
{
	return (double)(6 * (d * d * d * d * d) - 15 * (d * d * d * d) + 10 * (d * d * d));
}

__device__
double linterp(double t, double a, double b)
{
	return a + t * (b - a);

}

__device__
double grad(int hash, int x_d, int y_d)
{
	switch (hash & 0x7) {
		case 0x0: return (double)(x_d + y_d);
		case 0x1: return (double)(-x_d + y_d);
		case 0x2: return (double)(x_d - y_d);
		case 0x3: return (double)(-x_d - y_d);
		case 0x4: return (double)(x_d);
		case 0x5: return (double)(-x_d);
		case 0x6: return (double)(y_d);
		case 0x7: return (double)(-y_d);
	}
}

__global__
void perlin_fill_heights(double *height_map, unsigned int c_x, unsigned int c_y, unsigned int g)
{
	int num_elems = (c_x * g) * (c_y * g); 
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;


	// grid-stride loop
	for (int i = index; i < num_elems; i += stride) {
		int x_pos, y_pos;
		int x_disp, y_disp;
		double x_fade, y_fade;
		// un-linearise
		x_pos = i % (c_x * g);
		y_pos = (i - x_pos) / (c_x * g);
		// displacements
		x_disp = x_pos % g;
		y_disp = y_pos % g;
		// fade
		x_fade = fade(x_disp);
		y_fade = fade(y_disp);
		
	}
}

int main(void)
{
	// allocate host memory
	struct perlin_map *p = perlin_map_new(CELL_X, CELL_Y, GRAIN);

	for (int i = 0; i < (p->cells_x * p->grain); i++) {
		for (int j = 0; j < (p->cells_y * p->grain); j++) {	
			(p->heights)[i + j * (p->cells_x * p->grain)] = i + j;
		}
	} 	
	/*
	unsigned char *render = render_grayscale(p);
	unsigned char *png = 0;
	size_t pngsize;	
	unsigned int err = lodepng_encode32(&png, &pngsize, render, p->cells_x * p->grain, p->cells_y * p->grain);	
	lodepng_save_file(png, pngsize, "test.png");
	
*/	
	// allocate device memory
	//double *d_heights;
	//cudaMalloc((void **)&d_heights, sizeof(double) * (CELL_X * GRAIN) * (CELL_Y * GRAIN)); 	
	perlin_map_destroy(&p);



	return 0;
}


