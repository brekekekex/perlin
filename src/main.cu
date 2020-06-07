#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "lodepng.h"

#define CELL_X 60
#define CELL_Y 60
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
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 0] = (char)(0.3 * 255. * (((p->heights)[i + j * (p->cells_x * p->grain)] - min) / (max - min)));
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 1] = (char)(0.59 * 255. * (((p->heights)[i + j * (p->cells_x * p->grain)] - min) / (max - min)));
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 2] = (char)(0.11 * 255. * (((p->heights)[i + j * (p->cells_x * p->grain)] - min) / (max - min)));
			im[4 * i + (4 * j * (p->cells_x * p->grain)) + 3] = 255;
		}
	}
	return im;
}

__device__
double fade(int du)
{
	double d = ((float)du) / 10;
	return (double)(6 * (d * d * d * d * d) - 15 * (d * d * d * d) + 10 * (d * d * d));
}

__device__
double linterp(double t, double a, double b)
{
	return a + t * (b - a);
}

__device__
double grad(int hash, int x_du, int y_du)
{
	double x_d = ((double)x_du) / 10;
	double y_d = ((double)y_du) / 10;

	switch (hash & 0x7) {
		case 0x0: return (double)(x_d + y_d);
		case 0x1: return (double)(-x_d + y_d);
		case 0x2: return (double)(x_d - y_d);
		case 0x3: return (double)(-x_d - y_d);
		case 0x4: return (double)(x_d);
		case 0x5: return (double)(-x_d);
		case 0x6: return (double)(y_d);
		case 0x7: return (double)(-y_d);
		default:
			assert(0);
	}
}

__global__
void perlin_fill_heights(double *height_map, unsigned int c_x, unsigned int c_y, unsigned int g)
{
	int num_elems = (c_x * g) * (c_y * g); 
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;


	int p[512] = {151,160,137,91,90,15, 131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166, 77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196, 135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123, 5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9, 129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228, 251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254, 138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,151,160,137,91,90,15, 131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166, 77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196, 135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123, 5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9, 129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228, 251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254, 138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180};
	
	// grid-stride loop
	for (int i = index; i < num_elems; i += stride) {
		int x_pos, y_pos;
		int x_disp, y_disp;
		int x_ref, y_ref;
		int A, AA, AB, B, BA, BB;
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
		// find reference x, y
		x_ref = (x_pos - x_disp) % 256;
		y_ref = (y_pos - y_disp) % 256;
		// form hash
		A = p[x_ref] + y_ref;
		AA = p[A];
		AB = p[A + 1];
		B = p[x_ref + 1] + y_ref;
		BA = p[B];
		BB = p[B + 1];
		// assign height
		height_map[i] = linterp(y_fade,
		linterp(x_fade, grad(p[AA], x_disp, y_disp), grad(p[BA], x_disp - g, y_disp)),
		linterp(x_fade, grad(p[AB], x_disp, y_disp - g), grad(p[BB], x_disp - g, y_disp - g)));
	}
}

int main(void)
{
	// allocate host memory
	struct perlin_map *p = perlin_map_new(CELL_X, CELL_Y, GRAIN);

	// allocate device memory
	double *d_heights;
	cudaMalloc((void **)&d_heights, sizeof(double) * (p->cells_x * p->grain) * (p->cells_y * p->grain));
	
	// run kernel
	perlin_fill_heights<<<32, 256>>>(d_heights, p->cells_x, p->cells_y, p->grain);	
	
 	// transfer map to host
	cudaMemcpy(p->heights, d_heights, sizeof(double) * (p->cells_x * p->grain) * (p->cells_y * p->grain), cudaMemcpyDeviceToHost);


	// print
	for (int i = 0; i < p->cells_x * p->grain; i++) {
		for (int j = 0; j < p->cells_y * p->grain; j++) {
			printf("%f\n", (p->heights)[i + j * (p->cells_x * p->grain)]);
		}
	}


	// render
	unsigned char *render = render_grayscale(p);
	unsigned char *png = 0;
	size_t pngsize;
	unsigned int err = lodepng_encode32(&png, &pngsize, render, p->cells_x * p->grain, p->cells_y * p->grain);
	lodepng_save_file(png, pngsize, "test.png");

	// deallocate device memory
	cudaFree(d_heights);
	
	// deallocate host memory
	perlin_map_destroy(&p);
	
	/*
	unsigned char *render = render_grayscale(p);
	unsigned char *png = 0;
	size_t pngsize;	
	unsigned int err = lodepng_encode32(&png, &pngsize, render, p->cells_x * p->grain, p->cells_y * p->grain);	
	lodepng_save_file(png, pngsize, "test.png");
	
*/	
	return 0;
}

