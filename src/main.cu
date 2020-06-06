#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

char *render_grayscale(const struct perlin_map *p)
{
	assert(p != NULL);
	char *im = (char *)malloc(sizeof(char) * (p->cells_x * p->grain) * (p->cells_y * p->grain) * 4);
	assert(im != NULL);
	double min, max;
	min = (p->heights)[0];
	max = (p->heights)[0];
	int num_elems = (p->cells_x * p->grain) * (p->cells_y * p->grain);
	for (int i = 0; i < num_elems; i++) {
		if ((p->heights)[i] < min) {
			min = (p->heights)[i];
		}
		if ((p->heights)[i] > max) {
			max = (p->heights)[i];
		}
	}

	for (int i = 0; i < num_elems; i++) {
		

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
		// un-linearise
		x_pos = i % (c_x * g);
		y_pos = (i - x_pos) / (c_x * g);
		// snap to grid
		 
		


	
	}



}




int main(void)
{
	// allocate host memory
	struct perlin_map *p = perlin_map_new(CELL_X, CELL_Y, GRAIN);

			
	// allocate device memory
	double *d_heights;
	cudaMalloc((void **)&d_heights, sizeof(double) * (CELL_X * GRAIN) * (CELL_Y * GRAIN));
	




	return 0;
}


