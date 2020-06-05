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




int main(void)
{
	struct perlin_map *p = perlin_map_new(CELL_X, CELL_Y, GRAIN);

	




	return 0;
}


