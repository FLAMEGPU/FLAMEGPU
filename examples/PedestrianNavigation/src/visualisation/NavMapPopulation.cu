/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "header.h"

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
}

__constant__ int currentMap;
int h_currentMap;


inline __device__ float dot(glm::vec2 a, glm::vec2 b)
{ 
    return a.x * b.x + a.y * b.y;
}
inline __device__ float length(glm::vec2 v)
{
    return sqrtf(dot(v, v));
}

//KERNEL DEFINITIONS
/** output_navmaps_to_TBO
 * Outputs navmap agent data from FLAME GPU to a 4 component vector used for instancing
 * @param	agents	navmap agent list from FLAME GPU
 * @param	data four component vector used to output instance data 
 */
__global__ void output_navmaps_to_TBO(xmachine_memory_navmap_list* agents, glm::vec4* data){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//change thenext two outputs to visualise different sets of forces
	glm::vec2 velocity;
	if (currentMap == 0)
		velocity = glm::vec2(agents->collision_x[index], agents->collision_y[index]);
	else if(currentMap == 1)
		velocity = glm::vec2(agents->exit0_x[index], agents->exit0_y[index]);
	else if(currentMap == 2)
		velocity = glm::vec2(agents->exit1_x[index], agents->exit1_y[index]);
	else if(currentMap == 3)
		velocity = glm::vec2(agents->exit2_x[index], agents->exit2_y[index]);
	else if(currentMap == 4)
		velocity = glm::vec2(agents->exit3_x[index], agents->exit3_y[index]);
	else if(currentMap == 5)
		velocity = glm::vec2(agents->exit4_x[index], agents->exit4_y[index]);
	else if(currentMap == 6)
		velocity = glm::vec2(agents->exit5_x[index], agents->exit5_y[index]);
	else if(currentMap == 7)
		velocity = glm::vec2(agents->exit6_x[index], agents->exit6_y[index]);
	

	float angle;	
	if (length(velocity)<0.001f){
		angle = 8.0f;	//greater than 360 degrees (6.28 rands)
	}else
	{
		angle = atan(velocity.y/velocity.x);
		if (velocity.x >= 0)
			angle += 3.14159265f;
		angle += 1.57079633f;
	}
	
	data[index].x = agents->x[index];
	data[index].y = agents->y[index];
	data[index].z = angle;
	data[index].w = agents->height[index];
}

//EXTERNAL FUNCTIONS DEFINED IN NavMapPopulation.h
extern void generate_instances(GLuint* instances_tbo, cudaGraphicsResource_t * instances_cgr)
{
	//kernals sizes
	int threads_per_tile = 128;
	int tile_size;
	dim3 grid;
    dim3 threads;

	//pointer
	glm::vec4 *dptr_1;
	
	if (get_agent_navmap_static_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, instances_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr_1, 0, *instances_cgr));

		//cuda block size
		tile_size = (int) ceil((float)get_agent_navmap_static_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
		//call kernel
		output_navmaps_to_TBO<<< grid, threads>>>(get_device_navmap_static_agents(), dptr_1);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, instances_cgr));
	}
}

void displayMapNumber(int map_no){
	h_currentMap = map_no;
	gpuErrchk(cudaMemcpyToSymbol( currentMap, &map_no, sizeof(int)));	
}

int getCurrentMap()
{
	return h_currentMap;
}
