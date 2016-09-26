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
#include <glm/glm.hpp>
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
//KERNEL DEFINITIONS
/** output_navmaps_to_TBO
 * Outputs navmap agent data from FLAME GPU to a 4 component vector used for instancing
 * @param	agents	pedestrian agent list from FLAME GPU
 * @param	data1 four component vector used to output instance data 
 * @param	data2 four component vector used to output instance data 
 */
__global__ void output_pedestrians_to_TBO(xmachine_memory_agent_list* agents, glm::vec4* data1, glm::vec4* data2){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	data1[index].x = agents->x[index];
	data1[index].y = agents->y[index];
	data1[index].z = agents->animate[index];
	data1[index].w = agents->height[index];

	data2[index].x = agents->velx[index];
	data2[index].y = agents->vely[index];
	data2[index].z = (float)agents->exit_no[index];
	data2[index].w = 0.0;
}


void generate_pedestrian_instances(GLuint* instances_data1_tbo, GLuint* instances_data2_tbo)
{
	//kernals sizes
	int threads_per_tile = 128;
	int tile_size;
	dim3 grid;
    dim3 threads;

	//pointer
	glm::vec4 *dptr_1;
	glm::vec4 *dptr_2;
	
	if (get_agent_agent_default_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGLMapBufferObject( (void**)&dptr_1, *instances_data1_tbo));
		gpuErrchk(cudaGLMapBufferObject( (void**)&dptr_2, *instances_data2_tbo));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_agent_default_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
		//kernel
		output_pedestrians_to_TBO<<< grid, threads>>>(get_device_agent_default_agents(), dptr_1, dptr_2);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGLUnmapBufferObject(*instances_data1_tbo));
		gpuErrchk(cudaGLUnmapBufferObject(*instances_data2_tbo));
	}
}


int getPedestrianCount()
{
	return get_agent_agent_default_count();
}