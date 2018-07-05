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

#if defined __NVCC__
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <glm/glm.hpp>

#include "header.h"

//LOD counts
uint lod1_count;
uint lod2_count;
uint lod3_count;

//LOD feedback memory
uint* d_lod_counts;
uint* d_lod_counts_reduced;
size_t pitch;
uint pitch_int;


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
	data2[index].z = (float)agents->lod[index];
	data2[index].w = 0.0;
}

/** generate_agent_keyvalue_pairs
 * Outputs key value pairs based on agents LOD used to sort pesestrain agents by LOD
 * @param	keys sort key outpts lists
 * @param	values sort identifiers output list
 * @param	agents pedestrian agent list from FLAME GPU
 */
__global__ void generate_agent_keyvalue_pairs(uint* keys, uint* values, xmachine_memory_agent_list* agents)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned int sort_val = (uint)agents->lod[index];

	keys[index] = sort_val;
	values[index] = index;
}

/** generate_agent_lods
 * Creates 3 rows of flags (1 or 0) used to indicate the level of detail for each agent. A global reduction is then used for each list to calculate the number of each LOD in the population
 * @param	pitch memory pitch of each row
 * @param	lods block of memory for 3 rows of data
 * @param	agents pedestrian agent list from FLAME GPU
 */
__global__ void generate_agent_lods(uint pitch, uint* lods, xmachine_memory_agent_list* agents)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned int lod_val = agents->lod[index];

	//uint* x = (uint*)((char*)lods + 0 * pitch) + index;
	uint lod_x = __mul24(pitch, 0) + index;
	uint lod_y = __mul24(pitch, 1) + index;
	uint lod_z = __mul24(pitch, 2) + index;

	if (lod_val == 1)
		lods[lod_x] = 1;
	else if (lod_val == 2)
		lods[lod_y] = 1;
	else if (lod_val == 3)
		lods[lod_z] = 1;
		
}


//EXTERNAL FUNCTIONS DEFINED IN PedestrianPopulation.h
extern void initGPULODFeedback()
{
	size_t width;
	size_t height;

	//gpuErrchk( cudaMalloc( (void**) &d_lod_counts, sizeof(lod_count_list)));
	width = xmachine_memory_agent_MAX * sizeof(uint);
	height = 3;

    gpuErrchk( cudaMallocPitch( (void**) &d_lod_counts, &pitch, width, height));
	gpuErrchk( cudaMallocPitch( (void**) &d_lod_counts_reduced, &pitch, width, height));

	pitch_int = pitch/sizeof(uint); //pitch size in in chars so normalise for int size
}

extern void generate_instances_and_LOD(GLuint* instances_data1_tbo, GLuint* instances_data2_tbo, cudaGraphicsResource_t * p_instances_data1_cgr, cudaGraphicsResource_t * p_instances_data2_cgr)
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
		gpuErrchk(cudaGraphicsMapResources(1, p_instances_data1_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr_1, 0, *p_instances_data1_cgr));

		gpuErrchk(cudaGraphicsMapResources(1, p_instances_data2_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr_2, 0, *p_instances_data2_cgr));

		//cuda block size
		tile_size = (int) ceil((float)get_agent_agent_default_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
		//kernel
		output_pedestrians_to_TBO<<< grid, threads>>>(get_device_agent_default_agents(), dptr_1, dptr_2);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, p_instances_data1_cgr));
        gpuErrchk(cudaGraphicsUnmapResources(1, p_instances_data2_cgr));


		//Sort agents by lod
		sort_agents_default(&generate_agent_keyvalue_pairs);
		//reset counts
		gpuErrchk(cudaMemset(d_lod_counts, 0, pitch*3));
		//generate new counts
		generate_agent_lods<<<grid, threads>>>(pitch_int, d_lod_counts, get_device_agent_default_agents());
		//parallel reduce
		thrust::inclusive_scan(thrust::device_pointer_cast(&d_lod_counts[pitch_int*0]), thrust::device_pointer_cast(&d_lod_counts[pitch_int*0]) + get_agent_agent_default_count(), thrust::device_pointer_cast(&d_lod_counts_reduced[pitch_int*0]));
		thrust::inclusive_scan(thrust::device_pointer_cast(&d_lod_counts[pitch_int*1]), thrust::device_pointer_cast(&d_lod_counts[pitch_int*1]) + get_agent_agent_default_count(), thrust::device_pointer_cast(&d_lod_counts_reduced[pitch_int*1]));
		thrust::inclusive_scan(thrust::device_pointer_cast(&d_lod_counts[pitch_int*2]), thrust::device_pointer_cast(&d_lod_counts[pitch_int*2]) + get_agent_agent_default_count(), thrust::device_pointer_cast(&d_lod_counts_reduced[pitch_int*2]));
		//reset and then update counts
		lod1_count = 0;
		lod2_count = 0;
		lod3_count = 0;
		gpuErrchk( cudaMemcpy( &lod1_count, &d_lod_counts_reduced[(pitch_int*0)+get_agent_agent_default_count()-1], sizeof(uint), cudaMemcpyDeviceToHost));
		gpuErrchk( cudaMemcpy( &lod2_count, &d_lod_counts_reduced[(pitch_int*1)+get_agent_agent_default_count()-1], sizeof(uint), cudaMemcpyDeviceToHost));
		gpuErrchk( cudaMemcpy( &lod3_count, &d_lod_counts_reduced[(pitch_int*2)+get_agent_agent_default_count()-1], sizeof(uint), cudaMemcpyDeviceToHost));
	}

}

extern int getPedestrianLOD1Count()
{
	return lod1_count;
}

extern int getPedestrianLOD2Count()
{
	return lod2_count;
}

extern int getPedestrianLOD3Count()
{
	return lod3_count;
}
