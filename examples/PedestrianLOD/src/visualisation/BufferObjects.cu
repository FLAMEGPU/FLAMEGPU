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

extern void registerBO(cudaGraphicsResource_t* cudaResource, GLuint* bo)
{
    gpuErrchk(cudaGraphicsGLRegisterBuffer(cudaResource, *bo, cudaGraphicsMapFlagsWriteDiscard)); 
}

extern void unregisterBO(cudaGraphicsResource_t* cudaResource)
{
    gpuErrchk(cudaGraphicsUnregisterResource(*cudaResource));
}
