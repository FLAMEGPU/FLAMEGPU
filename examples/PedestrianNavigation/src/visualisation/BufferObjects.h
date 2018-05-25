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
#ifndef __BUFFER_OBJECTS
#define __BUFFER_OBJECTS

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

/** createVBO
 * Creates a Vertex Buffer Object (VBO)
 * @param	vbo	pointer to VBO
 * @param	target either GL_ARRAY_BUFFER or GL_ELEMENT_ARRAY_BUFFER
 * @param	size total size in bytes
 */
void createVBO(GLuint* vbo, GLenum target, GLuint size);
/** deleteVBO
 * Deletes a Vertex Buffer Object (VBO)
 * @param	vbo	pointer to VBO
 */
void deleteVBO( GLuint* vbo);

/** createTBO
 * Creates a Texture Buffer Object (TBO)
 * @param	tbo	pointer to TBO
 * @param	tex pointer to uninitialised texture instance
 * @param	size total size in bytes
 */
void createTBO(GLuint* tbo, GLuint* tex, GLuint size);
/** deleteTBO
 * Deletes a Vertex Buffer Object (TBO)
 * @param	tbo	pointer to TBO
 */
void deleteTBO( GLuint* tbo);

//EXTERNAL FUNCTIONS INPLEMENTED IN BufferObjects.cu
/** registerBO
 * Registers a Buffer Object (BO) for use with CUDA
 * @param   cudaResource pointer to a cuda Graphics Resource
 * @param   bo  pointer to BO
 */
extern void registerBO(cudaGraphicsResource_t* cudaResource, GLuint* bo);
/** unregisterBO
 * Unregisters a Buffer Object (BO) from use with CUDA
 * @param   cudaResource pointer to a cuda Graphics Resource
 */
extern void unregisterBO(cudaGraphicsResource_t* cudaResource);



#endif //__BUFFER_OBJECTS
