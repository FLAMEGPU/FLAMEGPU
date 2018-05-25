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
#ifndef __PEDESTRIAN_POPULATION
#define __PEDESTRIAN_POPULATION

#include <cuda_gl_interop.h>


/** initPedestrianPopulation
 * Initialises the pedestrian population by loading model data and creating appropriate buffer objects and shaders
 */
void initPedestrianPopulation();
/** renderPedestrianPopulation
 * Renders the pedestrian population by outputting agent data to texture buffer objects and then using vertex texture instancing 
 */
void renderPedestrianPopulation();



//EXTERNAL FUNCTIONS IMPLEMENTD IN PedestrianPopulation.cu CUDA FILE
/** initGPULODFeedback
 * Initialises the GPU Level of Detail (LOD) feedback which requires thrust for parallel reduction
 */
extern void initGPULODFeedback();
/** generate_instances_and_LOD
 *  Generates instances by calling a CUDA Kernel which outputs agent data to a texture buffer object. Orders agent population by Level of Detail (LOD) and counts the number of each detail level.
 * @param instances_data1_tbo Texture Buffer Object used for storing instances data
 * @param instances_data2_tbo Texture Buffer Object used for storing instances data
 */
extern void generate_instances_and_LOD(GLuint* instances_data1_tbo, GLuint* instances_data2_tbo, cudaGraphicsResource_t * p_instances_data1_cgr, cudaGraphicsResource_t * p_instances_data2_cgr);
/** getPedestrianLOD1Count
 * Returns the Level of Detail (LOD) count for detail level 1
 * @return the number of agents at detail level 1
 */
extern int getPedestrianLOD1Count();
/** getPedestrianLOD2Count
 * Returns the Level of Detail (LOD) count for detail level 2
 * @return the number of agents at detail level 2
 */
extern int getPedestrianLOD2Count();
/** getPedestrianLOD3Count
 * Returns the Level of Detail (LOD) count for detail level 3
 * @return the number of agents at detail level 3
 */
extern int getPedestrianLOD3Count();


/** Vertex Shader source for rendering animated (keyframed x2) directional pedestrians */
static const char pedestrian_vshader_source[] = 
{  
	"#version 120																	\n"
	"#extension EXT_gpu_shader4 : require   										\n"
	"uniform samplerBuffer data1_map;												\n"
	"uniform samplerBuffer data2_map;												\n"
	"attribute in float instance_index;												\n"
	"attribute in vec3 normal_l, position_r, normal_r;								\n"
	"void main()																	\n"
    "{																				\n"
	"   int index = int(instance_index);											\n"
	"	vec4 data1 = texelFetchBuffer(data1_map, index);							\n"
	"	vec4 data2 = texelFetchBuffer(data2_map, index);							\n"

	"   //blend keyframes															\n"
	"	//data1.z = 0.5;															\n"
	"   vec3 lerp_position = mix(gl_Vertex.xyz, position_r, data1.z);				\n"

	"	//calculate rotation angle componants										\n"
	"	vec2 velocity = data2.xy;													\n"
	"	float angle = atan(velocity.y/velocity.x);									\n"
	"	if (velocity.x >= 0)														\n"
	"		angle += 3.14159265;    //rot 180 degrees								\n"
	"	angle += 1.57079633;		//rot 90 degrees								\n"
	"	float cosLength = cos(angle);												\n"
	"	float sinLength = sin(angle);												\n"

	"	//rotate the model															\n"
	"	vec3 position;																\n"
	"	position[0] = cosLength * lerp_position[0] - sinLength * lerp_position[1];	\n"
	"	position[1] = sinLength * lerp_position[0] + cosLength * lerp_position[1];	\n"
	"	position[2] = lerp_position[2];												\n"

	"   //offset model position														\n"
	"   position.x += data1.x;														\n"
	"   position.y += data1.y;														\n"

	"	//color																		\n"
	"	int lod = int(data2.z);														\n"	
	"	if(lod == 1)																\n"	
	"		gl_FrontColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);							\n"
	"	else if(lod == 2)															\n"	
	"		gl_FrontColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);							\n"
	"	else																		\n"	
	"		gl_FrontColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);							\n"

	"   //apply model view proj														\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1);				\n"
	"   //gl_Normal = normal;														\n"
    "}																				\n"
};


#endif //__PEDESTRIAN_POPULATION
