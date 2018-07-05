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
#ifndef _NAVMAP_POPULATION
#define _NAVMAP_POPULATION

#include <cuda_gl_interop.h>
#include "CustomVisualisation.h"

/** initNavMapPopulation
 * Initialises the navigation Map Population by loading model data and creating appropriate buffer objects and shaders
 * @param use_large_vbo determins if arrows should be instanced or displayed using a single large vbo
 */
void initNavMapPopulation();

/** renderNavMapModel
 * Renders the navigation Map Population by outputting agent data to a texture buffer object and then using vertex texture instancing 
 */
void renderNavMapPopulation();

/** setGridDisplayOnOff
 * Toggles the grid display on or off depending on the current state
 */
void toggleGridDisplayOnOff();

/** setArrowDisplayOnOff
 * Turns the arrow display on or off
 * @param state on off state 
 */
void setArrowsDisplayOnOff(TOGGLE_STATE state);

/** toggleArrowDisplayOnOff
 * Toggles the arrow display on or off depending on the current state
 */
void toggleArrowsDisplayOnOff();

/** getActiveExit
 * Gets the active exit number
 */
int getActiveExit();

//EXTERNAL FUNCTIONS IMPLEMENTD IN NavMapPopulation.cu CUDA FILE
/** generate_instances
 *  Generates instances by calling a CUDA Kernel which outputs agent data to a texture buffer object
 * @param instances_tbo Texture Buffer Object used for storing instances data
 */
extern void generate_instances(GLuint* instances_tbo, cudaGraphicsResource_t * instances_cgr);

/** displayMapNumber
 *  Sets which map number should be displayed (0 = collision map, 1 = exit 1 map, etc).
 * @param map_no Map number to be displayed
 */
extern void displayMapNumber(int map_no);

/** getCurrentMap
 *  Gets the currently selected map
 */
extern int getCurrentMap();



/** Vertex Shader source for rendering directional arrows */
static const char navmap_vshader_source[] = 
{  
	"#version 120																	\n"
	"#extension EXT_gpu_shader4 : require   										\n"
	"uniform samplerBuffer instance_map;											\n"
	"uniform float NM_WIDTH;														\n"
	"uniform float ENV_MAX;															\n"
	"uniform float ENV_WIDTH;														\n"
	"attribute in float instance_index;												\n"
    "void main()																	\n"
    "{																				\n"
	"   int index = int(instance_index);											\n"
	"	vec4 instance = texelFetchBuffer(instance_map, index);						\n"

	"	//calculate rotation angle componants										\n"
	"	float angle = instance.z;													\n" 	
	"   bool is_force = true;														\n"
	"   if (angle >7.0f){	//more than 360 degrees(6.28 rads)						\n"
	"       is_force = false;														\n"
	"		angle = 1.57079633f;													\n"
	"	}																			\n"
	"	float cosLength = cos(angle);												\n"
	"	float sinLength = sin(angle);												\n"

	"	//rotate the model															\n"
	"	vec3 position;																\n"
	"   if (is_force){																\n"
	"		position[0] = cosLength * gl_Vertex[0] - sinLength * gl_Vertex[1];		\n"
	"		position[1] = sinLength * gl_Vertex[0] + cosLength * gl_Vertex[1];		\n"
	"		position[2] = gl_Vertex[2];												\n"
	"	}else{																		\n"
	"		position[0] = gl_Vertex[0];												\n"
	"	    position[1] = cosLength * gl_Vertex[1] - sinLength * gl_Vertex[2];		\n"
	"		position[2] = sinLength * gl_Vertex[1] + cosLength * gl_Vertex[2];		\n"
	"	}																			\n"

	"	//select output color														\n"
	"   if (is_force){																\n"
	"		gl_FrontColor = vec4(0.75, 0, 0, 0);									\n"			
	"	}else{																		\n"
	"		gl_FrontColor = vec4(0.9, 0.9, 0.9, 1);									\n"
	"	}																			\n"


	"   //offset model position														\n"
	"	float x_displace = ((instance.x+0.5)/(NM_WIDTH/ENV_WIDTH))-ENV_MAX;			\n"
	"	float y_displace = ((instance.y+0.5)/(NM_WIDTH/ENV_WIDTH))-ENV_MAX;			\n"
	"   position.x += x_displace;													\n"
	"   position.y += y_displace;													\n"
	"   position.z += instance.w*0.0775;													\n"

	"   //apply model view proj														\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1);				\n"
    "}																				\n"
};

#endif //_NAVMAP_POPULATION
