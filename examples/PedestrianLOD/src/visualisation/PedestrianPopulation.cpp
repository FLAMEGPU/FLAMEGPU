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
#include "PedestrianPopulation.h"
#include "OBJModel.h"
#include "BufferObjects.h"
#include "CustomVisualisation.h"

/** Pedestrian Model Scale  */
float PEDESTRIAN_MODEL_SCALE = 0.0025f;

//pedestrian instances 
GLuint p_instances_data1_tbo;
GLuint p_instances_data1_tex;
cudaGraphicsResource_t p_instances_data1_cgr;
GLuint p_instances_data2_tbo;
GLuint p_instances_data2_tex;
cudaGraphicsResource_t p_instances_data2_cgr;

//MODEL DATA FOR LOD 1
//primative count
int lod1_v_count;
int lod1_f_count;
//reft keyframe primative data
glm::vec3* lod1l_vertices;
glm::vec3* lod1l_normals;
glm::ivec3* lod1l_faces;
//right keyframe primative data
glm::vec3* lod1r_vertices;
glm::vec3* lod1r_normals;
glm::ivec3* lod1r_faces;
//buffer objects
GLuint lod1_elem_vbo;
GLuint lod1l_verts_vbo;
GLuint lod1l_norms_vbo;
GLuint lod1r_verts_vbo;
GLuint lod1r_norms_vbo;

//MODEL DATA FOR LOD 2
//primative count
int lod2_v_count;
int lod2_f_count;
//left keyframe primative data
glm::vec3* lod2l_vertices;
glm::vec3* lod2l_normals;
glm::ivec3* lod2l_faces;
//right keyframe primative data
glm::vec3* lod2r_vertices;
glm::vec3* lod2r_normals;
glm::ivec3* lod2r_faces;
//buffer objects
GLuint lod2_elem_vbo;
GLuint lod2l_verts_vbo;
GLuint lod2l_norms_vbo;
GLuint lod2r_verts_vbo;
GLuint lod2r_norms_vbo;

//MODEL DATA FOR LOD 3
//primative count
int lod3_v_count;
int lod3_f_count;
//left keyframe primative data
glm::vec3* lod3l_vertices;
glm::vec3* lod3l_normals;
glm::ivec3* lod3l_faces;
//right keyframe primative data
glm::vec3* lod3r_vertices;
glm::vec3* lod3r_normals;
glm::ivec3* lod3r_faces;
//buffer objects
GLuint lod3_elem_vbo;
GLuint lod3l_verts_vbo;
GLuint lod3r_verts_vbo;
//unused
//GLuint lod31_norms_vbo;
//GLuint lod3r_norms_vbo;

//Shader and shader attributes
GLuint p_vertexShader;
GLuint p_shaderProgram;
GLuint pvs_data1_map;
GLuint pvs_data2_map;
GLuint pvs_instance_index;
GLuint pvs_position_l;
GLuint pvs_position_r;
GLuint pvs_normal_l;
GLuint pvs_normal_r;

//external prototypes imported from FLAME GPU
extern int get_agent_agent_MAX_count();

//PRIVATE PROTOTYPES
/** initPedestrianShader
 * Creates all Buffer Objects for instancing and model data
 */
void initPedestrianShader();
/** createPedestrianBufferObjects
 * Initialises the Pedestrian Shader and shader attributes
 */
void createPedestrianBufferObjects();


void initPedestrianPopulation()
{
	//LOD 1
	lod1_v_count = 354;
	lod1_f_count = 704;
	//Left
	char lod1LeftString[] = "../../media/person-lod1-left.obj"; 
	allocateObjModel(lod1_v_count, lod1_f_count, &lod1l_vertices, &lod1l_normals, &lod1l_faces);
	loadObjFromFile(lod1LeftString,	lod1_v_count, lod1_f_count, lod1l_vertices, lod1l_normals, lod1l_faces);
	scaleObj(PEDESTRIAN_MODEL_SCALE, lod1_v_count, lod1l_vertices);		 
	//Right
	char lod1RightString[] = "../../media/person-lod1-right.obj"; 
	allocateObjModel(lod1_v_count, lod1_f_count, &lod1r_vertices, &lod1r_normals, &lod1r_faces);
	loadObjFromFile(lod1RightString, lod1_v_count, lod1_f_count, lod1r_vertices, lod1r_normals, lod1r_faces);
	scaleObj(PEDESTRIAN_MODEL_SCALE, lod1_v_count, lod1r_vertices);

	//LOD 2
	lod2_v_count = 135;
	lod2_f_count = 266;
	//Left
	char lod2LeftString[] = "../../media/person-lod2-left.obj"; 
	allocateObjModel(lod2_v_count, lod2_f_count, &lod2l_vertices, &lod2l_normals, &lod2l_faces);
	loadObjFromFile(lod2LeftString,	lod2_v_count, lod2_f_count, lod2l_vertices, lod2l_normals, lod2l_faces);
	scaleObj(PEDESTRIAN_MODEL_SCALE, lod2_v_count, lod2l_vertices);		 
	//Right
	char lod2RightString[] = "../../media/person-lod2-right.obj"; 
	allocateObjModel(lod2_v_count, lod2_f_count, &lod2r_vertices, &lod2r_normals, &lod2r_faces);
	loadObjFromFile(lod2RightString, lod2_v_count, lod2_f_count, lod2r_vertices, lod2r_normals, lod2r_faces);
	scaleObj(PEDESTRIAN_MODEL_SCALE, lod2_v_count, lod2r_vertices);

	//LOD 3
	lod3_v_count = 56;//135;
	lod3_f_count = 58;//266;
	//Left
	char lod3LeftString[] = "../../media/person-lod3-left.obj"; 
	allocateObjModel(lod3_v_count, lod3_f_count, &lod3l_vertices, &lod3l_normals, &lod3l_faces);
	loadObjFromFile(lod3LeftString,	lod3_v_count, lod3_f_count, lod3l_vertices, lod3l_normals, lod3l_faces);
	scaleObj(PEDESTRIAN_MODEL_SCALE, lod3_v_count, lod3l_vertices);		 
	//Right
	char lod3RightString[] = "../../media/person-lod3-right.obj"; 
	allocateObjModel(lod3_v_count, lod3_f_count, &lod3r_vertices, &lod3r_normals, &lod3r_faces);
	loadObjFromFile(lod3RightString, lod3_v_count, lod3_f_count, lod3r_vertices, lod3r_normals, lod3r_faces);
	scaleObj(PEDESTRIAN_MODEL_SCALE, lod3_v_count, lod3r_vertices);

	createPedestrianBufferObjects();

	initPedestrianShader();

	initGPULODFeedback();
}


void renderPedestrianPopulation()
{	
	int i;
	int count=0;

	//run CUDA
	generate_instances_and_LOD(&p_instances_data1_tbo, &p_instances_data2_tbo, &p_instances_data1_cgr, &p_instances_data2_cgr);

	glUseProgram(p_shaderProgram);

	//bind instance data
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, p_instances_data1_tex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, p_instances_data2_tex);


	//draw lod 0 geometry
	for (i=0; i<getPedestrianLOD1Count(); i++)
	{
		glVertexAttrib1f(pvs_instance_index, (float)count);
		count++;
		
		glBindBuffer(GL_ARRAY_BUFFER, lod1l_verts_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glEnableVertexAttribArray(pvs_position_r);
		glBindBuffer(GL_ARRAY_BUFFER, lod1r_verts_vbo);
		glVertexAttribPointer(pvs_position_r, 3, GL_FLOAT, 0, 0, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod1_elem_vbo);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
	    
		glDrawElements(GL_TRIANGLES, lod1_f_count*3, GL_UNSIGNED_INT, 0);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);
		glDisableVertexAttribArray(lod1r_verts_vbo);
	}

	//draw lod 1 geometry
	for (i=0; i<getPedestrianLOD2Count(); i++)
	{
		glVertexAttrib1f(pvs_instance_index, (float)count);
		count++;
		
		glBindBuffer(GL_ARRAY_BUFFER, lod2l_verts_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glEnableVertexAttribArray(pvs_position_r);
		glBindBuffer(GL_ARRAY_BUFFER, lod2r_verts_vbo);
		glVertexAttribPointer(pvs_position_r, 3, GL_FLOAT, 0, 0, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod2_elem_vbo);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
	    
		glDrawElements(GL_TRIANGLES, lod2_f_count*3, GL_UNSIGNED_INT, 0);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);
		glDisableVertexAttribArray(lod2r_verts_vbo);
	}

	//draw lod 2 geometry
	for (i=0; i<getPedestrianLOD3Count(); i++)
	{
		glVertexAttrib1f(pvs_instance_index, (float)count);
		count++;
		
		glBindBuffer(GL_ARRAY_BUFFER, lod3l_verts_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glEnableVertexAttribArray(pvs_position_r);
		glBindBuffer(GL_ARRAY_BUFFER, lod3r_verts_vbo);
		glVertexAttribPointer(pvs_position_r, 3, GL_FLOAT, 0, 0, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod3_elem_vbo);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
	    
		glDrawElements(GL_TRIANGLES, lod3_f_count*3, GL_UNSIGNED_INT, 0);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);
		glDisableVertexAttribArray(lod3r_verts_vbo);
	}

	glUseProgram(0);
	
}


void createPedestrianBufferObjects()
{
	//create TBO
	createTBO(&p_instances_data1_tbo, &p_instances_data1_tex, get_agent_agent_MAX_count()* sizeof(glm::vec4));
	createTBO(&p_instances_data2_tbo, &p_instances_data2_tex, get_agent_agent_MAX_count()* sizeof(glm::vec4));
	registerBO(&p_instances_data1_cgr, &p_instances_data1_tbo);
	registerBO(&p_instances_data2_cgr, &p_instances_data2_tbo);

	//create VBOs (LOD1, LOD2, LOD3)
	createVBO(&lod1l_verts_vbo, GL_ARRAY_BUFFER, lod1_v_count*sizeof(glm::vec3));
	createVBO(&lod1_elem_vbo, GL_ELEMENT_ARRAY_BUFFER, lod1_f_count*sizeof(glm::ivec3));
	createVBO(&lod1r_verts_vbo, GL_ARRAY_BUFFER, lod1_v_count*sizeof(glm::vec3));
	createVBO(&lod2l_verts_vbo, GL_ARRAY_BUFFER, lod2_v_count*sizeof(glm::vec3));
	createVBO(&lod2_elem_vbo, GL_ELEMENT_ARRAY_BUFFER, lod2_f_count*sizeof(glm::ivec3));
	createVBO(&lod2r_verts_vbo, GL_ARRAY_BUFFER, lod2_v_count*sizeof(glm::vec3));
	createVBO(&lod3l_verts_vbo, GL_ARRAY_BUFFER, lod3_v_count*sizeof(glm::vec3));
	createVBO(&lod3_elem_vbo, GL_ELEMENT_ARRAY_BUFFER, lod3_f_count*sizeof(glm::ivec3));
	createVBO(&lod3r_verts_vbo, GL_ARRAY_BUFFER, lod3_v_count*sizeof(glm::vec3));

	//bind VBOs LOD1
	glBindBuffer(GL_ARRAY_BUFFER, lod1l_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, lod1_v_count*sizeof(glm::vec3), lod1l_vertices, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod1_elem_vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, lod1_f_count*sizeof(glm::ivec3), lod1l_faces, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, lod1r_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, lod1_v_count*sizeof(glm::vec3), lod1r_vertices, GL_DYNAMIC_DRAW);

	//bind VBOs LOD2
	glBindBuffer(GL_ARRAY_BUFFER, lod2l_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, lod2_v_count*sizeof(glm::vec3), lod2l_vertices, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod2_elem_vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, lod2_f_count*sizeof(glm::ivec3), lod2l_faces, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, lod2r_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, lod2_v_count*sizeof(glm::vec3), lod2r_vertices, GL_DYNAMIC_DRAW);

	//bind VBOs lod3
	glBindBuffer(GL_ARRAY_BUFFER, lod3l_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, lod3_v_count*sizeof(glm::vec3), lod3l_vertices, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lod3_elem_vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, lod3_f_count*sizeof(glm::ivec3), lod3l_faces, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, lod3r_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, lod3_v_count*sizeof(glm::vec3), lod3r_vertices, GL_DYNAMIC_DRAW);
}


void initPedestrianShader()
{
	const char* v = pedestrian_vshader_source;
	int status;

	//vertex shader
	p_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(p_vertexShader, 1, &v, 0);
    glCompileShader(p_vertexShader);


	//program
    p_shaderProgram = glCreateProgram();
    glAttachShader(p_shaderProgram, p_vertexShader);
    glLinkProgram(p_shaderProgram);

	// check for errors

	glGetShaderiv(p_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		char data[1024];
		int len;
		printf("ERROR: Shader Compilation Error\n");
		glGetShaderInfoLog(p_vertexShader, 1024, &len, data); 
		printf("%s", data);
	}
	
	glGetProgramiv(p_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	pvs_data1_map = glGetUniformLocation(p_shaderProgram, "data1_map");
	pvs_data2_map = glGetUniformLocation(p_shaderProgram, "data2_map");
	pvs_instance_index = glGetAttribLocation(p_shaderProgram, "instance_index"); 
	pvs_position_r = glGetAttribLocation(p_shaderProgram, "position_r"); 

	//set shader uniforms
	glUseProgram(p_shaderProgram);
	glUniform1i(pvs_data1_map, 0);
	glUniform1i(pvs_data2_map, 1);
	glUseProgram(0);

}

