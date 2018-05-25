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
#include "NavMapPopulation.h"
#include "OBJModel.h"
#include "BufferObjects.h"

/** Macro for toggling drawing of the navigation map wireframe grid */
int drawGrid = 0;

/** Macro for toggling drawing of the navigation map arrows */
int drawArrows  = 1;

/** Macro for toggling the use of a single large vbo */
BOOLEAN useLargeVBO = TRUE;

//navigation map width
int nm_width;

//navmap instances
GLuint nm_instances_tbo;
GLuint nm_instances_tex;
cudaGraphicsResource_t nm_instances_cgr;

//model primative counts
int arrow_v_count;
int arrow_f_count;
//model primative data
glm::vec3* arrow_vertices;
glm::vec3* arrow_normals;
glm::ivec3* arrow_faces;
//model buffer obejcts
GLuint arrow_verts_vbo;
GLuint arrow_elems_vbo;

//vertex attribute buffer (for single large vbo)
GLuint arrow_attributes_vbo;

//Shader and shader attribute pointers
GLuint nm_vertexShader;
GLuint nm_shaderProgram;
GLuint nmvs_instance_map;
GLuint nmvs_instance_index;
GLuint nmvs_NM_WIDTH;
GLuint nmvs_ENV_MAX;
GLuint nmvs_ENV_WIDTH;

//external prototypes imported from FLAME GPU
extern int get_agent_navmap_MAX_count();
extern int get_agent_navmap_static_count();

//PRIVATE PROTOTYPES
/** createNavMapBufferObjects
 * Creates all Buffer Objects for instancing and model data
 */
void createNavMapBufferObjects();
/** initNavMapShader
 * Initialises the Navigation Map Shader and shader attributes
 */
void initNavMapShader();


void initNavMapPopulation()
{
	float scale;

	nm_width = (int)floor(sqrt((float)get_agent_navmap_MAX_count()));

	arrow_v_count = 25;
	arrow_f_count = 46;

	//load cone model
	allocateObjModel(arrow_v_count, arrow_f_count, &arrow_vertices, &arrow_normals, &arrow_faces);
	loadObjFromFile("../../media/cone.obj",	arrow_v_count, arrow_f_count, arrow_vertices, arrow_normals, arrow_faces);
	scale = ENV_MAX/(float)nm_width;
	scaleObj(scale, arrow_v_count, arrow_vertices);		 
	

	createNavMapBufferObjects();
	initNavMapShader();
	displayMapNumber(0);
}




void renderNavMapPopulation()
{	
	int i, x, y;

	if (drawArrows)
	{
		//generate instance data from FLAME GPU model
		generate_instances(&nm_instances_tbo, &nm_instances_cgr);

		
		//bind vertex program
		glUseProgram(nm_shaderProgram);

		//bind instance data
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, nm_instances_tex);
		glUniform1i(nmvs_instance_map, 0);


		if (useLargeVBO)
		{
			glBindBuffer(GL_ARRAY_BUFFER, arrow_attributes_vbo);
			glEnableVertexAttribArray(nmvs_instance_index);
			glVertexAttribPointer(nmvs_instance_index, 1, GL_FLOAT, 0, 0, 0);

			glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo);
			glVertexPointer(3, GL_FLOAT, 0, 0);
			
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo);

			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
		    
			glDrawElements(GL_TRIANGLES, arrow_f_count*3*get_agent_navmap_static_count(), GL_UNSIGNED_INT, 0);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);
			glDisableVertexAttribArray(nmvs_instance_index);
		}
		else
		{
			//draw arrows
			for (i=0; i<get_agent_navmap_static_count(); i++)
			{
				glVertexAttrib1f(nmvs_instance_index, (float)i);
				
				glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo);
				glVertexPointer(3, GL_FLOAT, 0, 0);

				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo);

				glEnableClientState(GL_VERTEX_ARRAY);
				glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
			    
				glDrawElements(GL_TRIANGLES, arrow_f_count*3, GL_UNSIGNED_INT, 0);

				glDisableClientState(GL_VERTEX_ARRAY);
				glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);

			}
		}

		glUseProgram(0);
	}

	if (drawGrid)
	{
		//draw line grid
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBegin(GL_QUADS);
		{
			for (y=0; y<nm_width; y++){
				for (x=0; x<nm_width; x++){
					float x_min = (float)(x)/((float)nm_width/(float)ENV_WIDTH)-ENV_MAX;
					float x_max = (float)(x+1)/((float)nm_width/(float)ENV_WIDTH)-ENV_MAX;
					float y_min = (float)(y)/((float)nm_width/(float)ENV_WIDTH)-ENV_MAX;
					float y_max = (float)(y+1)/((float)nm_width/(float)ENV_WIDTH)-ENV_MAX;

					glVertex2f(x_min, y_min);
					glVertex2f(x_min, y_max);
					glVertex2f(x_max, y_max);
					glVertex2f(x_max, y_min);
				}
			}
		}
		glEnd();
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
}


void createNavMapBufferObjects()
{
	//create TBO
	createTBO(&nm_instances_tbo, &nm_instances_tex, get_agent_navmap_MAX_count()* sizeof(glm::vec4));
	registerBO(&nm_instances_cgr, &nm_instances_tbo);

	if (useLargeVBO)
	{
		int i,v,f = 0;
		glm::vec3* verts;
		glm::ivec3* faces;
		float* atts;

		//create VBOs
		createVBO(&arrow_verts_vbo, GL_ARRAY_BUFFER, get_agent_navmap_MAX_count()*arrow_v_count*sizeof(glm::vec3));
		createVBO(&arrow_elems_vbo, GL_ELEMENT_ARRAY_BUFFER, get_agent_navmap_MAX_count()*arrow_f_count*sizeof(glm::ivec3));
		//create attributes vbo
		createVBO(&arrow_attributes_vbo, GL_ARRAY_BUFFER, get_agent_navmap_MAX_count()*arrow_v_count*sizeof(int));
		
		
		//bind and map vertex data
		glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo);
		verts = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (i=0;i<get_agent_navmap_MAX_count();i++){
			int offset = i*arrow_v_count;
			// int x = floor(i/64.0f);
			// int y = i%64;
			for (v=0;v<arrow_v_count;v++){
				verts[offset+v][0] = arrow_vertices[v][0];
				verts[offset+v][1] = arrow_vertices[v][1];
				verts[offset+v][2] = arrow_vertices[v][2];
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer( GL_ARRAY_BUFFER, 0);

		//bind and map face data
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo);
		faces = (glm::ivec3*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (i=0;i<get_agent_navmap_MAX_count();i++){
			int offset = i*arrow_f_count;
			int vert_offset = i*arrow_v_count;	//need to offset all face indices by number of verts in each model
			for (f=0;f<arrow_f_count;f++){
				faces[offset+f][0] = arrow_faces[f][0]+vert_offset;
				faces[offset+f][1] = arrow_faces[f][1]+vert_offset;
				faces[offset+f][2] = arrow_faces[f][2]+vert_offset;
			}
		}
		glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0);

		
		//bind and map vbo attrbiute data
		glBindBuffer(GL_ARRAY_BUFFER, arrow_attributes_vbo);
		atts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (i=0;i<get_agent_navmap_MAX_count();i++){
			int offset = i*arrow_v_count;
			for (v=0;v<arrow_v_count;v++){
				atts[offset+v] = (float)i;
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer( GL_ARRAY_BUFFER, 0);
		

		checkGLError();
	}
	else
	{
		//create VBOs
		createVBO(&arrow_verts_vbo, GL_ARRAY_BUFFER, arrow_v_count*sizeof(glm::vec3));
		createVBO(&arrow_elems_vbo, GL_ELEMENT_ARRAY_BUFFER, arrow_f_count*sizeof(glm::ivec3));

		//bind VBOs
		glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo);
		glBufferData(GL_ARRAY_BUFFER, arrow_v_count*sizeof(glm::vec3), arrow_vertices, GL_DYNAMIC_DRAW);
		glBindBuffer( GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, arrow_f_count*sizeof(glm::ivec3), arrow_faces, GL_DYNAMIC_DRAW);
		glBindBuffer( GL_ARRAY_BUFFER, 0);
	}
	

}

void initNavMapShader()
{
	const char* v = navmap_vshader_source;
	int status;

	//vertex shader
	nm_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(nm_vertexShader, 1, &v, 0);
    glCompileShader(nm_vertexShader);

	//program
    nm_shaderProgram = glCreateProgram();
    glAttachShader(nm_shaderProgram, nm_vertexShader);
    glLinkProgram(nm_shaderProgram);

	// check for errors
	glGetShaderiv(nm_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		char data[1024];
		int len;
		printf("ERROR: Shader Compilation Error\n");
		glGetShaderInfoLog(nm_vertexShader, 1024, &len, data); 
		printf("%s", data);
	}
	glGetProgramiv(nm_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	nmvs_instance_map = glGetUniformLocation(nm_shaderProgram, "instance_map");
	nmvs_instance_index = glGetAttribLocation(nm_shaderProgram, "instance_index"); 
	nmvs_NM_WIDTH = glGetUniformLocation(nm_shaderProgram, "NM_WIDTH");
	nmvs_ENV_MAX = glGetUniformLocation(nm_shaderProgram, "ENV_MAX");
	nmvs_ENV_WIDTH = glGetUniformLocation(nm_shaderProgram, "ENV_WIDTH");

	//set uniforms (need to use prgram to do so)
	glUseProgram(nm_shaderProgram);
	glUniform1f(nmvs_NM_WIDTH, (float)nm_width);
	glUniform1f(nmvs_ENV_MAX, ENV_MAX);
	glUniform1f(nmvs_ENV_WIDTH, ENV_WIDTH);
	glUseProgram(0);
}

void toggleGridDisplayOnOff()
{
	drawGrid = !drawGrid;
}

void setArrowsDisplayOnOff(TOGGLE_STATE state)
{
	drawArrows = state;
}


void toggleArrowsDisplayOnOff()
{
	drawArrows = !drawArrows;
}

int getActiveExit()
{
	return getCurrentMap();
}
