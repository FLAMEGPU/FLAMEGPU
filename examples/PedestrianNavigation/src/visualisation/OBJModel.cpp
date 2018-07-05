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

#include <GL/glew.h>
#include <GL/glut.h>

#include "OBJModel.h"

#ifdef _MSC_VER
// Disable _CRT_SECURE_NO_WARNINGS warnings
#pragma warning(disable:4996)
#endif

void allocateObjModel(int vertex_count, int face_count, glm::vec3** vertices, glm::vec3** normals, glm::ivec3** faces){
	*vertices = (glm::vec3*)malloc(vertex_count*sizeof(glm::vec3));
	*normals = (glm::vec3*)malloc(vertex_count*sizeof(glm::vec3));
	*faces = (glm::ivec3*)malloc(face_count*sizeof(glm::ivec3));
}

void cleanupObjModel(glm::vec3** vertices, glm::vec3** normals, glm::ivec3** faces){
	free(*vertices);
	free(*normals);
	free(*faces);
}

void loadObjFromFile(char* name, int vertex_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec3* faces){
	//placeholders
	char text[100];
	float x,y,z;
	int f1a, f1b, f2a, f2b, f3a, f3b;

	//counters
	int verts_read = 0; 
	int normals_read = 0; 
	int faces_read = 0;

	//open file
	FILE* file = fopen(name, "r");
	if (file == NULL){
		printf("Could not open file '%s'!\n", name);
		return;
	}



	while (!feof(file)) {
		if (fscanf(file, "%s", text) == 1){
			if(strcmp(text, VERTEX_IDENTIFIER) == 0){
				//expect 3 vertices
				if (fscanf(file, "%f %f %f", &x, &y, &z) == 3){
					//save data
					vertices[verts_read][0] = x;
					vertices[verts_read][1] = y;
					vertices[verts_read][2] = z;
					verts_read++;
					//printf("%f %f %f\n", x, y, z);
				}else{
					printf("Incomplete vertex data\n");
				}
						
			}else if(strcmp(text, VERTEX_NORMAL_IDENTIFIER) == 0){
				//expect 3 vertices
				if (fscanf(file, "%f %f %f", &x, &y, &z) == 3){
					//save data
					normals[normals_read][0] = x;
					normals[normals_read][1] = y;
					normals[normals_read][2] = z;
					normals_read++;
					//printf("%f %f %f\n", x, y, z);
				}else{
					printf("Incomplete vertex normal data\n");
				}
						
			}else if(strcmp(text, FACE_IDENTIFIER) == 0){
				//expect 3 vertices
				if (fscanf(file, "%i//%i %i//%i %i//%i", &f1a, &f1b, &f2a, &f2b, &f3a, &f3b) == 6){
					//save data
					faces[faces_read][0] = f1a-1;
					faces[faces_read][1] = f2a-1;
					faces[faces_read][2] = f3a-1;
					faces_read++;
					//printf("%i %i %i\n", f1a, f2a, f3a);
				}else{
					printf("Incomplete face data\n");
				}	
			}
		}
			
	}

	if (vertex_count != verts_read)
		printf("Found %i vertices, expected %i!\n", verts_read, vertex_count);

	if (vertex_count != normals_read)
		printf("Found %i normals, expected %i!\n", normals_read, vertex_count);

	if (face_count != faces_read)
		printf("Found %i faces, expected %i!\n", faces_read, face_count);

}


void scaleObj(float scale_factor, int vertex_count, glm::vec3* vertices){
	int i;
	for (i=0; i < vertex_count; i++){
		vertices[i][0] *= scale_factor;
		vertices[i][1] *= scale_factor;
		vertices[i][2] *= scale_factor;
	}
}

void drawObj(int vertex_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec3* faces){
	int f;
	glBegin(GL_TRIANGLES);
	for (f=0; f<face_count; f++){
			int v1 = faces[f][0];
			int v2 = faces[f][1];
			int v3 = faces[f][2];

			glNormal3f(normals[v1][0], normals[v1][1], normals[v1][2]);
			glVertex3f(vertices[v1][0], vertices[v1][1], vertices[v1][2]);

			glNormal3f(normals[v2][0], normals[v2][1], normals[v2][2]);
			glVertex3f(vertices[v2][0], vertices[v2][1], vertices[v2][2]);

			glNormal3f(normals[v3][0], normals[v3][1], normals[v3][2]);
			glVertex3f(vertices[v3][0], vertices[v3][1], vertices[v3][2]);
		}
	glEnd();
}

/**
* SUPPORT LOADING NEGATIVE INDICES OPTIMIZED NORMALS
*/

void xAllocateObjModel(int vertex_count, int normal_count, int face_count, glm::vec3** vertices, glm::vec3** normals, glm::ivec4** faces){
	*vertices = (glm::vec3*)malloc(vertex_count*sizeof(glm::vec3));
	*normals = (glm::vec3*)malloc(normal_count*sizeof(glm::vec3));
	*faces = (glm::ivec4*)malloc(face_count*sizeof(glm::ivec4));
}
void xCleanupObjModel(glm::vec3** vertices, glm::vec3** normals, glm::ivec4** faces){
	free(*vertices);
	free(*normals);
	free(*faces);
}

void xLoadObjFromFile(char* name, int vertex_count, int normal_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec4* faces){
	//placeholders
	char text[100];
	float x,y,z;
	int f1a, f1b, f2a, f2b, f3a, f3b;

	//counters
	int verts_read = 0; 
	int normals_read = 0; 
	int faces_read = 0;

	//open file
	FILE* file = fopen(name, "r");
	if (file == NULL){
		printf("Could not open file '%s'!\n", name);
		return;
	}



	while (!feof(file)) {
		if (fscanf(file, "%s", text) == 1)
		{
			if(strcmp(text, VERTEX_IDENTIFIER) == 0)
			{
				//expect 3 vertices
				if (fscanf(file, "%f %f %f", &x, &y, &z) == 3)
				{
					//save data
					vertices[verts_read][0] = x;
					vertices[verts_read][1] = y;
					vertices[verts_read][2] = z;
					verts_read++;
					//printf("%f %f %f\n", x, y, z);
				}else{
					printf("Incomplete vertex data\n");
				}
						
			}else if(strcmp(text, VERTEX_NORMAL_IDENTIFIER) == 0)
			{
				//expect 3 vertices
				if (fscanf(file, "%f %f %f", &x, &y, &z) == 3){
					//save data
					normals[normals_read][0] = x;
					normals[normals_read][1] = y;
					normals[normals_read][2] = z;
					normals_read++;
					//printf("%f %f %f\n", x, y, z);
				}else{
					printf("Incomplete vertex normal data\n");
				}
						
			}else if(strcmp(text, FACE_IDENTIFIER) == 0)
			{
				//expect 3 vertices
				if (fscanf(file, "%i//%i %i//%i %i//%i", &f1a, &f1b, &f2a, &f2b, &f3a, &f3b) == 6){
					//save data
					if( f1a > 0) faces[faces_read][0] = f1a - 1;
					else faces[faces_read][0] = verts_read + f1a;

					if( f2a > 0) faces[faces_read][1] = f2a - 1;
					else faces[faces_read][1] = verts_read + f2a;

					if( f3a > 0) faces[faces_read][2] = f3a - 1;
					else faces[faces_read][2] = verts_read + f3a;

					if( f1b > 0) faces[faces_read][3] = f1b - 1;
					else faces[faces_read][3] = normals_read + f1b;

					faces_read++;
					//printf("%i %i %i\n", f1a, f2a, f3a);
				}else{
					printf("Incomplete face data\n");
				}	
			}
		}
			
	}

	if (vertex_count != verts_read)
		printf("Found %i vertices, expected %i!\n", verts_read, vertex_count);

	if (normal_count != normals_read)
		printf("Found %i normals, expected %i!\n", normals_read, vertex_count);

	if (face_count != faces_read)
		printf("Found %i faces, expected %i!\n", faces_read, face_count);

}


void xScaleObj(float scale_factor, int vertex_count, glm::vec3* vertices){
	int i;
	for (i=0; i < vertex_count; i++){
		vertices[i][0] *= scale_factor;
		vertices[i][1] *= scale_factor;
		vertices[i][2] *= scale_factor;
	}
}

void xDrawObj(int vertex_count, int normal_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec4* faces){
	int f;
	glBegin(GL_TRIANGLES);
	if(normal_count != vertex_count)
	{
		//Draws 1 normal triangle
		for (f=0; f<face_count; f++){
			int v1 = faces[f][0];
			int v2 = faces[f][1];
			int v3 = faces[f][2];
			int n1 = faces[f][3];

			glNormal3f(normals[n1][0], normals[n1][1], normals[n1][2]);
			glVertex3f(vertices[v1][0], vertices[v1][1], vertices[v1][2]);
			glVertex3f(vertices[v2][0], vertices[v2][1], vertices[v2][2]);
			glVertex3f(vertices[v3][0], vertices[v3][1], vertices[v3][2]);
		}
	}
	else
	{
		//Draws a normal for each vertex
		for (f=0; f<face_count; f++){
			int v1 = faces[f][0];
			int v2 = faces[f][1];
			int v3 = faces[f][2];

			glNormal3f(normals[v1][0], normals[v1][1], normals[v1][2]);
			glVertex3f(vertices[v1][0], vertices[v1][1], vertices[v1][2]);

			glNormal3f(normals[v2][0], normals[v2][1], normals[v2][2]);
			glVertex3f(vertices[v2][0], vertices[v2][1], vertices[v2][2]);

			glNormal3f(normals[v3][0], normals[v3][1], normals[v3][2]);
			glVertex3f(vertices[v3][0], vertices[v3][1], vertices[v3][2]);
		}
	}

	
	glEnd();
}
