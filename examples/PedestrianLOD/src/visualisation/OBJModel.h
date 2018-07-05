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
#ifndef __OBJMODEL
#define __OBJMODEL

//Identifiers used for reading data
static const char* VERTEX_IDENTIFIER = "v";
static const char* VERTEX_NORMAL_IDENTIFIER = "vn";
static const char* FACE_IDENTIFIER = "f";

//Type definitions
#include <glm/glm.hpp>

//OBJ1 FORMAT FUNCTIONS
void allocateObjModel(int vertex_count, int face_count, glm::vec3** vertices, glm::vec3** normals, glm::ivec3** faces);
void cleanupObjModel(glm::vec3** vertices, glm::vec3** normals, glm::ivec3** faces);
void loadObjFromFile(char* name, int vertex_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec3* faces);
void scaleObj(float scale_factor, int vertex_count, glm::vec3* vertices);
void drawObj(int vertex_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec3* faces);

//OBJ2 FORMAT FUNCTIONS
void xAllocateObjModel(int vertex_count, int normal_count, int face_count, glm::vec3** vertices, glm::vec3** normals, glm::ivec4** faces);
void xCleanupObjModel(glm::vec3** vertices, glm::vec3** normals, glm::ivec4** faces);
void xLoadObjFromFile(char* name, int vertex_count, int normal_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec4* faces);
void xScaleObj(float scale_factor, int vertex_count, glm::vec3* vertices);
void xDrawObj(int vertex_count, int normal_count, int face_count, glm::vec3* vertices, glm::vec3* normals, glm::ivec4* faces);


#endif //__OBJMODEL
