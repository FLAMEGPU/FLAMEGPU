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
typedef float float3[3];
typedef float float4[4];
typedef int int3[3];
typedef int int4[4];

//OBJ1 FORMAT FUNCTIONS
void allocateObjModel(int vertex_count, int face_count, float3** vertices, float3** normals, int3** faces);
void cleanupObjModel(float3** vertices, float3** normals, int3** faces);
void loadObjFromFile(char* name, int vertex_count, int face_count, float3* vertices, float3* normals, int3* faces);
void scaleObj(float scale_factor, int vertex_count, float3* vertices);
void drawObj(int vertex_count, int face_count, float3* vertices, float3* normals, int3* faces);

//OBJ2 FORMAT FUNCTIONS
void xAllocateObjModel(int vertex_count, int normal_count, int face_count, float3** vertices, float3** normals, int4** faces);
void xCleanupObjModel(float3** vertices, float3** normals, int4** faces);
void xLoadObjFromFile(char* name, int vertex_count, int normal_count, int face_count, float3* vertices, float3* normals, int4* faces);
void xScaleObj(float scale_factor, int vertex_count, float3* vertices);
void xDrawObj(int vertex_count, int normal_count, int face_count, float3* vertices, float3* normals, int4* faces);


#endif __OBJMODEL