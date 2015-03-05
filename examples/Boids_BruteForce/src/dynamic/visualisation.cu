
/*
* FLAME GPU v 1.4.0 for CUDA 6
* Copyright 2015 University of Sheffield.
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

// includes, project
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
	
#include <vector_operators.h>
    
#include "header.h"
#include "visualisation.h"

// bo variables
GLuint sphereVerts;
GLuint sphereNormals;

//Simulation output buffers/textures

GLuint Boid_default_tbo;
GLuint Boid_default_displacementTex;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;

// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_displacementMap;
GLuint vs_mapIndex;



//timer
cudaEvent_t start, stop;
const int average = 50;
int frame_count;

#ifdef SIMULATION_DELAY
//delay
int delay_count = 0;
#endif

// prototypes
int initGL();
void initShader();
void createVBO( GLuint* vbo, GLuint size);
void deleteVBO( GLuint* vbo);
void createTBO( GLuint* tbo, GLuint* tex, GLuint size);
void deleteTBO( GLuint* tbo);
void setVertexBufferData();
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError();

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

const char vertexShaderSource[] = 
{  
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
    "void main()																\n"
    "{																			\n"
	"	vec4 position = gl_Vertex;											    \n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
    "	if (lookup.w > 6.5)	                								\n"
	"		colour = vec4(1.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 5.5)	                								\n"
	"		colour = vec4(1.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w > 4.5)	                								\n"
	"		colour = vec4(0.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 3.5)	                								\n"
	"		colour = vec4(1.0, 1.0, 0.0, 0.0);								    \n"
	"	else if (lookup.w > 2.5)	                								\n"
	"		colour = vec4(0.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w > 1.5)	                								\n"
	"		colour = vec4(0.0, 1.0, 0.0, 0.0);								    \n"
    "	else if (lookup.w > 0.5)	                								\n"
	"		colour = vec4(1.0, 0.0, 0.0, 0.0);								    \n"
    "	else                      	                								\n"
	"		colour = vec4(0.0, 0.0, 0.0, 0.0);								    \n"
	"																    		\n"
	"	lookup.w = 1.0;												    		\n"
	"	position += lookup;											    		\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n"
	"																			\n"
	"	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);			    	\n"
	"	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);				\n"
	"	normal = gl_NormalMatrix * gl_Normal;									\n"
    "}																			\n"
};

const char fragmentShaderSource[] = 
{  
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	// Defining The Material Colors											\n"
	"	vec4 AmbientColor = vec4(0.25, 0.0, 0.0, 1.0);					\n"
	"	vec4 DiffuseColor = colour;					                	\n"
	"																			\n"
	"	// Scaling The Input Vector To Length 1									\n"
	"	vec3 n_normal = normalize(normal);							        	\n"
	"	vec3 n_lightDir = normalize(lightDir);	                                \n"
	"																			\n"
	"	// Calculating The Diffuse Term And Clamping It To [0;1]				\n"
	"	float DiffuseTerm = clamp(dot(n_normal, n_lightDir), 0.0, 1.0);\n"
	"																			\n"
	"	// Calculating The Final Color											\n"
	"	gl_FragColor = AmbientColor + DiffuseColor * DiffuseTerm;				\n"
	"																			\n"
	"}																			\n"
};

//GPU Kernels

__global__ void output_Boid_agent_to_VBO(xmachine_memory_Boid_list* agents, float4* vbo, float3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
	vbo[index].x = agents->x[index] - centralise.x;
	vbo[index].y = agents->y[index] - centralise.y;
	vbo[index].z = agents->z[index] - centralise.z;
	vbo[index].w = 1.0;
}


void initVisualisation()
{
	//set the CUDA GL device: Will cause an error without this since CUDA 3.0
	cudaGLSetGLDevice(0);

	// Create GL context
	int   argc   = 1;
	char *argv[] = {"GLUT application", NULL};
	glutInit( &argc, argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow( "FLAME GPU Visualiser");

	// initialize GL
	if( !initGL()) {
			return;
	}
	initShader();

	// register callbacks
	glutDisplayFunc( display);
	glutKeyboardFunc( keyboard);
	glutMouseFunc( mouse);
	glutMotionFunc( motion);
    
	// create VBO's
	createVBO( &sphereVerts, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof(float3));
	createVBO( &sphereNormals, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof (float3));
	setVertexBufferData();

	// create TBO
	createTBO( &Boid_default_tbo, &Boid_default_displacementTex, xmachine_memory_Boid_MAX * sizeof( float4));
	

	//set shader uniforms
	glUseProgram(shaderProgram);

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void runVisualisation(){
	// start rendering mainloop
	glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
#ifdef SIMULATION_DELAY
	delay_count++;
	if (delay_count == SIMULATION_DELAY){
		delay_count = 0;
		singleIteration();
	}
#else
	singleIteration();
#endif

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;
	float3 centralise;

	//pointer
	float4 *dptr;

	
	if (get_agent_Boid_default_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGLMapBufferObject( (void**)&dptr, Boid_default_tbo));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Boid_default_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Boid_agent_to_VBO<<< grid, threads>>>(get_device_Boid_default_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGLUnmapBufferObject(Boid_default_tbo));
	}
	
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
	// initialize necessary OpenGL extensions
	glewInit();
	if (! glewIsSupported( "GL_VERSION_2_0 " 
		"GL_ARB_pixel_buffer_object")) {
		fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush( stderr);
		return 1;
	}

	// default initialization
	glClearColor( 1.0, 1.0, 1.0, 1.0);
	glEnable( GL_DEPTH_TEST);

	// viewport
	glViewport( 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	// projection
	glMatrixMode( GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLfloat)WINDOW_WIDTH / (GLfloat) WINDOW_HEIGHT, NEAR_CLIP, FAR_CLIP);

	checkGLError();

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GLSL Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void initShader()
{
	const char* v = vertexShaderSource;
	const char* f = fragmentShaderSource;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
	glCompileShader(fragmentShader);

	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data); 
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &len, data); 
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex"); 
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint size)
{
	// create buffer object
	glGenBuffers( 1, vbo);
	glBindBuffer( GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData( GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	glBindBuffer( GL_ARRAY_BUFFER, 0);

	checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
	glBindBuffer( 1, *vbo);
	glDeleteBuffers( 1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create TBO
////////////////////////////////////////////////////////////////////////////////
void createTBO(GLuint* tbo, GLuint* tex, GLuint size)
{
	// create buffer object
	glGenBuffers( 1, tbo);
	glBindBuffer( GL_TEXTURE_BUFFER_EXT, *tbo);

	// initialize buffer object
	glBufferData( GL_TEXTURE_BUFFER_EXT, size, 0, GL_DYNAMIC_DRAW);

	//tex
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo); 
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

    // register buffer object with CUDA
    gpuErrchk(cudaGLRegisterBufferObject(*tbo));

    checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete TBO
////////////////////////////////////////////////////////////////////////////////
void deleteTBO( GLuint* tbo)
{
	glBindBuffer( 1, *tbo);
	glDeleteBuffers( 1, tbo);

	gpuErrchk(cudaGLUnregisterBufferObject(*tbo));

	*tbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Vertex Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereVertex(float3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
	double st = 2*PI*stack/SPHERE_STACKS;
 
	data->x = cos(st)*sin(sl) * SPHERE_RADIUS;
	data->y = sin(st)*sin(sl) * SPHERE_RADIUS;
	data->z = cos(sl) * SPHERE_RADIUS;
}


////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Normal Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereNormal(float3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
	double st = 2*PI*stack/SPHERE_STACKS;
 
	data->x = cos(st)*sin(sl);
	data->y = sin(st)*sin(sl);
	data->z = cos(sl);
}


////////////////////////////////////////////////////////////////////////////////
//! Set Vertex Buffer Data
////////////////////////////////////////////////////////////////////////////////
void setVertexBufferData()
{
	int slice, stack;
	int i;

	// upload vertex points data
	glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
	float3* verts =( float3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereVertex(&verts[i++], slice, stack);
			setSphereVertex(&verts[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	float3* normals =( float3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereNormal(&normals[i++], slice, stack);
			setSphereNormal(&normals[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	float millis;
	
	//CUDA start Timing
	cudaEventRecord(start);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	//zoom
	glTranslatef(0.0, 0.0, translate_z); 
	//move
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);


	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);

	
	//Draw Boid Agents in default state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Boid_default_displacementTex);
	//loop
	for (int i=0; i< get_agent_Boid_default_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	

	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	if(frame_count == average){
		char title [100];
		sprintf(title, "Execution & Rendering Total: %f (FPS)", average/(millis/1000.0f));
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
	}else{
		frame_count++;
	}


	glutSwapBuffers();
	glutPostRedisplay();

}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
	switch( key) {
	case( 27) :
		deleteVBO( &sphereVerts);
		deleteVBO( &sphereNormals);
		
		deleteTBO( &Boid_default_tbo);
		
		cudaEventDestroy(start);
    cudaEventDestroy(stop);
		exit( 0);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2;
		rotate_y += dx * 0.2;
	} else if (mouse_buttons & 4) {
		translate_z += dy * VIEW_DISTANCE* 0.001;
	}

  mouse_old_x = x;
  mouse_old_y = y;
}

void checkGLError(){
  int Error;
  if((Error = glGetError()) != GL_NO_ERROR)
  {
    const char* Message = (const char*)gluErrorString(Error);
    fprintf(stderr, "OpenGL Error : %s\n", Message);
  }
}
