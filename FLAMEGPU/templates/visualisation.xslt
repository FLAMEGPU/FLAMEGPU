<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:template match="/">
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
#include &lt;cuda_runtime.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;cmath&gt;

#include &lt;GL/glew.h&gt;
#include &lt;GL/glut.h&gt;
#include &lt;cuda_gl_interop.h&gt;
	    
#include "header.h"
#include "visualisation.h"

// bo variables
GLuint sphereVerts;
GLuint sphereNormals;

//Simulation output buffers/textures
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
GLuint <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_tbo;
GLuint <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_displacementTex;
</xsl:for-each>

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
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;

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
    "	if (lookup.w &gt; 6.5)	                								\n"
	"		colour = vec4(1.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w &gt; 5.5)	                								\n"
	"		colour = vec4(1.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w &gt; 4.5)	                								\n"
	"		colour = vec4(0.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w &gt; 3.5)	                								\n"
	"		colour = vec4(1.0, 1.0, 0.0, 0.0);								    \n"
	"	else if (lookup.w &gt; 2.5)	                								\n"
	"		colour = vec4(0.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w &gt; 1.5)	                								\n"
	"		colour = vec4(0.0, 1.0, 0.0, 0.0);								    \n"
    "	else if (lookup.w &gt; 0.5)	                								\n"
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
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
__global__ void output_<xsl:value-of select="xmml:name"/>_agent_to_VBO(xmachine_memory_<xsl:value-of select="xmml:name"/>_list* agents, glm::vec4* vbo, glm::vec3 centralise<xsl:if test="gpu:type='discrete'">, int population_width</xsl:if>){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
	<xsl:choose>
		<xsl:when test="xmml:memory/gpu:variable/xmml:name='x'">vbo[index].x = agents->x[index] - centralise.x;</xsl:when>
    <xsl:when test="gpu:type='discrete' and xmml:memory/gpu:variable/xmml:name='location_id'">vbo[index].x = (agents->location_id[index] % population_width) - centralise.x;</xsl:when>
		<xsl:otherwise>vbo[index].x = 0.0;</xsl:otherwise>
	</xsl:choose><xsl:text>
	</xsl:text><xsl:choose>
		<xsl:when test="xmml:memory/gpu:variable/xmml:name='y'">vbo[index].y = agents->y[index] - centralise.y;</xsl:when>
    <xsl:when test="gpu:type='discrete' and xmml:memory/gpu:variable/xmml:name='location_id'">vbo[index].y = floor((float)agents->location_id[index] / (float)population_width) - centralise.y;</xsl:when>
		<xsl:otherwise>vbo[index].y = 0.0;</xsl:otherwise>
	</xsl:choose><xsl:text>
	</xsl:text><xsl:choose>
		<xsl:when test="xmml:memory/gpu:variable/xmml:name='z'">vbo[index].z = agents->z[index] - centralise.z;</xsl:when>
		<xsl:otherwise>vbo[index].z = 0.0;</xsl:otherwise>
	</xsl:choose><xsl:text>
	</xsl:text><xsl:choose>
		<xsl:when test="xmml:memory/gpu:variable/xmml:name='state'">vbo[index].w = agents->state[index];</xsl:when>
        <xsl:when test="xmml:memory/gpu:variable/xmml:name='type'">vbo[index].w = agents->type[index];</xsl:when>
		<xsl:otherwise>vbo[index].w = 1.0;</xsl:otherwise>
	</xsl:choose>
}
</xsl:for-each>

void initVisualisation()
{
	//set the CUDA GL device: Will cause an error without this since CUDA 3.0
	cudaGLSetGLDevice(0);

	// Create GL context
	int   argc   = 1;
	char *argv[] = {"GLUT application", NULL};
	glutInit( &amp;argc, argv);
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
	createVBO( &amp;sphereVerts, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof(glm::vec3));
	createVBO( &amp;sphereNormals, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof (glm::vec3));
	setVertexBufferData();

	// create TBO<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
	createTBO( &amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_tbo, &amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_displacementTex, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX * sizeof( glm::vec4));
	</xsl:for-each>

	//set shader uniforms
	glUseProgram(shaderProgram);

	//create a events for timer
	cudaEventCreate(&amp;start);
	cudaEventCreate(&amp;stop);
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
	glm::vec3 centralise;

	//pointer
	glm::vec4 *dptr;

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
	if (get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGLMapBufferObject( (void**)&amp;dptr, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_tbo));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        <xsl:choose>
        <xsl:when test="../../gpu:type='discrete'">//discrete variables
        int population_width = (int)floor(sqrt((float)get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()));
		centralise.x = population_width / 2.0;
        centralise.y = population_width / 2.0;
        centralise.z = 0.0;
        </xsl:when>
        <xsl:otherwise>
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        </xsl:otherwise>
        </xsl:choose>
		output_<xsl:value-of select="../../xmml:name"/>_agent_to_VBO&lt;&lt;&lt; grid, threads&gt;&gt;&gt;(get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), dptr, centralise<xsl:if test="../../gpu:type='discrete'">, population_width</xsl:if>);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGLUnmapBufferObject(<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_tbo));
	}
	</xsl:for-each>
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
		fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
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
	glShaderSource(vertexShader, 1, &amp;v, 0);
	glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &amp;f, 0);
	glCompileShader(fragmentShader);

	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &amp;status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &amp;len, data); 
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &amp;status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &amp;len, data); 
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &amp;status);
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

static void setSphereVertex(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
	double st = 2*PI*stack/SPHERE_STACKS;
 
	data-&gt;x = cos(st)*sin(sl) * SPHERE_RADIUS;
	data-&gt;y = sin(st)*sin(sl) * SPHERE_RADIUS;
	data-&gt;z = cos(sl) * SPHERE_RADIUS;
}


////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Normal Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereNormal(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
	double st = 2*PI*stack/SPHERE_STACKS;
 
	data-&gt;x = cos(st)*sin(sl);
	data-&gt;y = sin(st)*sin(sl);
	data-&gt;z = cos(sl);
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
	glm::vec3* verts =( glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice&lt;SPHERE_SLICES/2; slice++) {
		for (stack=0; stack&lt;=SPHERE_STACKS; stack++) {
			setSphereVertex(&amp;verts[i++], slice, stack);
			setSphereVertex(&amp;verts[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	glm::vec3* normals =( glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice&lt;SPHERE_SLICES/2; slice++) {
		for (stack=0; stack&lt;=SPHERE_STACKS; stack++) {
			setSphereNormal(&amp;normals[i++], slice, stack);
			setSphereNormal(&amp;normals[i++], slice+1, stack);
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

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
	//Draw <xsl:value-of select="../../xmml:name"/> Agents in <xsl:value-of select="xmml:name"/> state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_displacementTex);
	//loop
	for (int i=0; i&lt; get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count(); i++){
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
	</xsl:for-each>

	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&amp;millis, start, stop);
  frame_time += millis;

	if(frame_count == display_rate){
		char title [100];
		sprintf(title, "Execution &amp; Rendering Total: %f (FPS), %f milliseconds per frame", display_rate/(frame_time/1000.0f), frame_time/display_rate);
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
    frame_time = 0.0;
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
		deleteVBO( &amp;sphereVerts);
		deleteVBO( &amp;sphereNormals);
		<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
		deleteTBO( &amp;<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_tbo);
		</xsl:for-each>
		cudaEventDestroy(start);
    cudaEventDestroy(stop);
		exit(EXIT_SUCCESS);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1&lt;&lt;button;
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

	if (mouse_buttons &amp; 1) {
		rotate_x += dy * 0.2;
		rotate_y += dx * 0.2;
	} else if (mouse_buttons &amp; 4) {
		translate_z += dy * VIEW_DISTANCE * 0.001;
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
</xsl:template>
</xsl:stylesheet>
