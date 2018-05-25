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

// includes, project
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include <time.h>

#include "CustomVisualisation.h"
#include "GLUTInputController.h"
#include "PedestrianPopulation.h"
#include "MenuDisplay.h"
#include "GlobalsController.h"

int window_width = 800;
int window_height = 600;

//full screen mode
int fullScreenMode;

//light
GLfloat lightPosition[] = {25.0, 25.0f, 25.0f, 1.0f};

//framerate
float start_time;
float end_time;
float frame_time;
float fps;
int frames;
int av_frames;

extern void initVisualisation();
extern void runVisualisation();


extern void set_EYE_X(float* eye_x);
extern void set_EYE_Y(float* eye_y);
extern void set_EYE_Z(float* eye_z);


extern void initVisualisation()
{



    // Create GL context
   int   argc   = 1;
   char glutString[] = "GLUT application"; 
   char *argv[] = {glutString, NULL};
   //char *argv[] = {"GLUT application", NULL};	
	
    glutInit(&argc, argv);


    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "FLAME GPU Visualiser");
	glutReshapeFunc(windowResize);

    // initialize GL
    if(FALSE == initGL()) {
        return;
    }

	//load pedestrians
	initPedestrianPopulation();

	//initialise input control
	initInputConroller();

	//init FLAME GPU globals controller
	initGlobalsController();

	//init menu
	initMenuItems();


	//FPS
	start_time = 0;
	end_time = 0;
	frame_time = 0;
	fps = 0;
	frames = 0;
	av_frames = 25;

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
	glutSpecialFunc( specialKeyboard);
    glutMouseFunc( mouse);





}

extern void runVisualisation()
{
    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return FALSE;
    }

    // default initialization
    glClearColor( 1.0, 1.0, 1.0, 1.0);


    return TRUE;
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display(void)
{

	//start timing
	//glFinish();
	start_time = clock();

	// viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.001, 50.0);
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable( GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	//lookat
	gluLookAt(eye[0], eye[1], eye[2], look[0], look[1], look[2], up[0], up[1], up[2]);

	//lighting
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

	updateSimulationConstants();
	
	stepFLAMESimulation();
	renderPedestrianPopulation();

	drawInfoDisplay(window_width, window_height);
	drawMenuDisplay(window_width, window_height);

	//end timing
	glFinish();
	end_time = clock();
	frame_time += (end_time - start_time);
	if (frames == av_frames){
		fps = (float)av_frames/(frame_time/(float)CLOCKS_PER_SEC);
		frames = 0;
		frame_time = 0.0f;
	}else{
		frames++;
	}

	//redraw
    glutSwapBuffers();
    glutPostRedisplay();

}

void updateSimulationConstants(){
	set_EYE_X(&eye[0]);
	set_EYE_Y(&eye[1]);
	set_EYE_Z(&eye[2]);
}

void windowResize(int width, int height){
	window_width = width;
	window_height = height;
}


void toggleFullScreenMode()
{
	fullScreenMode = !fullScreenMode;
	glutFullScreen();
}

float getFPS()
{
	return fps;
}


void checkGLError(){
	int Error;
    if((Error = glGetError()) != GL_NO_ERROR)
    {
	    const char* Message = (const char*)gluErrorString(Error);
        fprintf(stderr, "OpenGL Error : %s\n", Message);
    }
}
