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
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <math.h>
#include "CustomVisualisation.h"
#include "GLUTInputController.h"
#include "MenuDisplay.h"
#include "GlobalsController.h"
#include "MenuDisplay.h"


float theta;
float phi;
float cos_theta;
float sin_theta;
float cos_phi;
float sin_phi;

int mouse_old_x, mouse_old_y;

int zoom_key = 0;

#define TRANSLATION_SCALE 0.005f
#define ROTATION_SCALE 0.01f
#define ZOOM_SCALE 0.01f

#define MAX_ZOOM 0.01f
#define MIN_PHI 0.0f

#define PI 3.14
#define rad(x) (PI / 180) * x

//prototypes
void updateRotationComponents();
//mouse motion funcs
void rotate(int x, int y);
void zoom(int x, int y);
void translate(int x, int y);

void initInputConroller()
{
	//init view
	eye_distance = ENV_MAX*1.75f;
	up[0] = 0.0f;
	up[1] = 1.0f;
	up[2] = 0.0f;
	eye[0] = 0.0f;
	eye[1] = 0.0f;
	eye[2] = eye_distance;
	look[0] = 0.0f;
	look[1] = 0.0f;
	look[2] = 0.0f;

	theta = 3.14159265f;
	phi = 1.57079633f;
}


void mouse(int button, int state, int x, int y)
{
	if (zoom_key)
		button  = GLUT_MIDDLE_BUTTON;

    if (state == GLUT_DOWN) {
        switch(button)
		{
			case(GLUT_LEFT_BUTTON):
			{
				glutMotionFunc(translate);
				break;	
			}
			case(GLUT_RIGHT_BUTTON):
			{
				glutMotionFunc(rotate);
				break;	
			}
			case(GLUT_MIDDLE_BUTTON):
			{
				glutMotionFunc(zoom);
				break;	
			}
		}
    } else if (state == GLUT_UP) {
		glutMotionFunc(NULL);
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void updateRotationComponents()
{
	cos_theta = (float) cos(theta);
	sin_theta = (float) sin(theta);
	cos_phi = (float) cos(phi);
	sin_phi = (float) sin(phi);
}

void rotate(int x, int y)
{
	float dx, dy;
	//calc change in mouse movement
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	//update rotation component values
	updateRotationComponents();

	//update eye distance
	theta-=dx*ROTATION_SCALE;
	phi+=dy*ROTATION_SCALE;

	phi = (phi<MIN_PHI)?0.0f:phi;

	//update eye and and up vectors
	eye[0]= look[0] + -eye_distance*sin_theta*cos_phi;
	eye[1]= look[1] + eye_distance*cos_theta*cos_phi;
	eye[2]= look[2] + eye_distance*sin_phi;
	up[0]= sin_theta*sin_phi;
	up[1]= -cos_theta*sin_phi;
	up[2]= cos_phi;
	//update prev positions
	mouse_old_x = x;
	mouse_old_y = y;
}

void zoom(int x, int y)
{
	float dx, dy;
	//calc change in mouse movement
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	//update rotation component values
	updateRotationComponents();

	//update eye distance
	eye_distance -= dy*ZOOM_SCALE;
	eye_distance = (eye_distance<MAX_ZOOM)?MAX_ZOOM:eye_distance;

	//update eye vector
	eye[0]= look[0] + -eye_distance*sin_theta*cos_phi;
	eye[1]= look[1] + eye_distance*cos_theta*cos_phi;
	eye[2]= look[2] + eye_distance*sin_phi;

	//update prev positions
	mouse_old_x = x;
	mouse_old_y = y;
}

void translate(int x, int y)
{
	float dx, dy;
	//calc change in mouse movement
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	//update rotation component values
	updateRotationComponents();

	//translate look and eye vector position
	look[0] += ((dx*cos_theta) + (dy*sin_theta))*TRANSLATION_SCALE;
	look[1] += ((dx*sin_theta) - (dy*cos_theta))*TRANSLATION_SCALE;
	look[2] += 0.0;
	eye[0]= look[0] + -eye_distance*sin_theta*cos_phi;
	eye[1]= look[1] + eye_distance*cos_theta*cos_phi;
	eye[2]= look[2] + eye_distance*sin_phi;


	//update prev positions
	mouse_old_x = x;
	mouse_old_y = y;
}

void keyboard( unsigned char key, int x, int y)
{
    switch( key) {
		
		case('f'):
		{
			toggleFullScreenMode();
			break;
		}
		case('i'):
		{
			setMenuDisplayOnOff(0);
			toggleInformationDisplayOnOff();
			break;
		}
		case('m'):
		{
			setInformationDisplayOnOff(0);
			toggleMenuDisplayOnOff();
			break;
		}
		case('z'):
		{
			zoom_key = !zoom_key;
			break;
		}
		

		//exit
		case('q') :
		{
			exit(0);
			break;
		}

		default:
		{
			break;
		}
    }
}

void specialKeyboard(int key, int x, int y)
{
	if (menuDisplayed())
	{
		switch(key) {
			case(GLUT_KEY_DOWN):
			{
				handleDownKey();
				break;
			}
			case(GLUT_KEY_UP):
			{
				handleUpKey();
				break;
			}
			case(GLUT_KEY_LEFT):
			{
				handleLeftKey();
				break;
			}
			case(GLUT_KEY_RIGHT):
			{
				handleRightKey();
				break;
			}
			default:
			{
				break;
			}
		}
	}
}