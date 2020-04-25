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

#ifndef __VISUALISATION_H
#define __VISUALISATION_H

// constants
/*const unsigned int WINDOW_WIDTH = 1200;
const unsigned int WINDOW_HEIGHT = 850;*/
const unsigned int WINDOW_WIDTH = 1500;
const unsigned int WINDOW_HEIGHT = 1000;

/*-supposedly- global variables*/
//extern int frame_count;
//static float frame_time



//frustrum
const double NEAR_CLIP = 0.1;
const double FAR_CLIP = 100;

//Circle model fidelity
const int SPHERE_SLICES = 20;
const int SPHERE_STACKS = 20;


const double SPHERE_RADIUS = 0.0150f; // best fit for (1,000) agents.
//const double SPHERE_RADIUS = 0.0070f; // best fit for (100,000) agents.
//const double SPHERE_RADIUS = 0.0035f; // best fit for (500,000) agents.


//const double SPHERE_RADIUS = 0.120f; // best fit for smaller population (10) agents.

//const double SPHERE_RADIUS = 0.020f;
const double VIEW_DISTANCE = 4.0;//Viewing Distance


//light position
GLfloat LIGHT_POSITION[] = {10.0f, 10.0f, 10.0f, 1.0f};

//#define SIMULATION_DELAY 0.5

#endif //__VISUALISATION_H
