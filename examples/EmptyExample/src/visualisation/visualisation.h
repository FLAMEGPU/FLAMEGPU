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

#define SIMULATION_DELAY 1

// constants
const unsigned int WINDOW_WIDTH = 1280;
const unsigned int WINDOW_HEIGHT = 720;

//frustrum
const double NEAR_CLIP = 0.1;
const double FAR_CLIP = 300;

//Circle model fidelity
const int SPHERE_SLICES = 8;
const int SPHERE_STACKS = 8;
const double SPHERE_RADIUS = 0.1;

//Viewing Distance
const double VIEW_DISTANCE = 16;

//light position
GLfloat LIGHT_POSITION[] = {10.0f, 10.0f, 10.0f, 1.0f};

#endif //__VISUALISATION_H
