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
#ifndef __VISUALISATION
#define __VISUALISATION

#ifndef FALSE
enum BOOLEAN {FALSE, TRUE};
typedef enum BOOLEAN BOOLEAN;
#endif


enum TOGGLE_STATE {TOGGLE_OFF, TOGGLE_ON};
typedef enum TOGGLE_STATE TOGGLE_STATE;


#define ENV_MAX 1.0f
#define ENV_MIN -ENV_MAX
#define ENV_WIDTH (2*ENV_MAX)

// prototypes
int initGL();
void display(void);
void windowResize(int width, int height);
void toggleFullScreenMode();
float getFPS();

//external functions
extern void stepFLAMESimulation();

//common functions
void checkGLError();


#endif //__VISUALISATION
