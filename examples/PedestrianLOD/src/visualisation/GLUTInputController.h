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
#ifndef __GLUT_INPUT_CONTROLLER
#define __GLUT_INPUT_CONTROLLER

/** initInputConroller
 * Initilises the input controller by setting the initial viewpoint vectors
 */
void initInputConroller();

/** mouse
 * Function for controlling mouse input with GLUT
 * @param	button	muse button state
 * @param	state mouse state (i.e. up down)
 * @param	x x screen position 
 * @param	y y screen position 
 */
void mouse(int button, int state, int x, int y);

/** keyboard
 * Function for controlling keyboard input with GLUT
 * @param	key	key pressed
 * @param	x 
 * @param	y 
 */
void keyboard( unsigned char key, int x, int y);

/** specialKeyboard
 * Function for controlling special keyboard (arrow keys) input with GLUT
 * @param	key	key pressed
 * @param	x 
 * @param	y 
 */
void specialKeyboard(int key, int x, int y);

//viewpoint vectors and eye distance
extern float eye[3];
extern float up[3];
extern float look[3];
extern float eye_distance;

#endif //__GLUT_INPUT_CONTROLLER
