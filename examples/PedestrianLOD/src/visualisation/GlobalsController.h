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
#ifndef __GLOBALS_CONTROLLER
#define __GLOBALS_CONTROLLER


#define INITIAL_TIME_SCALER		0.0003f
#define TIME_SCALER_INCREMENT	0.00001f

#define INITIAL_STEER_WEIGHT		0.10f
#define INITIAL_AVOID_WEIGHT		0.02f
#define INITIAL_COLLISION_WEIGHT	0.50f
#define INITIAL_GOAL_WEIGHT			0.20f

#define STEER_WEIGHT_INCREMENT		0.001f
#define AVOID_WEIGHT_INCREMENT		0.001f
#define COLLISION_WEIGHT_INCREMENT	0.001f
#define GOAL_WEIGHT_INCREMENT		0.001f


void initGlobalsController();

//time
void increaseTimeScaler();
void decreaseTimeScaler();
float getTimeScaler();
void setTimeScalerText(char* text);

//rule weights
void increaseSteerWeight();
void decreaseSteerWeight();
float getSteerWeight();
void setSteerWeightText(char* text);
void increaseAvoidWeight();
void decreaseAvoidWeight();
float getAvoidWeight();
void setAvoidWeightText(char* text);
void increaseCollisionWeight();
void decreaseCollisionWeight();
float getCollisionWeight();
void setCollisionWeightText(char* text);
void increaseGoalWeight();
void decreaseGoalWeight();
float getGoalWeight();
void setGoalWeightText(char* text);

#endif //__GLOBALS_CONTROLLER
