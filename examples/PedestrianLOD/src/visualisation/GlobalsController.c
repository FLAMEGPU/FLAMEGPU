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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include "GlobalsController.h"
#include "CustomVisualisation.h"


float timeScaler = INITIAL_TIME_SCALER;

float steerWeight = INITIAL_STEER_WEIGHT;
float avoidWeight = INITIAL_AVOID_WEIGHT;
float collisionWeight = INITIAL_COLLISION_WEIGHT;
float goalWeight = INITIAL_GOAL_WEIGHT;

extern void set_TIME_SCALER(float* h_EMMISION_RATE);

extern void set_STEER_WEIGHT(float* h_weight);
extern void set_AVOID_WEIGHT(float* h_weight);
extern void set_COLLISION_WEIGHT(float* h_weight);
extern void set_GOAL_WEIGHT(float* h_weight);

//private prototypes
float getExitProbabilityCounts();


void initGlobalsController()
{
	set_TIME_SCALER(&timeScaler);
	set_STEER_WEIGHT(&steerWeight);
	set_AVOID_WEIGHT(&avoidWeight);
	set_COLLISION_WEIGHT(&collisionWeight);
	set_GOAL_WEIGHT(&goalWeight);
}


//time scaler
void increaseTimeScaler()
{
	timeScaler += TIME_SCALER_INCREMENT;
	set_TIME_SCALER(&timeScaler);
}

void decreaseTimeScaler()
{
	timeScaler -= TIME_SCALER_INCREMENT;
	set_TIME_SCALER(&timeScaler);
}

float getTimeScaler()
{
	return timeScaler;
}

void setTimeScalerText(char* text)
{
	sprintf(text, "Time Scaler: %f", timeScaler);
}

/* RULE WEIGHTS */
//steer
void increaseSteerWeight(){
	steerWeight += STEER_WEIGHT_INCREMENT;
	set_STEER_WEIGHT(&steerWeight);
}
void decreaseSteerWeight(){
	steerWeight -= STEER_WEIGHT_INCREMENT;
	if (steerWeight < 0)
		steerWeight = 0;
	set_STEER_WEIGHT(&steerWeight);
}
float getSteerWeight(){
	return steerWeight;
}
void setSteerWeightText(char* text){
	sprintf(text, "Steer Rule Weight: %f", steerWeight);
}
//avoid
void increaseAvoidWeight(){
	avoidWeight += AVOID_WEIGHT_INCREMENT;
	set_AVOID_WEIGHT(&avoidWeight);
}
void decreaseAvoidWeight(){
	avoidWeight -= AVOID_WEIGHT_INCREMENT;
	if (avoidWeight < 0)
		avoidWeight = 0;
	set_AVOID_WEIGHT(&avoidWeight);
}
float getAvoidWeight(){
	return avoidWeight;
}
void setAvoidWeightText(char* text){
	sprintf(text, "Avoid Rule Weight: %f", avoidWeight);
}
//collision
void increaseCollisionWeight(){
	collisionWeight += COLLISION_WEIGHT_INCREMENT;
	set_COLLISION_WEIGHT(&collisionWeight);
}
void decreaseCollisionWeight(){
	collisionWeight -= COLLISION_WEIGHT_INCREMENT;
	if (collisionWeight < 0)
		collisionWeight = 0;
	set_COLLISION_WEIGHT(&collisionWeight);
}
float getCollisionWeight(){
	return collisionWeight;
}
void setCollisionWeightText(char* text){
	sprintf(text, "Collision Rule Weight: %f", collisionWeight);
}
//goal
void increaseGoalWeight(){
	goalWeight += GOAL_WEIGHT_INCREMENT;
	set_GOAL_WEIGHT(&goalWeight);
}
void decreaseGoalWeight(){
	goalWeight -= GOAL_WEIGHT_INCREMENT;
	if (goalWeight < 0)
		goalWeight = 0;
	set_GOAL_WEIGHT(&goalWeight);
}
float getGoalWeight(){
	return goalWeight;
}
void setGoalWeightText(char* text){
	sprintf(text, "Goal Rule Weight: %f", goalWeight);
}
