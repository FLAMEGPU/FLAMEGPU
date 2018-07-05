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
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>

#include "GlobalsController.h"
#include "CustomVisualisation.h"

#ifdef _MSC_VER
// Disable _CRT_SECURE_NO_WARNINGS warnings
#pragma warning(disable:4996)
#endif

//globals, initialised to 0 then loaded from the relevant variable as specified in the model or initial states file (or init function)
float emmisionRateExit1 = 0;
float emmisionRateExit2 = 0;
float emmisionRateExit3 = 0;
float emmisionRateExit4 = 0;
float emmisionRateExit5 = 0;
float emmisionRateExit6 = 0;
float emmisionRateExit7 = 0;

int exitProbability1 = 0;
int exitProbability2 = 0;
int exitProbability3 = 0;
int exitProbability4 = 0;
int exitProbability5 = 0;
int exitProbability6 = 0;
int exitProbability7 = 0;

int exitState1 = 0;
int exitState2 = 0;
int exitState3 = 0;
int exitState4 = 0;
int exitState5 = 0;
int exitState6 = 0;
int exitState7 = 0;

float timeScaler = 0;

float steerWeight = 0;
float avoidWeight = 0;
float collisionWeight = 0;
float goalWeight = 0;

//imported functions from FLAME GPU
extern void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE);
extern void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE);
extern void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE);
extern void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE);
extern void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE);
extern void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE);
extern void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE);

extern void set_EXIT1_PROBABILITY(int* h_PROBABILITY);
extern void set_EXIT2_PROBABILITY(int* h_PROBABILITY);
extern void set_EXIT3_PROBABILITY(int* h_PROBABILITY);
extern void set_EXIT4_PROBABILITY(int* h_PROBABILITY);
extern void set_EXIT5_PROBABILITY(int* h_PROBABILITY);
extern void set_EXIT6_PROBABILITY(int* h_PROBABILITY);
extern void set_EXIT7_PROBABILITY(int* h_PROBABILITY);

extern void set_EXIT1_STATE(int* h_STATE);
extern void set_EXIT2_STATE(int* h_STATE);
extern void set_EXIT3_STATE(int* h_STATE);
extern void set_EXIT4_STATE(int* h_STATE);
extern void set_EXIT5_STATE(int* h_STATE);
extern void set_EXIT6_STATE(int* h_STATE);
extern void set_EXIT7_STATE(int* h_STATE);

extern void set_TIME_SCALER(float* h_EMMISION_RATE);

extern void set_STEER_WEIGHT(float* h_weight);
extern void set_AVOID_WEIGHT(float* h_weight);
extern void set_COLLISION_WEIGHT(float* h_weight);
extern void set_GOAL_WEIGHT(float* h_weight);

extern const float * get_EMMISION_RATE_EXIT1();
extern const float * get_EMMISION_RATE_EXIT2();
extern const float * get_EMMISION_RATE_EXIT3();
extern const float * get_EMMISION_RATE_EXIT4();
extern const float * get_EMMISION_RATE_EXIT5();
extern const float * get_EMMISION_RATE_EXIT6();
extern const float * get_EMMISION_RATE_EXIT7();

extern const int * get_EXIT1_PROBABILITY();
extern const int * get_EXIT2_PROBABILITY();
extern const int * get_EXIT3_PROBABILITY();
extern const int * get_EXIT4_PROBABILITY();
extern const int * get_EXIT5_PROBABILITY();
extern const int * get_EXIT6_PROBABILITY();
extern const int * get_EXIT7_PROBABILITY();

extern const int * get_EXIT1_STATE();
extern const int * get_EXIT2_STATE();
extern const int * get_EXIT3_STATE();
extern const int * get_EXIT4_STATE();
extern const int * get_EXIT5_STATE();
extern const int * get_EXIT6_STATE();
extern const int * get_EXIT7_STATE();

extern const int * get_EXIT1_CELL_COUNT();
extern const int * get_EXIT2_CELL_COUNT();
extern const int * get_EXIT3_CELL_COUNT();
extern const int * get_EXIT4_CELL_COUNT();
extern const int * get_EXIT5_CELL_COUNT();
extern const int * get_EXIT6_CELL_COUNT();
extern const int * get_EXIT7_CELL_COUNT();

extern const float * get_TIME_SCALER();

extern const float * get_STEER_WEIGHT();
extern const float * get_AVOID_WEIGHT();
extern const float * get_COLLISION_WEIGHT();
extern const float * get_GOAL_WEIGHT();

//private prototypes
float getExitProbabilityCounts();


void initGlobalsController()
{
	// Fetch each value 
	timeScaler = *get_TIME_SCALER();
	steerWeight = *get_STEER_WEIGHT();
	avoidWeight = *get_AVOID_WEIGHT();
	collisionWeight = *get_COLLISION_WEIGHT();
	goalWeight = *get_GOAL_WEIGHT();

	emmisionRateExit1 = *get_EMMISION_RATE_EXIT1();
	emmisionRateExit2 = *get_EMMISION_RATE_EXIT2();
	emmisionRateExit3 = *get_EMMISION_RATE_EXIT3();
	emmisionRateExit4 = *get_EMMISION_RATE_EXIT4();
	emmisionRateExit5 = *get_EMMISION_RATE_EXIT5();
	emmisionRateExit6 = *get_EMMISION_RATE_EXIT6();
	emmisionRateExit7 = *get_EMMISION_RATE_EXIT7();

	exitProbability1 = *get_EXIT1_PROBABILITY();
	exitProbability2 = *get_EXIT2_PROBABILITY();
	exitProbability3 = *get_EXIT3_PROBABILITY();
	exitProbability4 = *get_EXIT4_PROBABILITY();
	exitProbability5 = *get_EXIT5_PROBABILITY();
	exitProbability6 = *get_EXIT6_PROBABILITY();
	exitProbability7 = *get_EXIT7_PROBABILITY();

	exitState1 = *get_EXIT1_STATE();
	exitState2 = *get_EXIT2_STATE();
	exitState3 = *get_EXIT3_STATE();
	exitState4 = *get_EXIT4_STATE();
	exitState5 = *get_EXIT5_STATE();
	exitState6 = *get_EXIT6_STATE();
	exitState7 = *get_EXIT7_STATE();
}

//global emmision rate

void increaseGlobalEmmisionRate()
{
	increaseEmmisionRateExit1();
	increaseEmmisionRateExit2();
	increaseEmmisionRateExit3();
	increaseEmmisionRateExit4();
	increaseEmmisionRateExit5();
	increaseEmmisionRateExit6();
	increaseEmmisionRateExit7();
}

void decreaseGlobalEmmisionRate()
{
	decreaseEmmisionRateExit1();
	decreaseEmmisionRateExit2();
	decreaseEmmisionRateExit3();
	decreaseEmmisionRateExit4();
	decreaseEmmisionRateExit5();
	decreaseEmmisionRateExit6();
	decreaseEmmisionRateExit7();
}

/* EMMISION RATES */


//emmision rate exit 1
void increaseEmmisionRateExit1()
{
	emmisionRateExit1 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT1(&emmisionRateExit1);
}
void decreaseEmmisionRateExit1()
{
	emmisionRateExit1 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT1(&emmisionRateExit1);
}
float getEmmisionRateExit1(){	return emmisionRateExit1;}
void setEmmisionRateExit1Text(char* text)
{	
	float rate_pm = emmisionRateExit1 * (*get_EXIT1_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 1: %f", rate_pm);
}

//emmision rate exit 2
void increaseEmmisionRateExit2()
{
	emmisionRateExit2 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT2(&emmisionRateExit2);
}
void decreaseEmmisionRateExit2()
{
	emmisionRateExit2 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT2(&emmisionRateExit2);
}
float getEmmisionRateExit2(){	return emmisionRateExit2;}
void setEmmisionRateExit2Text(char* text)
{	
	float rate_pm = emmisionRateExit2 * (*get_EXIT2_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 2: %f", rate_pm);
}

//emmision rate exit 3
void increaseEmmisionRateExit3()
{
	emmisionRateExit3 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT3(&emmisionRateExit3);
}
void decreaseEmmisionRateExit3()
{
	emmisionRateExit3 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT3(&emmisionRateExit3);
}
float getEmmisionRateExit3(){	return emmisionRateExit3;}
void setEmmisionRateExit3Text(char* text)
{	
	float rate_pm = emmisionRateExit3 * (*get_EXIT3_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 3: %f", rate_pm);
}

//emmision rate exit 4
void increaseEmmisionRateExit4()
{
	emmisionRateExit4 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT4(&emmisionRateExit4);
}
void decreaseEmmisionRateExit4()
{
	emmisionRateExit4 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT4(&emmisionRateExit4);
}
float getEmmisionRateExit4(){	return emmisionRateExit4;}
void setEmmisionRateExit4Text(char* text)
{	
	float rate_pm = emmisionRateExit4 * (*get_EXIT4_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 4: %f", rate_pm);
}

//emmision rate exit 5
void increaseEmmisionRateExit5()
{
	emmisionRateExit5 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT5(&emmisionRateExit5);
}
void decreaseEmmisionRateExit5()
{
	emmisionRateExit5 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT5(&emmisionRateExit5);
}
float getEmmisionRateExit5(){	return emmisionRateExit5;}
void setEmmisionRateExit5Text(char* text)
{	
	float rate_pm = emmisionRateExit5 * (*get_EXIT5_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 5: %f", rate_pm);
}

//emmision rate exit 6
void increaseEmmisionRateExit6()
{
	emmisionRateExit6 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT6(&emmisionRateExit6);
}
void decreaseEmmisionRateExit6()
{
	emmisionRateExit6 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT6(&emmisionRateExit6);
}
float getEmmisionRateExit6(){	return emmisionRateExit6;}
void setEmmisionRateExit6Text(char* text)
{	
	float rate_pm = emmisionRateExit6 * (*get_EXIT6_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 6: %f", rate_pm);
}

//emmision rate exit 7
void increaseEmmisionRateExit7()
{
	emmisionRateExit7 += EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT7(&emmisionRateExit7);
}
void decreaseEmmisionRateExit7()
{
	emmisionRateExit7 -= EMISSION_RATE_INCREMENT;
	set_EMMISION_RATE_EXIT7(&emmisionRateExit7);
}
float getEmmisionRateExit7(){	return emmisionRateExit7;}
void setEmmisionRateExit7Text(char* text)
{	
	float rate_pm = emmisionRateExit7 * (*get_EXIT7_CELL_COUNT()) * getFPS() * 60.0f * timeScaler;
	sprintf(text, "Emmision Rate Exit 7: %f", rate_pm);
}



/* PROBABILITY RATES */

//exit 1 prob
void increaseProbabilityExit1(){
	exitProbability1 += 1;
	set_EXIT1_PROBABILITY(&exitProbability1);
}
void decreaseProbabilityExit1(){
	exitProbability1 -= 1;
	if (exitProbability1<1)
		exitProbability1 = 0;
	set_EXIT1_PROBABILITY(&exitProbability1);
}
float getProbabilityExit1() {
	return (float)exitProbability1/getExitProbabilityCounts();
}
void setProbabilityExit1Text(char* text) { 
	sprintf(text, "Exit Probability 1: %f", getProbabilityExit1()); 
}

//exit 2 prob
void increaseProbabilityExit2(){
	exitProbability2 += 1;
	set_EXIT2_PROBABILITY(&exitProbability2);
}
void decreaseProbabilityExit2(){
	exitProbability2 -= 1;
	if (exitProbability2<1)
		exitProbability2 = 0;
	set_EXIT2_PROBABILITY(&exitProbability2);
}
float getProbabilityExit2() {
	return (float)exitProbability2/getExitProbabilityCounts();
}
void setProbabilityExit2Text(char* text) { 
	sprintf(text, "Exit Probability 2: %f", getProbabilityExit2()); 
}

//exit 3 prob
void increaseProbabilityExit3(){
	exitProbability3 += 1;
	set_EXIT3_PROBABILITY(&exitProbability3);
}
void decreaseProbabilityExit3(){
	exitProbability3 -= 1;
	if (exitProbability3<1)
		exitProbability3 = 0;
	set_EXIT3_PROBABILITY(&exitProbability3);
}
float getProbabilityExit3() {
	return (float)exitProbability3/getExitProbabilityCounts();
}
void setProbabilityExit3Text(char* text) { 
	sprintf(text, "Exit Probability 3: %f", getProbabilityExit3()); 
}

//exit 4 prob
void increaseProbabilityExit4(){
	exitProbability4 += 1;
	set_EXIT4_PROBABILITY(&exitProbability4);
}
void decreaseProbabilityExit4(){
	exitProbability4 -= 1;
	if (exitProbability4<1)
		exitProbability4 = 0;
	set_EXIT4_PROBABILITY(&exitProbability4);
}
float getProbabilityExit4() {
	return (float)exitProbability4/getExitProbabilityCounts();
}
void setProbabilityExit4Text(char* text) { 
	sprintf(text, "Exit Probability 4: %f", getProbabilityExit4()); 
}

//exit 5 prob
void increaseProbabilityExit5(){
	exitProbability5 += 1;
	set_EXIT5_PROBABILITY(&exitProbability5);
}
void decreaseProbabilityExit5(){
	exitProbability5 -= 1;
	if (exitProbability5<1)
		exitProbability5 = 0;
	set_EXIT5_PROBABILITY(&exitProbability5);
}
float getProbabilityExit5() {
	return (float)exitProbability5/getExitProbabilityCounts();
}
void setProbabilityExit5Text(char* text) { 
	sprintf(text, "Exit Probability 5: %f", getProbabilityExit5()); 
}

//exit 6 prob
void increaseProbabilityExit6(){
	exitProbability6 += 1;
	set_EXIT6_PROBABILITY(&exitProbability6);
}
void decreaseProbabilityExit6(){
	exitProbability6 -= 1;
	if (exitProbability6<1)
		exitProbability6 = 0;
	set_EXIT6_PROBABILITY(&exitProbability6);
}
float getProbabilityExit6() {
	return (float)exitProbability6/getExitProbabilityCounts();
}
void setProbabilityExit6Text(char* text) { 
	sprintf(text, "Exit Probability 6: %f", getProbabilityExit6()); 
}

//exit 7 prob
void increaseProbabilityExit7(){
	exitProbability7 += 1;
	set_EXIT7_PROBABILITY(&exitProbability7);
}
void decreaseProbabilityExit7(){
	exitProbability7 -= 1;
	if (exitProbability7<1)
		exitProbability7 = 0;
	set_EXIT7_PROBABILITY(&exitProbability7);
}
float getProbabilityExit7() {
	return (float)exitProbability7/getExitProbabilityCounts();
}
void setProbabilityExit7Text(char* text) { 
	sprintf(text, "Exit Probability 7: %f", getProbabilityExit7()); 
}


/* exit states */
//exit 1
void toggleStateExit1()
{
	exitState1 = !exitState1;
	set_EXIT1_STATE(&exitState1);
}
int getStateExit1()
{
	return exitState1;
}
void setStateExit1Text(char* text)
{
	if (exitState1)
		sprintf(text, "Exit 1 State: OPEN");
	else
		sprintf(text, "Exit 1 State: CLOSED");
}

//exit 2
void toggleStateExit2()
{
	exitState2 = !exitState2;
	set_EXIT2_STATE(&exitState2);
}
int getStateExit2()
{
	return exitState2;
}
void setStateExit2Text(char* text)
{
	if (exitState2)
		sprintf(text, "Exit 2 State: OPEN");
	else
		sprintf(text, "Exit 2 State: CLOSED");
}

//exit 3
void toggleStateExit3()
{
	exitState3 = !exitState3;
	set_EXIT3_STATE(&exitState3);
}
int getStateExit3()
{
	return exitState3;
}
void setStateExit3Text(char* text)
{
	if (exitState3)
		sprintf(text, "Exit 3 State: OPEN");
	else
		sprintf(text, "Exit 3 State: CLOSED");
}

//exit 4
void toggleStateExit4()
{
	exitState4 = !exitState4;
	set_EXIT4_STATE(&exitState4);
}
int getStateExit4()
{
	return exitState4;
}
void setStateExit4Text(char* text)
{
	if (exitState4)
		sprintf(text, "Exit 4 State: OPEN");
	else
		sprintf(text, "Exit 4 State: CLOSED");
}

//exit 5
void toggleStateExit5()
{
	exitState5 = !exitState5;
	set_EXIT5_STATE(&exitState5);
}
int getStateExit5()
{
	return exitState5;
}
void setStateExit5Text(char* text)
{
	if (exitState5)
		sprintf(text, "Exit 5 State: OPEN");
	else
		sprintf(text, "Exit 5 State: CLOSED");
}

//exit 6
void toggleStateExit6()
{
	exitState6 = !exitState6;
	set_EXIT6_STATE(&exitState6);
}
int getStateExit6()
{
	return exitState6;
}
void setStateExit6Text(char* text)
{
	if (exitState6)
		sprintf(text, "Exit 6 State: OPEN");
	else
		sprintf(text, "Exit 6 State: CLOSED");
}

//exit 7
void toggleStateExit7()
{
	exitState7 = !exitState7;
	set_EXIT7_STATE(&exitState7);
}
int getStateExit7()
{
	return exitState7;
}
void setStateExit7Text(char* text)
{
	if (exitState7)
		sprintf(text, "Exit 7 State: OPEN");
	else
		sprintf(text, "Exit 7 State: CLOSED");
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
	//prevent negative time scaler
	if (timeScaler < 0)
		timeScaler += TIME_SCALER_INCREMENT;
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

//privates
float getExitProbabilityCounts()
{
	return (float)exitProbability1 + exitProbability2 + exitProbability3 + exitProbability4 + exitProbability5 + exitProbability6 + exitProbability7; 
}
