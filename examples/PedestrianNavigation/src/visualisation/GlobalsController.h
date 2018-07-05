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

#define EMISSION_RATE_INCREMENT 0.01f
#define TIME_SCALER_INCREMENT	0.00001
#define STEER_WEIGHT_INCREMENT		0.001
#define AVOID_WEIGHT_INCREMENT		0.001
#define COLLISION_WEIGHT_INCREMENT	0.001
#define GOAL_WEIGHT_INCREMENT		0.001

void initGlobalsController();

void increaseGlobalEmmisionRate();
void decreaseGlobalEmmisionRate();

//emmision rates
void increaseEmmisionRateExit1();
void decreaseEmmisionRateExit1();
float getEmmisionRateExit1();
void setEmmisionRateExit1Text(char* text);
void increaseEmmisionRateExit2();
void decreaseEmmisionRateExit2();
float getEmmisionRateExit2();
void setEmmisionRateExit2Text(char* text);
void increaseEmmisionRateExit3();
void decreaseEmmisionRateExit3();
float getEmmisionRateExit3();
void setEmmisionRateExit3Text(char* text);
void increaseEmmisionRateExit4();
void decreaseEmmisionRateExit4();
float getEmmisionRateExit4();
void setEmmisionRateExit4Text(char* text);
void increaseEmmisionRateExit5();
void decreaseEmmisionRateExit5();
float getEmmisionRateExit5();
void setEmmisionRateExit5Text(char* text);
void increaseEmmisionRateExit6();
void decreaseEmmisionRateExit6();
float getEmmisionRateExit6();
void setEmmisionRateExit6Text(char* text);
void increaseEmmisionRateExit7();
void decreaseEmmisionRateExit7();
float getEmmisionRateExit7();
void setEmmisionRateExit7Text(char* text);

//exit probabilities
void increaseProbabilityExit1();
void decreaseProbabilityExit1();
float getProbabilityExit1();
void setProbabilityExit1Text(char* text);
void increaseProbabilityExit2();
void decreaseProbabilityExit2();
float getProbabilityExit2();
void setProbabilityExit2Text(char* text);
void increaseProbabilityExit3();
void decreaseProbabilityExit3();
float getProbabilityExit3();
void setProbabilityExit3Text(char* text);
void increaseProbabilityExit4();
void decreaseProbabilityExit4();
float getProbabilityExit4();
void setProbabilityExit4Text(char* text);
void increaseProbabilityExit5();
void decreaseProbabilityExit5();
float getProbabilityExit5();
void setProbabilityExit5Text(char* text);
void increaseProbabilityExit6();
void decreaseProbabilityExit6();
float getProbabilityExit6();
void setProbabilityExit6Text(char* text);
void increaseProbabilityExit7();
void decreaseProbabilityExit7();
float getProbabilityExit7();
void setProbabilityExit7Text(char* text);

//exit states
void toggleStateExit1();
int getStateExit1();
void setStateExit1Text(char* text);
void toggleStateExit2();
int getStateExit2();
void setStateExit2Text(char* text);
void toggleStateExit3();
int getStateExit3();
void setStateExit3Text(char* text);
void toggleStateExit4();
int getStateExit4();
void setStateExit4Text(char* text);
void toggleStateExit5();
int getStateExit5();
void setStateExit5Text(char* text);
void toggleStateExit6();
int getStateExit6();
void setStateExit6Text(char* text);
void toggleStateExit7();
int getStateExit7();
void setStateExit7Text(char* text);

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
