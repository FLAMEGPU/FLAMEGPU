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

#include "MenuDisplay.h"
#include "GlobalsController.h"
#include "CustomVisualisation.h"

#ifdef _MSC_VER
// Disable _CRT_SECURE_NO_WARNINGS warnings
#pragma warning(disable:4996)
#endif 

//holder for window width and height
int menu_width;
int menu_height;

int drawInfoDisplayState = 0;
int drawMenuDisplayState = 0;

//menu items

//info diaplay textoutput
int text_position = 0;

//external FLAME GPU functions from header.h
extern int get_agent_agent_MAX_count();
extern int get_agent_agent_default_count();
extern int get_agent_navmap_static_count();


//private prototypes
void updateAllProbabilityTexts(char* text);
void set2DProjection();
void printInfoLine(char *string);
void printMenuItem(menu_item* menu_item);


//menu items
menu_item* menu;
menu_item *em_rate1, *em_rate2, *em_rate3, *em_rate4, *em_rate5, *em_rate6, *em_rate7;
menu_item *probability_1, *probability_2, *probability_3, *probability_4, *probability_5, *probability_6, *probability_7;
menu_item *time, *em_rate, *exit_state_1, *exit_state_2, *exit_state_3, *exit_state_4, *exit_state_5, *exit_state_6, *exit_state_7, *steer, *avoid, *collision, *goal;

void initMenuItems()
{

	time = (menu_item*)malloc(sizeof(menu_item));
	time->selected = 1;
	time->increase = increaseTimeScaler;
	time->decrease = decreaseTimeScaler;
	time->updateText = setTimeScalerText;
	time->updateText(time->text);

	steer = (menu_item*)malloc(sizeof(menu_item));
	steer->selected = 0;
	steer->increase = increaseSteerWeight;
	steer->decrease = decreaseSteerWeight;
	steer->updateText = setSteerWeightText;
	steer->updateText(steer->text);

	avoid = (menu_item*)malloc(sizeof(menu_item));
	avoid->selected = 0;
	avoid->increase = increaseAvoidWeight;
	avoid->decrease = decreaseAvoidWeight;
	avoid->updateText = setAvoidWeightText;
	avoid->updateText(avoid->text);

	collision = (menu_item*)malloc(sizeof(menu_item));
	collision->selected = 0;
	collision->increase = increaseCollisionWeight;
	collision->decrease = decreaseCollisionWeight;
	collision->updateText = setCollisionWeightText;
	collision->updateText(collision->text);

	goal = (menu_item*)malloc(sizeof(menu_item));
	goal->selected = 0;
	goal->increase = increaseGoalWeight;
	goal->decrease = decreaseGoalWeight;
	goal->updateText = setGoalWeightText;
	goal->updateText(goal->text);

	em_rate = (menu_item*)malloc(sizeof(menu_item));
	em_rate->selected = 0;
	em_rate->increase = increaseGlobalEmmisionRate;
	em_rate->decrease = decreaseGlobalEmmisionRate;
	em_rate->updateText = updateAllEmmsionRatesTexts;
	sprintf(em_rate->text, "GLOBAL EMMISION RATE");

	em_rate1 = (menu_item*)malloc(sizeof(menu_item));
	em_rate1->selected = 0;
	em_rate1->increase = increaseEmmisionRateExit1;
	em_rate1->decrease = decreaseEmmisionRateExit1;
	em_rate1->updateText = setEmmisionRateExit1Text;
	em_rate1->updateText(em_rate1->text);

	em_rate2 = (menu_item*)malloc(sizeof(menu_item));
	em_rate2->selected = 0;
	em_rate2->increase = increaseEmmisionRateExit2;
	em_rate2->decrease = decreaseEmmisionRateExit2;
	em_rate2->updateText = setEmmisionRateExit2Text;
	em_rate2->updateText(em_rate2->text);

	em_rate3 = (menu_item*)malloc(sizeof(menu_item));
	em_rate3->selected = 0;
	em_rate3->increase = increaseEmmisionRateExit3;
	em_rate3->decrease = decreaseEmmisionRateExit3;
	em_rate3->updateText = setEmmisionRateExit3Text;
	em_rate3->updateText(em_rate3->text);

	em_rate4 = (menu_item*)malloc(sizeof(menu_item));
	em_rate4->selected = 0;
	em_rate4->increase = increaseEmmisionRateExit4;
	em_rate4->decrease = decreaseEmmisionRateExit4;
	em_rate4->updateText = setEmmisionRateExit4Text;
	em_rate4->updateText(em_rate4->text);

	em_rate5 = (menu_item*)malloc(sizeof(menu_item));
	em_rate5->selected = 0;
	em_rate5->increase = increaseEmmisionRateExit5;
	em_rate5->decrease = decreaseEmmisionRateExit5;
	em_rate5->updateText = setEmmisionRateExit5Text;
	em_rate5->updateText(em_rate5->text);

	em_rate6 = (menu_item*)malloc(sizeof(menu_item));
	em_rate6->selected = 0;
	em_rate6->increase = increaseEmmisionRateExit6;
	em_rate6->decrease = decreaseEmmisionRateExit6;
	em_rate6->updateText = setEmmisionRateExit6Text;
	em_rate6->updateText(em_rate6->text);

	em_rate7 = (menu_item*)malloc(sizeof(menu_item));
	em_rate7->selected = 0;
	em_rate7->increase = increaseEmmisionRateExit7;
	em_rate7->decrease = decreaseEmmisionRateExit7;
	em_rate7->updateText = setEmmisionRateExit7Text;
	em_rate7->updateText(em_rate7->text);

	probability_1 = (menu_item*)malloc(sizeof(menu_item));
	probability_1->selected = 0;
	probability_1->increase = increaseProbabilityExit1;
	probability_1->decrease = decreaseProbabilityExit1;
	probability_1->updateText = updateAllProbabilityTexts;

	probability_2 = (menu_item*)malloc(sizeof(menu_item));
	probability_2->selected = 0;
	probability_2->increase = increaseProbabilityExit2;
	probability_2->decrease = decreaseProbabilityExit2;
	probability_2->updateText = updateAllProbabilityTexts;
	
	probability_3 = (menu_item*)malloc(sizeof(menu_item));
	probability_3->selected = 0;
	probability_3->increase = increaseProbabilityExit3;
	probability_3->decrease = decreaseProbabilityExit3;
	probability_3->updateText = updateAllProbabilityTexts;

	probability_4 = (menu_item*)malloc(sizeof(menu_item));
	probability_4->selected = 0;
	probability_4->increase = increaseProbabilityExit4;
	probability_4->decrease = decreaseProbabilityExit4;
	probability_4->updateText = updateAllProbabilityTexts;

	probability_5 = (menu_item*)malloc(sizeof(menu_item));
	probability_5->selected = 0;
	probability_5->increase = increaseProbabilityExit5;
	probability_5->decrease = decreaseProbabilityExit5;
	probability_5->updateText = updateAllProbabilityTexts;

	probability_6 = (menu_item*)malloc(sizeof(menu_item));
	probability_6->selected = 0;
	probability_6->increase = increaseProbabilityExit6;
	probability_6->decrease = decreaseProbabilityExit6;
	probability_6->updateText = updateAllProbabilityTexts;

	probability_7 = (menu_item*)malloc(sizeof(menu_item));
	probability_7->selected = 0;
	probability_7->increase = increaseProbabilityExit7;
	probability_7->decrease = decreaseProbabilityExit7;
	probability_7->updateText = updateAllProbabilityTexts;

	updateAllProbabilityTexts("");

	exit_state_1 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_1->selected = 0;
	exit_state_1->increase = toggleStateExit1;
	exit_state_1->decrease = toggleStateExit1;
	exit_state_1->updateText = setStateExit1Text;
	exit_state_1->updateText(exit_state_1->text);

	exit_state_2 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_2->selected = 0;
	exit_state_2->increase = toggleStateExit2;
	exit_state_2->decrease = toggleStateExit2;
	exit_state_2->updateText = setStateExit2Text;
	exit_state_2->updateText(exit_state_2->text);

	exit_state_3 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_3->selected = 0;
	exit_state_3->increase = toggleStateExit3;
	exit_state_3->decrease = toggleStateExit3;
	exit_state_3->updateText = setStateExit3Text;
	exit_state_3->updateText(exit_state_3->text);

	exit_state_4 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_4->selected = 0;
	exit_state_4->increase = toggleStateExit4;
	exit_state_4->decrease = toggleStateExit4;
	exit_state_4->updateText = setStateExit4Text;
	exit_state_4->updateText(exit_state_4->text);

	exit_state_5 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_5->selected = 0;
	exit_state_5->increase = toggleStateExit5;
	exit_state_5->decrease = toggleStateExit5;
	exit_state_5->updateText = setStateExit5Text;
	exit_state_5->updateText(exit_state_5->text);

	exit_state_6 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_6->selected = 0;
	exit_state_6->increase = toggleStateExit6;
	exit_state_6->decrease = toggleStateExit6;
	exit_state_6->updateText = setStateExit6Text;
	exit_state_6->updateText(exit_state_6->text);

	exit_state_7 = (menu_item*)malloc(sizeof(menu_item));
	exit_state_7->selected = 0;
	exit_state_7->increase = toggleStateExit7;
	exit_state_7->decrease = toggleStateExit7;
	exit_state_7->updateText = setStateExit7Text;
	exit_state_7->updateText(exit_state_7->text);


	//build linked list
	time->previous = exit_state_7;

	time->next = steer;

	steer->previous = time;
	steer->next = avoid;
	avoid->previous = steer;
	avoid->next = collision;
	collision->previous = avoid;
	collision->next = goal;
	goal->previous = collision;
	goal->next = em_rate;

	em_rate->previous = goal;
	em_rate->next = em_rate1;
	em_rate1->previous = em_rate;
	em_rate1->next = em_rate2;
	em_rate2->previous = em_rate1;
	em_rate2->next = em_rate3;
	em_rate3->previous = em_rate2;
	em_rate3->next = em_rate4;
	em_rate4->previous = em_rate3;
	em_rate4->next = em_rate5;
	em_rate5->previous = em_rate4;
	em_rate5->next = em_rate6;
	em_rate6->previous = em_rate5;
	em_rate6->next = em_rate7;
	em_rate7->previous = em_rate6;
	em_rate7->next = probability_1;
	probability_1->previous = em_rate7;
	probability_1->next = probability_2;
	probability_2->previous = probability_1;
	probability_2->next = probability_3;
	probability_3->previous = probability_2;
	probability_3->next = probability_4;
	probability_4->previous = probability_3;
	probability_4->next = probability_5;
	probability_5->previous = probability_4;
	probability_5->next = probability_6;
	probability_6->previous = probability_5;
	probability_6->next = probability_7;
	probability_7->previous = probability_6;
	probability_7->next = exit_state_1;
	exit_state_1->previous = probability_7;
	exit_state_1->next = exit_state_2;
	exit_state_2->previous = exit_state_1;
	exit_state_2->next = exit_state_3;
	exit_state_3->previous = exit_state_2;
	exit_state_3->next = exit_state_4;
	exit_state_4->previous = exit_state_3;
	exit_state_4->next = exit_state_5;
	exit_state_5->previous = exit_state_4;
	exit_state_5->next = exit_state_6;
	exit_state_6->previous = exit_state_5;
	exit_state_6->next = exit_state_7;
	exit_state_7->previous = exit_state_6;

	exit_state_7->next = time;


	


	menu = time;

}


int menuDisplayed()
{
	return drawMenuDisplayState;
}

void handleUpKey()
{
	if (menu)
	{
		menu_item* next = menu;
		int selected = 0;

		while(!selected)
		{
			next = next->next;
			selected = next->selected;
		}

		next->selected = 0;
		next->previous->selected = 1;
	}
}

void handleDownKey()
{
	if (menu)
	{
		menu_item* next = menu;
		int selected = 0;

		while(!selected)
		{
			next = next->next;
			selected = next->selected;
		}

		next->selected = 0;
		next->next->selected = 1;
	}
}

void handleLeftKey()
{
	if (menu)
	{
		menu_item* selected_item = menu;
		int selected = 0;

		while(!selected)
		{
			selected_item = selected_item->next;
			selected = selected_item->selected;
		}

		selected_item->decrease();
		if (selected_item->updateText != NULL)
			selected_item->updateText(selected_item->text);
	}
}

void handleRightKey()
{
	if (menu)
	{
		menu_item* selected_item = menu;
		int selected = 0;

		while(!selected)
		{
			selected_item = selected_item->next;
			selected = selected_item->selected;
		}

		selected_item->increase();
		if (selected_item->updateText != NULL)
			selected_item->updateText(selected_item->text);
	}
}


void drawInfoDisplay(int width, int height)
{
	if (drawInfoDisplayState)
	{			
		//draw text info
		char output_buffer[256];

		menu_width = width;
		menu_height = height;
		
		set2DProjection();
		
		text_position = 0;

                char simInfoString[] = "********** Simulation Information **********"; 
                char simEndInfoString[] = "******** End Simulation Information ********"; 
                
		glColor3f(0.0, 0.0, 0.0);
		printInfoLine(simInfoString);

		sprintf(output_buffer,"Current Frames Per Second: %f", getFPS());
		printInfoLine(output_buffer);

		sprintf(output_buffer,"Current Pedestrian Agent Count: %i", get_agent_agent_default_count());
		printInfoLine(output_buffer);

		sprintf(output_buffer,"Maximum Pedestrian Agent Count: %i", get_agent_agent_MAX_count());
		printInfoLine(output_buffer);

		sprintf(output_buffer,"Navigation Map Grid Cells: %i", get_agent_navmap_static_count());
		printInfoLine(output_buffer);

		sprintf(output_buffer,"Emmission Rate: %f", getEmmisionRateExit1());
		printInfoLine(output_buffer);

		printInfoLine(simEndInfoString);
	}
}

void toggleInformationDisplayOnOff()
{
	drawInfoDisplayState = !drawInfoDisplayState;
}

void setInformationDisplayOnOff(int state)
{
	drawInfoDisplayState = state;
}

void drawMenuDisplay(int width, int height)
{
	if (drawMenuDisplayState)
	{			
		menu_item* next;

		menu_width = width;
		menu_height = height;
		
		set2DProjection();

		text_position = 0;

                char simMenueString[] = "********** Simulation Menu **********"; 
                char simEndMenueString[] = "******** End Simulation Menu ********"; 
                
		glColor3f(0.0f,0.0f,0.0f);
		printInfoLine(simMenueString);

		//print menu
		if (menu)
		{
			printMenuItem(menu);

			next = menu->next;

			while(next != menu)
			{
				printMenuItem(next);
				next = next->next;
			}
		}

		glColor3f(0.0f,0.0f,0.0f);
		printInfoLine(simEndMenueString);
	}
}

void toggleMenuDisplayOnOff()
{
	drawMenuDisplayState = !drawMenuDisplayState;
}

void setMenuDisplayOnOff(int state)
{
	drawMenuDisplayState = state;
}

void updateAllTexts(){
	// Iterate the linked list updating menu item texts, until we get to the start again.
	menu_item* nextItem = menu;
	do{
		nextItem->updateText(nextItem->text);
		nextItem = nextItem->next;
	} while(nextItem != menu);
}

void updateAllEmmsionRatesTexts(char* text)
{
	em_rate1->updateText(em_rate1->text);
	em_rate2->updateText(em_rate2->text);
	em_rate3->updateText(em_rate3->text);
	em_rate4->updateText(em_rate4->text);
	em_rate5->updateText(em_rate5->text);
	em_rate6->updateText(em_rate6->text);
	em_rate7->updateText(em_rate7->text);
}

void updateAllProbabilityTexts(char* text){
	setProbabilityExit1Text(probability_1->text);
	setProbabilityExit2Text(probability_2->text);
	setProbabilityExit3Text(probability_3->text);
	setProbabilityExit4Text(probability_4->text);
	setProbabilityExit5Text(probability_5->text);
	setProbabilityExit6Text(probability_6->text);
	setProbabilityExit7Text(probability_7->text);
}

void set2DProjection()
{
	//set projection mode for rendering text
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, menu_width, menu_height, 0);
	glScalef(1, -1, 1);
	glTranslatef(0, -menu_height, 0);
	glColor3f(1.0, 1.0, 1.0);

	glDisable( GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
}

void printInfoLine(char *string)
{
 
  char *c;
  glRasterPos2f(10.0f, (menu_height - 20) - text_position);
  for (c=string; *c != '\0'; c++) {
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);
  }
  text_position += 20.0f;
}

void printMenuItem(menu_item* menu_item)
{
 
	if (menu_item->selected)
		  glColor3f(1.0f,0.0f,0.0f);
	else
		  glColor3f(0.0f,0.0f,0.0f);

	printInfoLine(menu_item->text);
}
