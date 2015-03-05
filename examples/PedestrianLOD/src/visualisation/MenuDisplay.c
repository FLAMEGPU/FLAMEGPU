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
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include "MenuDisplay.h"
#include "GlobalsController.h"
#include "CustomVisualisation.h"

//holder for width and height
int window_width;
int window_height;

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
void set2DProjection();
void printInfoLine(char *string);
void printMenuItem(menu_item* menu_item);


//menu items
menu_item* menu;

void initMenuItems()
{
	menu_item *time, *steer, *avoid, *collision, *goal;

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

	//build linked list
	time->previous = goal;
	time->next = steer;
	steer->previous = time;
	steer->next = avoid;
	avoid->previous = steer;
	avoid->next = collision;
	collision->previous = avoid;
	collision->next = goal;
	goal->previous = collision;
	goal->next = time;

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

		window_width = width;
		window_height = height;
		
		set2DProjection();
		
		text_position = 0;

		glColor3f(0.0, 0.0, 0.0);
		printInfoLine("********** Simulation Information **********");

		sprintf(output_buffer,"Current Frames Per Second: %f", getFPS());
		printInfoLine(output_buffer);

		sprintf(output_buffer,"Current Pedestrian Agent Count: %i", get_agent_agent_default_count());
		printInfoLine(output_buffer);

		sprintf(output_buffer,"Maximum Pedestrian Agent Count: %i", get_agent_agent_MAX_count());
		printInfoLine(output_buffer);

		printInfoLine("******** End Simulation Information ********");
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

		window_width = width;
		window_height = height;
		
		set2DProjection();

		text_position = 0;

		glColor3f(0.0f,0.0f,0.0f);
		printInfoLine("********** Simulation Menu **********");

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
		printInfoLine("******** End Simulation Menu ********");
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


void set2DProjection()
{
	//set projection mode for rendering text
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, window_width, window_height, 0);
	glScalef(1, -1, 1);
	glTranslatef(0, -window_height, 0);
	glColor3f(1.0, 1.0, 1.0);

	glDisable( GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
}

void printInfoLine(char *string)
{
 
  char *c;
  glRasterPos2f(10.0f, (window_height-20)-text_position);
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