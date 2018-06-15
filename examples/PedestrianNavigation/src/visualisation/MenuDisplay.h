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
#ifndef __MENU_DISPLAY
#define __MENU_DISPLAY

struct menu_item{
	int selected;
	char text [128];
	void (*increase)();
	void (*decrease)();
	void (*updateText)(char* text);
	struct menu_item* next;
	struct menu_item* previous;
};

typedef struct menu_item  menu_item;

void initMenuItems();

int menuDisplayed();
void handleUpKey();
void handleDownKey();
void handleLeftKey();
void handleRightKey();

void drawInfoDisplay(int window_width, int window_height);
void toggleInformationDisplayOnOff();
void setInformationDisplayOnOff(int state);

void drawMenuDisplay(int window_width, int window_height);
void toggleMenuDisplayOnOff();
void setMenuDisplayOnOff(int state);
void updateAllTexts();
void updateAllEmmsionRatesTexts(char* text);

#endif //__MENU_DISPLAY
