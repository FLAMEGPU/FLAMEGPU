
/*
* FLAME GPU v 1.4.0 for CUDA 6
* Copyright 2015 University of Sheffield.
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

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
	

// include header
#include "header.h"

float3 agent_maximum;
float3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent Memeory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent Memeory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void readFloatArrayInput(char* buffer, float *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent Memeory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = (float)atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent Memeory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_cell_list* h_cells_default, xmachine_memory_cell_list* d_cells_default, int h_xmachine_memory_cell_default_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_cells_default, d_cells_default, sizeof(xmachine_memory_cell_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying cell Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing itteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each cell agent to xml
	for (int i=0; i<h_xmachine_memory_cell_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>cell</name>\n", file);
        
		fputs("<state>", file);
        sprintf(data, "%i", h_cells_default->state[i]);
		fputs(data, file);
		fputs("</state>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%i", h_cells_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%i", h_cells_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_cell_list* h_cells, int* h_xmachine_memory_cell_count)
{

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;
    int in_cell_state;
    int in_cell_x;
    int in_cell_y;

	/* for continuous agents: set agent count to zero */
	
	/* Variables for initial state data */
	int cell_state;
	int cell_x;
	int cell_y;
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("error opening initial states\n");
		exit(0);
	}
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;
	in_cell_state = 0;
	in_cell_x = 0;
	in_cell_y = 0;
	//set all cell values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_cell_MAX; k++)
	{	
		h_cells->state[k] = 0;
		h_cells->x[k] = 0;
		h_cells->y[k] = 0;
	}
	

	/* Default variables for memory */
    cell_state = 0;
    cell_x = 0;
    cell_y = 0;

	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;
			
			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "cell") == 0)
				{		
					if (*h_xmachine_memory_cell_count > xmachine_memory_cell_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent cell exceeded whilst reading data\n", xmachine_memory_cell_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_cells->state[*h_xmachine_memory_cell_count] = cell_state;
					h_cells->x[*h_xmachine_memory_cell_count] = cell_x;//Check maximum x value
                    if(agent_maximum.x < cell_x)
                        agent_maximum.x = (float)cell_x;
                    //Check minimum x value
                    if(agent_minimum.x > cell_x)
                        agent_minimum.x = (float)cell_x;
                    
					h_cells->y[*h_xmachine_memory_cell_count] = cell_y;//Check maximum y value
                    if(agent_maximum.y < cell_y)
                        agent_maximum.y = (float)cell_y;
                    //Check minimum y value
                    if(agent_minimum.y > cell_y)
                        agent_minimum.y = (float)cell_y;
                    
					(*h_xmachine_memory_cell_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                cell_state = 0;
                cell_x = 0;
                cell_y = 0;

			}
			if(strcmp(buffer, "state") == 0) in_cell_state = 1;
			if(strcmp(buffer, "/state") == 0) in_cell_state = 0;
			if(strcmp(buffer, "x") == 0) in_cell_x = 1;
			if(strcmp(buffer, "/x") == 0) in_cell_x = 0;
			if(strcmp(buffer, "y") == 0) in_cell_y = 1;
			if(strcmp(buffer, "/y") == 0) in_cell_y = 0;
			
			
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				if(in_cell_state){ 
                    cell_state = (int) atoi(buffer);    
                }
				if(in_cell_x){ 
                    cell_x = (int) atoi(buffer);    
                }
				if(in_cell_y){ 
                    cell_y = (int) atoi(buffer);    
                }
				
			}
			
			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
}

float3 getMaximumBounds(){
    return agent_maximum;
}

float3 getMinimumBounds(){
    return agent_minimum;
}

