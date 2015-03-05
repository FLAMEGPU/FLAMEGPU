
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_agents_default, d_agents_default, sizeof(xmachine_memory_agent_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying agent Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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

	//Write each agent agent to xml
	for (int i=0; i<h_xmachine_memory_agent_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>agent</name>\n", file);
        
		fputs("<location_id>", file);
        sprintf(data, "%i", h_agents_default->location_id[i]);
		fputs(data, file);
		fputs("</location_id>\n", file);
        
		fputs("<agent_id>", file);
        sprintf(data, "%i", h_agents_default->agent_id[i]);
		fputs(data, file);
		fputs("</agent_id>\n", file);
        
		fputs("<state>", file);
        sprintf(data, "%i", h_agents_default->state[i]);
		fputs(data, file);
		fputs("</state>\n", file);
        
		fputs("<sugar_level>", file);
        sprintf(data, "%i", h_agents_default->sugar_level[i]);
		fputs(data, file);
		fputs("</sugar_level>\n", file);
        
		fputs("<metabolism>", file);
        sprintf(data, "%i", h_agents_default->metabolism[i]);
		fputs(data, file);
		fputs("</metabolism>\n", file);
        
		fputs("<env_sugar_level>", file);
        sprintf(data, "%i", h_agents_default->env_sugar_level[i]);
		fputs(data, file);
		fputs("</env_sugar_level>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count)
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
    int in_agent_location_id;
    int in_agent_agent_id;
    int in_agent_state;
    int in_agent_sugar_level;
    int in_agent_metabolism;
    int in_agent_env_sugar_level;

	/* for continuous agents: set agent count to zero */
	
	/* Variables for initial state data */
	int agent_location_id;
	int agent_agent_id;
	int agent_state;
	int agent_sugar_level;
	int agent_metabolism;
	int agent_env_sugar_level;
	
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
	in_agent_location_id = 0;
	in_agent_agent_id = 0;
	in_agent_state = 0;
	in_agent_sugar_level = 0;
	in_agent_metabolism = 0;
	in_agent_env_sugar_level = 0;
	//set all agent values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_agent_MAX; k++)
	{	
		h_agents->location_id[k] = 0;
		h_agents->agent_id[k] = 0;
		h_agents->state[k] = 0;
		h_agents->sugar_level[k] = 0;
		h_agents->metabolism[k] = 0;
		h_agents->env_sugar_level[k] = 0;
	}
	

	/* Default variables for memory */
    agent_location_id = 0;
    agent_agent_id = 0;
    agent_state = 0;
    agent_sugar_level = 0;
    agent_metabolism = 0;
    agent_env_sugar_level = 0;

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
				if(strcmp(agentname, "agent") == 0)
				{		
					if (*h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent agent exceeded whilst reading data\n", xmachine_memory_agent_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_agents->location_id[*h_xmachine_memory_agent_count] = agent_location_id;
					h_agents->agent_id[*h_xmachine_memory_agent_count] = agent_agent_id;
					h_agents->state[*h_xmachine_memory_agent_count] = agent_state;
					h_agents->sugar_level[*h_xmachine_memory_agent_count] = agent_sugar_level;
					h_agents->metabolism[*h_xmachine_memory_agent_count] = agent_metabolism;
					h_agents->env_sugar_level[*h_xmachine_memory_agent_count] = agent_env_sugar_level;
					(*h_xmachine_memory_agent_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                agent_location_id = 0;
                agent_agent_id = 0;
                agent_state = 0;
                agent_sugar_level = 0;
                agent_metabolism = 0;
                agent_env_sugar_level = 0;

			}
			if(strcmp(buffer, "location_id") == 0) in_agent_location_id = 1;
			if(strcmp(buffer, "/location_id") == 0) in_agent_location_id = 0;
			if(strcmp(buffer, "agent_id") == 0) in_agent_agent_id = 1;
			if(strcmp(buffer, "/agent_id") == 0) in_agent_agent_id = 0;
			if(strcmp(buffer, "state") == 0) in_agent_state = 1;
			if(strcmp(buffer, "/state") == 0) in_agent_state = 0;
			if(strcmp(buffer, "sugar_level") == 0) in_agent_sugar_level = 1;
			if(strcmp(buffer, "/sugar_level") == 0) in_agent_sugar_level = 0;
			if(strcmp(buffer, "metabolism") == 0) in_agent_metabolism = 1;
			if(strcmp(buffer, "/metabolism") == 0) in_agent_metabolism = 0;
			if(strcmp(buffer, "env_sugar_level") == 0) in_agent_env_sugar_level = 1;
			if(strcmp(buffer, "/env_sugar_level") == 0) in_agent_env_sugar_level = 0;
			
			
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
				if(in_agent_location_id){ 
                    agent_location_id = (int) atoi(buffer);    
                }
				if(in_agent_agent_id){ 
                    agent_agent_id = (int) atoi(buffer);    
                }
				if(in_agent_state){ 
                    agent_state = (int) atoi(buffer);    
                }
				if(in_agent_sugar_level){ 
                    agent_sugar_level = (int) atoi(buffer);    
                }
				if(in_agent_metabolism){ 
                    agent_metabolism = (int) atoi(buffer);    
                }
				if(in_agent_env_sugar_level){ 
                    agent_env_sugar_level = (int) atoi(buffer);    
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

