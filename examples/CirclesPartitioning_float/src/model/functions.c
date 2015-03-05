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

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"

#define radius 2.0


__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Circle* xmemory, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix)
{
	
	int kr = 0.1; /* Stiffness variable for repulsion */
	int ka = 0.0; /* Stiffness variable for attraction */

	float x1, y1, x2, y2;
    float deep_distance_check, separation_distance;
    float k;
    xmemory->fx = 0.0f;
    xmemory->fy = 0.0f;
    x1 = xmemory->x;
    y1 = xmemory->y;
    
    // Loop through all messages 
	xmachine_message_location* location_message = get_first_location_message(location_messages, partition_matrix, (float)xmemory->x, (float)xmemory->y, (float)xmemory->z);
	
	
	int count = 0;
    while(location_message)
    {
		count++;

        if((location_message->id != xmemory->id))
        {
            x2 = location_message->x;
            y2 = location_message->y;
            //Deep (expensive) check 
            deep_distance_check = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
            separation_distance = (deep_distance_check - radius - radius);
            if(separation_distance < radius)
            {
                if(separation_distance > 0.0) k = ka;
                else k = kr;
                xmemory->fx += k*(separation_distance)*((x2-x1)/deep_distance_check);
                xmemory->fy += k*(separation_distance)*((y2-y1)/deep_distance_check);
				
            }
        }
        //Move onto next message to check 
        location_message = get_next_location_message(location_message, location_messages, partition_matrix);
    }
	
	
	//xmemory->fx = count;
	return 0;
}

__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Circle* xmemory, xmachine_message_location_list* location_messages)
{
    float x, y, z;

	x = xmemory->x;
    y = xmemory->y;
	z = xmemory->z;
    
	add_location_message(location_messages, xmemory->id, x, y, z);

	return 0;
}

__FLAME_GPU_FUNC__ int move(xmachine_memory_Circle* xmemory)
{

	xmemory->x += xmemory->fx;
	xmemory->y += xmemory->fy;
	
	return 0;
}


#endif // #ifndef _FUNCTIONS_H_
