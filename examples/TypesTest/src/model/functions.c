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

//Environment Variables
#define STATE_ALIVE 1
#define STATE_DEAD 0



//cell Agent Functions

//The following function arguments have been generated automatically by the FLAMEGPU XParser and are dependant on the function input and outputs. If they are changed manually be sure to match any arguments to the XMML specification.
//Input : 
//Output: state 
//Agent Output: 
__FLAME_GPU_FUNC__ int test_function(xmachine_memory_agent* xmemory) 
{
	//xmemory->v_ivec2 += 1;
	//xmemory->v_fvec2 += 0.1f;
	//xmemory->v_dvec2 += 0.25;
	//array value increases
	return 0;
}



#endif // #ifndef _FUNCTIONS_H_
