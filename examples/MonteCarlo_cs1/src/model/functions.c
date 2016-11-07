
/*
 * Copyright 2016 University of Sheffield.
 * Authors: Dr Paul Richmond , Dr Mozhgan Kabiri Chimeh
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *          m.kabiri-chimeh@sheffield.ac.uk (http://www.mkchimeh.staff.shef.ac.uk)
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define I_agg 0.0.875f // high index aggregation
#define aggNo 1

float *d_dt;

__device__ float normcdff (float y); //Calculate the standard normal cumulative distribution function



__FLAME_GPU_INIT_FUNC__ void initConstants()
{
	float i_agg = 0.0.875f; // high index aggregation
	set_I_agg(&i_agg);

	int aggno = 1;
	set_aggNo(&aggno);

	printf("FLAME GPU Init function. aggno=%f, I_agg=%d\n", aggno, I_agg);
}

// The function calculates the time step needed for CASE STUDY 1 - NOT USING IT !
__FLAME_GPU_STEP_FUNC__ void DELTA_T_func(){

    beta0 = h_I_agg; // it's the aggregation number
    
    gpuErrchk(cudaMemcpyFromSymbol( &popnNo,d_xmachine_memory_crystal_count, sizeof(float))); // Nc-aggNo

	float dt = ((2*h_aggNo)/(beta0*popnNo*(1-h_I_agg)*(popnNo-h_aggNo))); // equation 10 and 11 

   	cudamalloc((void**)&d_dt, sizeof(float));
    gpuErrchk(cudaMemcpy(d_dt, &dt, sizeof(float), cudaMemcpyHostToDevice));


	printf("FLAME GPU Step function. Delta T is %f\n", dt);
}

__FLAME_GPU_EXIT_FUNC__ void exitFunction(){

	printf("FLAME GPU Exit function");
}


// we may not be needing this function as we can create a an input .xml file to include all possible inputs
// so, no agents will create another agent
__FLAME_GPU_FUNC__ int create_crystals(xmachine_memory_crystal* agent, xmachine_message_crystal_list* internal_coords,RNG_rand48* rand48){


// put this in another file where you create agents
	//float growth = normcdff(rnd<DISCRETE_2D>(rand48));
    //float length = normcdff(rnd<DISCRETE_2D>(rand48)); 
	// required size distribution ?
	// cube root of the mean volume of the charge?

	//output crystal internal coordinates
    add_internal_coord(internal_coords, agent->rank, agent->l);

	return 0;
}


__FLAME_GPU_FUNC__ int check_aggregate(xmachine_memory_crystal* agent, xmachine_message_internal_coords_list* internal_coords_messages){

 xmachine_message_internal_coords* current_message = get_first_internal_coords_message(internal_coords_messages);

int lower_rank=0;

    while (current_message)
    {
		//check the rank to be less than aggNo
		if (current_message->rank <= agent->rank){
			lower_rank++;
		}
        
        current_message = get_next_internal_coords_message(current_message, internal_coords_messages);
    }
   
   if (lower_rank < 2*aggNo)// or 2*aggNo
   	agent->agg_flag=1;

    return 0;
}


__FLAME_GPU_FUNC__ int aggregate(xmachine_memory_crystal* agent, xmachine_message_internal_coords_list* internal_coords_messages,RNG_rand48* rand48){

 xmachine_message_internal_coords* current_message = get_first_internal_coords_message(internal_coords_messages);


int aggregated=0;

if (agent->agg_flag = 0)
	return 0;

else{	
    while (current_message)
    {
		//check the aggregation flag to be one, so that they can aggregate NOW
		if (current_message->agg_flag == 1){

			int l_agg = ((agent->l ^ 3)+(current_message->l^3))^(1/3); // adding the new length
			float rand_rank=rnd<DISCRETE_2D>(rand48); 
			//add_internal_coord(internal_coords, rand_rank, l_agg, 0); // first case study ignores growth

//Note: can't kill two agents at the same time. Instead of creating a function to do that, we replace one with new parameters
			current_message->aggregated=0;
			current_message->rank=rand_rank;
			current_message->l=l_agg;

			aggregated=1;
		}
        
        current_message = get_next_internal_coords_message(current_message, internal_coords_messages);
    }
}
 
     return aggregated;
}


#endif //_FLAMEGPU_FUNCTIONS
