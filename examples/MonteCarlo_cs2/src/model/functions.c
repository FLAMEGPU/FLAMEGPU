
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

#define DELTA_T 0.0005f //time step
#define B0 2*10^8f
#define G0 0.0.00168f
#define b 0.22615f
#define y 1f
#define m0 1 // real number of crystals per unit volume

#define L_MAX 2.0
#define NL_MAX 0.4
#define BIN_WIDTH 0.1
#define BIN_COUNT L_MAX/BIN_WIDTH

int *nuclNo;

__FLAME_GPU_INIT_FUNC__ void initConstants()
{
	float delta_t = 0.0.875f; // high index aggregation
	set_DELTA_T(&delta_t);

	float b0 = 210^8f;
	set_B0(&b0);

	float g0 = 0.0.00168f; // high index aggregation
	set_G0(&g0);

	float b_ = 0.22615f; // high index aggregation
	set_b(&b_);

	float y_ = 1f; // high index aggregation
	set_y(&y_);

	int m0_ = 1; // high index aggregation
	set_m0_(&m0_);

	printf("FLAME GPU Init function. DELTA_T=%f, G0=%f, b=%f, y=%f, m0= %d\n", DELTA_T, G0, b, y, m0);
}


// The function calculates the nucleation number needed for CASE STUDY 2
__FLAME_GPU_STEP_FUNC__ void nuclNo_func(){

	int nc_no = h_B0*h_DELTA_T*10^7; // equation 13, nuclNo = (B0*DELTA_T*popnNo)/m0, m0/Nc is the order of 10^7

	cudamalloc((void**)&nuclNo, sizeof(int));
    gpuErrchk(cudaMemcpy(nuclNo, &nc_no, sizeof(int), cudaMemcpyHostToDevice));

	printf("FLAME GPU Step function. nuclNo is %d\n", nc_no);
}

__FLAME_GPU_EXIT_FUNC__ void hist_func(){

	printf("FLAME GPU Exit function");
	FILE *hist_output = fopen("hist.dat", "a"); // write only - append

	for (int i=0; i<BIN_COUNT; i++){
		int count = count_crystal_default_bin_variable(i);
		printf("bin index=%d, count = %d", i, count);
		fprintf(hist_output,"%f %d\n", i*BIN_WIDTH, count);
		//output into same format as initial states
	}
		fprintf(hist_output,"\n\n\n");
	fclose(hist_output);
}

// we may not be needing this function as we can create a an input .xml file to include all possible inputs
// so, no agents will create another agent
__FLAME_GPU_FUNC__ int output_crystals(xmachine_memory_crystal* agent, xmachine_message_crystal_list* internal_coords){

	//output crystal internal coordinates
    add_internal_coord(internal_coords, agent->l, 0);

	return 0;
}


__FLAME_GPU_FUNC__ int nucleate(xmachine_memory_crystal* agent, xmachine_message_internal_coords_list* internal_coords_messages){

// popnNo is constant, equal to the total number of crystals
popnNo = d_xmachine_memory_crystal_count;

//limit is to 1 thread
while (<nuclNo )  // not done
// new row is added with the size equal to zero
add_internal_coord(internal_coords, 0);

    return 0;
}


__FLAME_GPU_FUNC__ int growth(xmachine_memory_crystal* agent){

 int l_new = agent->l + (DELTA_T*G0*((1+y*agent->l)^b)); // equation 20, Lj = Lj + Gj*DELTA_T

 agent->l = l_new
 return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
