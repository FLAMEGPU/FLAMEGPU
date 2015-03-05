/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond (Originally from Dawn Walker)
 * See: http://bib.oxfordjournals.org/content/11/3/334.abstract 

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

/***********************************************
** Definitions 
************************************************/
struct distance_result
{
	float nearest_distance;
	float nearest_xy;
	float nearest_z;
};

#define FALSE 0
#define TRUE  1

/* general constants*/
#ifndef PI
#define PI						3.142857143f
#endif

#define SURFACE_WIDTH 			500.0f
#define K_WIDTH		  			20.0f

/* keratinocyte cell types*/
#define K_TYPE_STEM     	 	0
#define K_TYPE_TA       	 	1
#define K_TYPE_COMM     	 	2
#define K_TYPE_CORN				3
#define K_TYPE_HACAT			4

/* forces act within this radius of a cell*/
#define FORCE_IRADIUS  			10

/* control G0 and cornicyte phases*/
#define MAX_TO_G0_CONTACT_INHIBITED_TICKS  300
#define MAX_DEAD_TICKS                     600

/* bond number constants*/
#define MAX_NUM_LATERAL_BONDS	4


__FLAME_GPU_INIT_FUNC__ void setConstants(){
	float h_calcium_level = 1.300000f;
	int h_CYCLE_LENGTH[5] = {120, 60, 0, 0, 120};
	float h_SUBSTRATE_FORCE[5] = {0.3f, 0.1f, 0.2f, 0.1f, 0.3f};
	float h_DOWNWARD_FORCE[5]	= {0.1f, 0.6f, 0.3f, 0.6f, 0.1f};
	float h_FORCE_MATRIX[5*5] = {0.06f, 0.01f, 0.01f, 0.01f, 0.0f, 
							   0.01f, 0.01f, 0.01f, 0.01f, 0.0f,  
							   0.01f, 0.01f, 0.06f, 0.01f, 0.0f,  
							   0.01f, 0.01f, 0.01f, 0.08f, 0.0f,  
							   0.01f, 0.01f, 0.01f, 0.08f, 0.0f}; 
	float h_FORCE_REP      = 0.5f;
	float h_FORCE_DAMPENER = 0.4f;
	int h_BASEMENT_MAX_Z = 5;

	set_calcium_level(&h_calcium_level);
	set_CYCLE_LENGTH(h_CYCLE_LENGTH);
	set_SUBSTRATE_FORCE(h_SUBSTRATE_FORCE);
	set_DOWNWARD_FORCE(h_DOWNWARD_FORCE);
	set_FORCE_MATRIX(h_FORCE_MATRIX);
	set_FORCE_REP(&h_FORCE_REP);
	set_FORCE_DAMPENER(&h_FORCE_DAMPENER);
	set_BASEMENT_MAX_Z(&h_BASEMENT_MAX_Z);

}



/* tests if cell is deemed to be on the substrate surface */
/* used by resolve forces, differentiate and mitigate*/
__FLAME_GPU_FUNC__ int on_substrate_surface(float z)
{
	return (z < (float) BASEMENT_MAX_Z);
}

/* used by differentiate*/
__FLAME_GPU_FUNC__ float get_ta_to_comm_diff_minor_axis(float calcium_level)
{
	return K_WIDTH * 1.5f;
}

/* used by differetiate */
__FLAME_GPU_FUNC__ float get_ta_to_comm_diff_major_axis(float calcium_level)
{
	return K_WIDTH * 5;
}

/* used in cycle */
__FLAME_GPU_FUNC__ int get_max_num_bonds(float calcium_level)
{
	return 6;
}

/* used in cycle*/
__FLAME_GPU_FUNC__ int can_stratify(int cell_type, float calcium_level)
{
	if (cell_type == K_TYPE_HACAT) {
		return FALSE;
	} else {
		return TRUE;
	}
}

/* used in differentiate */
__FLAME_GPU_FUNC__ float get_max_stem_colony_size(float calcium_level)
{
	return 20;
}

/* used in cycle */
__FLAME_GPU_FUNC__ float get_new_motility(int cell_type, float calcium_level)
{
	if (cell_type == K_TYPE_TA) {
		return 0.5f;
	} 
	else
	{
		return 0;
	}
}

/* checks if a cell can divide */
/* used in cycle*/
__FLAME_GPU_FUNC__ int divide(int type, int cycle)
{
	return (type == K_TYPE_STEM || type == K_TYPE_TA || type == K_TYPE_HACAT) && cycle > CYCLE_LENGTH[type];
}

/* returns a new coordinate based on the old one, but deviated slightly */
/* used in cycle*/
__FLAME_GPU_FUNC__ float get_new_coord(float old_coord, int pos_only, RNG_rand48* rand48)
{
	float coord = 0;

	while (coord == 0) {
		coord = rnd(rand48) * K_WIDTH / 10;
	}

	if (!pos_only && coord > 0.5f) {
		coord = -coord;
	}


	return old_coord + coord;
}

/* generate a new starting position in the cell's cycle */
/* used in cycle*/
__FLAME_GPU_FUNC__ int start_new_cycle_postion(int type, RNG_rand48* rand48)
{
	float cycle_fraction = CYCLE_LENGTH[type] / 4;
	float pos = rnd(rand48) * cycle_fraction;

	return (int) round(pos);
}


/* get the radius of an ellipse given its radii and an angle theta */
__FLAME_GPU_FUNC__ float ellipse_radius(float major_radius, float minor_radius, float theta)
{
	float a_squ = major_radius * major_radius;
	float b_squ = minor_radius * minor_radius;
	float sin_theta = sin(theta);
	float sin_theta_squ = sin_theta * sin_theta;
	float cos_theta = cos(theta);
	float cos_theta_squ = cos_theta * cos_theta;
	float r_squ = (a_squ * b_squ) / (a_squ * sin_theta_squ  + b_squ * cos_theta_squ);
	float r = sqrt(r_squ);
	return r;
}

/* is the nearest stem cell in range */
/* used in differentiate*/
__FLAME_GPU_FUNC__ int check_distance(struct distance_result nearest,
				   	   float major_radius,
				       float minor_radius,
				       float thres)
{
	if (nearest.nearest_distance == -1.0f) {
		return 1;
	} else {
		float theta = tan(nearest.nearest_z/nearest.nearest_xy);
		float er = ellipse_radius(major_radius, minor_radius, theta);
		return (thres * er < nearest.nearest_distance);
	}
}

/* test if on the edge of the colony */
/* used in differentiate */
__FLAME_GPU_FUNC__ int on_colony_edge(int num_bonds)
{
	return num_bonds <= 2;
}

/***********************************************
** Keratinocyte Agent Functions
************************************************/


//Input : 
//Output: location 
//Agent Output: 
__FLAME_GPU_FUNC__ int output_location(xmachine_memory_keratinocyte* xmemory, xmachine_message_location_list* location_messages) 
{
	add_location_message(location_messages,
		xmemory->id,
		xmemory->type,
		xmemory->x,
		xmemory->y,
		xmemory->z,
		xmemory->dir,
		xmemory->motility,
        SURFACE_WIDTH,
		0);

	return 0;
}

//Input : 
//Output: 
//Agent Output: keratinocyte 
__FLAME_GPU_FUNC__ int cycle(xmachine_memory_keratinocyte* xmemory, xmachine_memory_keratinocyte_list* keratinocyte_agents, RNG_rand48* rand48) 
{
	
	/* find number of contacts/bonds*/
	int contacts = xmemory->num_xy_bonds + xmemory->num_z_bonds;

	/* touching a wall counts as two contacts*/
	if (xmemory->x == 0 || xmemory->x == SURFACE_WIDTH) {
		contacts += 2;
	}
	if (xmemory->y == 0 || xmemory->y == SURFACE_WIDTH) {
		contacts += 2;
	}

	if (contacts <= get_max_num_bonds(calcium_level)) {
		/* cell comes out of G0*/
		xmemory->cycle = xmemory->cycle+1;
		xmemory->contact_inhibited_ticks = 0;
	} else {
		/* cell enters G0*/
		xmemory->contact_inhibited_ticks = xmemory->contact_inhibited_ticks+1;
	}

	/* check to see if enough time has elapsed as to whether cell can divide*/
	if (divide(xmemory->type, xmemory->cycle)) {

		int new_cycle 				 = start_new_cycle_postion(xmemory->type, rand48);
		float new_x                 = get_new_coord(xmemory->x, FALSE, rand48);
		float new_y                 = get_new_coord(xmemory->y, FALSE, rand48);
		float new_z                 = xmemory->z;
		float new_diff_noise_factor = 0.9f + (rnd(rand48)*0.2f);
		float new_dir               = rnd(rand48) * 2 * PI;
		float new_motility          = (0.5f + (rnd(rand48) * 0.5f)) * get_new_motility(xmemory->type, calcium_level);

		if (can_stratify(xmemory->type, calcium_level) && xmemory->num_xy_bonds >= MAX_NUM_LATERAL_BONDS) {
			new_z = get_new_coord(xmemory->z, TRUE, rand48);
		}
		xmemory->cycle = start_new_cycle_postion(xmemory->type, rand48);


		add_keratinocyte_agent(
			keratinocyte_agents,
			xmemory->id+1,
			xmemory->type,      		/* type*/
			new_x, 						/* x*/
			new_y, 						/* y*/
			new_z,						/* z*/
			0,                			/* force_x*/
			0,				  			/* force_y*/
			0,							/* force_z*/
			0,				  			/* num_xy_bonds*/
			0,				  			/* num_z_bonds*/
			0,                			/* num_stem_bonds*/
			new_cycle,        			/* cycle*/
			new_diff_noise_factor,		/* diff_noise_factor*/
			0,                  		/* dead_ticks*/
			0,  	            		/* contact_inhibited_ticks*/
			new_motility,  				/* motility*/
			new_dir,					/* dir*/
			SURFACE_WIDTH	  	  		/* range*/
			);
			
	}
	return 0;
}

//Input : location 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int differentiate(xmachine_memory_keratinocyte* xmemory, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix) 
{
	float x1 = xmemory->x;
	float y1 = xmemory->y;
	float z1 = xmemory->z;

	struct distance_result nearest_stem;

	nearest_stem.nearest_distance = -1.0f;
	nearest_stem.nearest_xy = -1.0f;
	nearest_stem.nearest_z  = -1.0f;

	xmachine_message_location* location_message = get_first_location_message(location_messages, partition_matrix, x1, y1, z1);
	int num_stem_neighbours = 0;



	while(location_message){

		if (xmemory->type == K_TYPE_STEM) {

			if (on_colony_edge(xmemory->num_stem_bonds)) {
				float x2 = location_message->x;
				float y2 = location_message->y;
				float z2 = location_message->z;

				if (location_message->type == K_TYPE_STEM) {
					float distance_check = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));

					float max_distance = K_WIDTH * (get_max_stem_colony_size(calcium_level) / 2);

					if (distance_check < max_distance) {
						num_stem_neighbours ++;
					}
				}
			}

		} else if (xmemory->type  == K_TYPE_TA) {
			/* If the TA cell is too far from the stem cell centre, it differentiates into in a committed cell.*/

			float x2 = location_message->x;
			float y2 = location_message->y;
			float z2 = location_message->z;

			if (location_message->type == K_TYPE_STEM) {
				float distance_check = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
				if (nearest_stem.nearest_distance == -1.0 || distance_check < nearest_stem.nearest_distance) {
					nearest_stem.nearest_distance = distance_check;
					nearest_stem.nearest_xy = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
					nearest_stem.nearest_z  = fabs(z1-z2);
				}
			}

		}

		//load next message
		location_message = get_next_location_message(location_message, location_messages, partition_matrix);
	}

	//after processing
	if (xmemory->type == K_TYPE_STEM) {
		/* For stem cells, we check if the colony is too big and if they are on the edge.*/
		/* If so, they differentiate into TA cells.	*/
		if (on_colony_edge(xmemory->num_stem_bonds)) {
			if (num_stem_neighbours > get_max_stem_colony_size(calcium_level)) {
				xmemory->type = K_TYPE_TA;
			}
		}

		/* If the cell stratifies, it also differentiates into a TA cell*/
		if (!on_substrate_surface(xmemory->z)) {
			xmemory->type = K_TYPE_TA;
		}

	}else if (xmemory->type  == K_TYPE_TA) {
		int do_diff = check_distance(nearest_stem,
									 get_ta_to_comm_diff_major_axis(calcium_level),
									 get_ta_to_comm_diff_minor_axis(calcium_level),
									 xmemory->diff_noise_factor);


		if (do_diff) {
			xmemory->type = K_TYPE_COMM;

		/* If it has been in G0 for a certain period, it also differentiates.*/
		} else if (xmemory->contact_inhibited_ticks >= MAX_TO_G0_CONTACT_INHIBITED_TICKS) {
			xmemory->type = K_TYPE_COMM;
		}

	}else if (xmemory->type == K_TYPE_COMM) {
		/* after a period as a committed cell, it dies for good - differentiation into a corneocyte*/

		xmemory->dead_ticks = xmemory->dead_ticks+1;
		if (xmemory->dead_ticks > MAX_DEAD_TICKS) {
			xmemory->type = K_TYPE_CORN;
		}

	} else if (xmemory->type == K_TYPE_HACAT) {

		if (xmemory->contact_inhibited_ticks >= MAX_TO_G0_CONTACT_INHIBITED_TICKS) {
			xmemory->type = K_TYPE_COMM;
		}
	}
	

	return 0;
}

//Input : location 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int death_signal(xmachine_memory_keratinocyte* xmemory, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, RNG_rand48* rand48) 
{
	float x1 = xmemory->x;
	float y1 = xmemory->y;
	float z1 = xmemory->z;

	int num_corn_neighbours = 0;

	xmachine_message_location* location_message = get_first_location_message(location_messages, partition_matrix, x1, y1, z1);
	while(location_message)
	{
		if (xmemory->type != K_TYPE_CORN) {
			float x2 = location_message->x;
			float y2 = location_message->y;
			float z2 = location_message->z;

			if (location_message->type == K_TYPE_CORN) {
				float distance_check = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
				if (distance_check != 0) {
					if (distance_check - K_WIDTH <= FORCE_IRADIUS) {
						num_corn_neighbours ++;
					}
				}
			}
		}

		location_message = get_next_location_message(location_message, location_messages, partition_matrix);
	}

	float prob = num_corn_neighbours * num_corn_neighbours * 0.01f;

	if (rnd(rand48) < prob) {
		/* jump through another hoop*/
		if (rnd(rand48) < 0.01f) {
			xmemory->type = K_TYPE_CORN;
		}
	}

	return 0;
}

//Input : location 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int migrate(xmachine_memory_keratinocyte* xmemory, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, RNG_rand48* rand48) 
{
	/* these are the 'current' parameters*/
	float x1 = xmemory->x;
	float y1 = xmemory->y;
	float dir1 = xmemory->dir;
	float motility1 = xmemory->motility;

	/* if rnd less than 0.1, then changed direction within +/- 45 degrees*/
	if (rnd(rand48) < 0.1f) {
		dir1 += PI * rnd(rand48)/4.0f;
	}
	x1 += motility1 * cos(dir1);
	y1 += motility1 * sin(dir1);

	
	// check if we're about to bump into a stationary cell
	xmachine_message_location* location_message = get_first_location_message(location_messages, partition_matrix, x1, y1, 0);
	while(location_message)	{

		// only TAs and HACATs can move
		if (xmemory->type == K_TYPE_TA || xmemory->type == K_TYPE_HACAT) {

			float x2        = location_message->x;
			float y2        = location_message->y;
			float z2        = location_message->z;
			float motility2 = location_message->motility;


			// check if we're on the base of the dish and other cell is stationary
			if (on_substrate_surface(z2) && motility2 == 0)
			{
				// find distance
				float distance_check = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
				if (distance_check != 0 && distance_check < K_WIDTH) {
					dir1 -= PI;

					// reverse direction
					if (dir1 > 2 * PI) {
						dir1 -= 2 * PI;
					}

					x1 = xmemory->x + motility1 * cos(dir1);
					y1 = xmemory->y + motility1 * sin(dir1);
				}

			}

		}

		location_message = get_next_location_message(location_message, location_messages, partition_matrix);
	}
	

	/* update memory with new parameters*/
	xmemory->x = x1;
	xmemory->y = y1;
	xmemory->dir = dir1;
	xmemory->motility = motility1;

	/* check we've not gone over the edge of the dish!*/
	/* if so, reverse direction*/
	if (xmemory->x > SURFACE_WIDTH) {
		xmemory->x   = SURFACE_WIDTH - xmemory->motility *rnd(rand48);
		xmemory->dir = PI + PI * (rnd(rand48)-0.5f)/4.0f;
	}
	if (xmemory->x < 0) {
		xmemory->x   = xmemory->motility *rnd(rand48);
		xmemory->dir = PI * (rnd(rand48)-0.5f)/4.0f;
	}
	if (xmemory->y > SURFACE_WIDTH) {
		xmemory->y   = SURFACE_WIDTH -xmemory->motility *rnd(rand48);
		xmemory->dir = (3.0f * PI/2.0f) + PI * (rnd(rand48)-0.5f)/4.0f;
	}
	if (xmemory->y < 0) {
		xmemory->y   = xmemory->motility * rnd(rand48);
		xmemory->dir = (PI/2.0f) + PI * (rnd(rand48)-0.5f)/4.0f;
	}

	return 0;
}

//Input : 
//Output: force 
//Agent Output: 
__FLAME_GPU_FUNC__ int force_resolution_output(xmachine_memory_keratinocyte* xmemory, xmachine_message_force_list* force_messages) 
{
	add_force_message(force_messages,
		xmemory->type,
		xmemory->x,
		xmemory->y,
		xmemory->z,
		xmemory->id);

	return 0;
}

//Input : force 
//Output: 
//Agent Output: 
__FLAME_GPU_FUNC__ int resolve_forces(xmachine_memory_keratinocyte* xmemory, xmachine_message_force_list* force_messages, xmachine_message_force_PBM* partition_matrix) 
{

	float x1 = xmemory->x;
	float y1 = xmemory->y;
	float z1 = xmemory->z;
	int type1 = xmemory->type;

	int num_xy_bonds   = 0;
	int num_z_bonds    = 0;
	int num_stem_bonds = 0;

	xmemory->force_x = 0;
	xmemory->force_y = 0;
	xmemory->force_z = 0;

	xmachine_message_force* force_message = get_first_force_message(force_messages, partition_matrix, x1, y1, z1);


	while (force_message) {


	    float x2 = force_message->x; 
	    float y2 = force_message->y;
	    float z2 = force_message->z;
	    int type2 = force_message->type;

	    float distance_check = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)  + (z1-z2)*(z1-z2));

		if (distance_check != 0.0f)
		{
			float force;
			float separation_distance = (distance_check - K_WIDTH);
	        if (separation_distance <= FORCE_IRADIUS) {

				if (z2 >= z1) {
					if (z2 - z1 > (K_WIDTH/2)) {
						num_z_bonds ++;
					}  else {
						num_xy_bonds ++;
					}
				}

				if (force_message->type == K_TYPE_STEM) {
					num_stem_bonds ++;
				}

	            if (separation_distance > 0.0f) {
	            	force = FORCE_MATRIX[type1+ (type2*5)];
	            } else {
	            	force = FORCE_REP;
	            }

	            if (on_substrate_surface(z1)) {
	            	force *= DOWNWARD_FORCE[xmemory->type];
	            }

				force *= FORCE_DAMPENER;

				xmemory->force_x = (xmemory->force_x + force * (separation_distance)*((x2-x1)/distance_check));
				xmemory->force_y = (xmemory->force_y + force * (separation_distance)*((y2-y1)/distance_check));
				xmemory->force_z = (xmemory->force_z + force * (separation_distance)*((z2-z1)/distance_check));

	    	}
	    }
		force_message = get_next_force_message(force_message, force_messages, partition_matrix);
	}




	/* attraction force to substrate*/
	if (z1 <= (K_WIDTH * 1.5f)) {
		xmemory->force_z = (xmemory->force_z - SUBSTRATE_FORCE[type1]);
	}

	xmemory->num_xy_bonds = num_xy_bonds;
	xmemory->num_z_bonds = num_z_bonds;
	xmemory->num_stem_bonds = num_stem_bonds;

	x1 += xmemory->force_x;
	y1 += xmemory->force_y;
	z1 += xmemory->force_z;

	if (x1 < 0) {
		x1 = 0;
	}

	if (y1 < 0) {
		y1 = 0;
	}

	if (z1 < 0) {
		z1 = 0;
	}

	if (x1 > SURFACE_WIDTH) {
		x1 = SURFACE_WIDTH;
	}

	if (y1 > SURFACE_WIDTH) {
		y1 = SURFACE_WIDTH;
	}
	
	xmemory->movement = sqrt((x1-xmemory->x)*(x1-xmemory->x) + (y1-xmemory->y)*(y1-xmemory->y)  + (z1-xmemory->z)*(z1-xmemory->z));


	xmemory->x = x1;
	xmemory->y = y1;
	xmemory->z = z1;

	return 0;
}



#endif // #ifndef _FUNCTIONS_H_
