
#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
#include "cutil_math.h"


#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x


#if _DEBUG
#define DEBUG_LOG( x, ... ) printf( x, __VA_ARGS__ );
#define DEBUG_LOG_POSITION( name, position ) printf( name ": %f, %f, %f\r\n", position.x, position.y, position.z )
#else
#define DEBUG_LOG( x, ... )
#define DEBUG_LOG_POSITION( name, position )
#endif

//#define tol_h 1.0e-4
#define epsilon 1.0e-3
#define emsmall 1.0e-12
#define g 9.80665 
#define sqrt3 1.73205081
//#define Krivo_threshold 0.99
//#define Manning 0.000
//#define CFL 0.3
#define BIG_NUMBER 800000
#define INITIAL_MAXDEPTH 10

__device__ void DG2_2D(double dx_loc,
	double dy_loc,
	double3& F_pos_x,
	double3& F_neg_x,
	double3& G_pos_y,
	double3& G_neg_y,
	double z0x_loc,
	double z0y_loc,
	double z1x_loc,
	double z1y_loc,
	double et0_loc,
	double et1x_loc,
	double et1y_loc,
	double qx0_loc,
	double qx1x_loc,
	double qx1y_loc,
	double qy0_loc,
	double qy1x_loc,
	double qy1y_loc,
	double3& L0_loc,
	double3& L1x_loc,
	double3& L1y_loc
);

inline __device__ double3 hll_x(double zb_LR, double z_L, double z_R, double qx_L, double qx_R, double qy_L, double qy_R);
inline __device__ double3 hll_y(double zb_SN, double z_S, double z_N, double qx_S, double qx_N, double qy_S, double qy_N);

enum ECellDirection { NORTH = 1, EAST = 2, SOUTH = 3, WEST = 4, NORTHEAST = 5, NORTHWEST = 6, SOUTHEAST = 7, SOUTHWEST = 8 };

struct __align__(16) AgentFlowData
{
	double zb0;
	double zb1x;
	double zb1y;

	double z0;
	double qx0;
	double qy0;

	double z1x;
	double qx1x;
	double qy1x;

	double z1y;
	double qx1y;
	double qy1y;
};

struct __align__(16) LFVResult
{
	__device__ LFVResult(double _h_face, double _z_face, double2 _qFace, double3 _xHat, double3 _yHat)
	{
		h_face = _h_face;
		z_face = _z_face;
		qFace = _qFace;
		xHat = _xHat;
		yHat = _yHat;
	}

	__device__ LFVResult()
	{
		h_face = 0.0;
		z_face = 0.0;
		qFace = make_double2(0.0, 0.0);
		xHat = make_double3(0.0, 0.0, 0.0);
		yHat = make_double3(0.0, 0.0, 0.0);
	}

	double h_face;
	double z_face;
	double2 qFace;
	double3 xHat;
	double3 yHat;

};


inline __device__ AgentFlowData GetFlowDataFromAgent(xmachine_memory_FloodCell* agent)
{
	AgentFlowData result;

	result.zb0 = agent->zb0;
	result.z1x = agent->z1x;
	result.z1y = agent->z1y;

	result.zb1x = agent->zb1x;
	result.zb1y = agent->zb1y;

	if (!INTERMEDIATE_STAGE)
	{
		result.z0 = agent->z0;
		result.qx0 = agent->qx0;
		result.qy0 = agent->qy0;

		result.z1x = agent->z1x;
		result.qx1x = agent->qx1x;
		result.qy1x = agent->qy1x;

		result.z1y = agent->z1y;
		result.qx1y = agent->qx1y;
		result.qy1y = agent->qy1y;
	}
	else
	{
		result.z0 = agent->z0_int;
		result.qx0 = agent->qx0_int;
		result.qy0 = agent->qy0_int;

		result.z1x = agent->z1x_int;
		result.qx1x = agent->qx1x_int;
		result.qy1x = agent->qy1x_int;

		result.z1y = agent->z1y_int;
		result.qx1y = agent->qx1y_int;
		result.qy1y = agent->qy1y_int;
	}

	return result;

}

inline __device__ void centbound(xmachine_memory_FloodCell* agent,
	const AgentFlowData&	flowData,
	AgentFlowData& centBoundData
)
{
	// Purpose: Set the averages and and slope coefficients (of the flow variables)
	// 		   at a missing boundary from the 'ndir' direction of cell(ic) 
	//          and at an 'nmod' RK1/2 level.
	//
	// NB. It is assumed that the ghost cell is at the same level as the present cell.

	//AgentFlowData flowData  = GetFlowDataFromAgent( agent );

	//** Assign the the same topography data for the ghost cell **!
	/*AgentFlowData ghostData;

	ghostData.zb0  = flowData.zb0;
	ghostData.zb1x = -flowData.zb1x;
	ghostData.zb1y = -flowData.zb1y;

	ghostData.z0  = flowData.z0;
	ghostData.qx0 = flowData.qx0;
	ghostData.qy0 = flowData.qy0;

	ghostData.z1x  = -flowData.z1x;
	ghostData.qx1x = -flowData.qx1x;
	ghostData.qy1x = -flowData.qy1x;

	ghostData.z1y  = -flowData.z1y_p;
	ghostData.qx1y = -flowData.qx1y_p;
	ghostData.qy1y = -flowData.qy1y_p;*/


	//Default is a reflective boundary
	centBoundData.z0 = flowData.z0;
	centBoundData.z1x = -flowData.z1x;
	centBoundData.z1y = -flowData.z1y;

	centBoundData.zb0 = flowData.zb0;
	centBoundData.zb1x = -flowData.zb1x;
	centBoundData.zb1y = -flowData.zb1y;

	centBoundData.qx0 = -flowData.qx0;
	centBoundData.qx1x = -flowData.qx1x;
	centBoundData.qx1y = -flowData.qx1y;

	centBoundData.qy0 = -flowData.qy0;
	centBoundData.qy1x = -flowData.qy1x;
	centBoundData.qy1y = -flowData.qy1y;

}

//global functions
inline __device__ float GetRandomNumber(const float minNumber, const float maxNumber, RNG_rand48* rand48)
{
	return minNumber + (rnd(rand48)*(maxNumber - minNumber));
}

inline __device__ bool IsDry(double waterHeight)
{
	return waterHeight <= TOL_H;
}

__inline __device__ double2 GetWorldPosition(xmachine_memory_FloodCell* agent, double2 offset)
{
	double x = (agent->x * DXL) + offset.x;
	double y = (agent->y * DYL) + offset.y;

	return make_double2(x, y);
}

inline __device__  double2 friction_2D(double dt_loc, double et_loc, double qx_loc, double qy_loc, double z_loc)
{
	//This function add the friction contribution to wet cells.
	//This fucntion should not be called when: minh_loc.LE.TOL_H

	double2 result;

	result.x = 0.0f;
	result.y = 0.0f;

	// WET water-depth at cell(ic)
	double h_loc = et_loc - z_loc;

	// Local velocities    
	double u_loc = qx_loc / h_loc;
	double v_loc = qy_loc / h_loc;

	// Friction forces are incative as the flow is motionless.
	if ((fabs(u_loc) <= emsmall)
		&& (fabs(v_loc) <= emsmall)
		)
	{
		result.x = qx_loc;
		result.y = qy_loc;
	}
	else
	{
		// The is motional. The FRICTIONS CONTRUBUTION HAS TO BE ADDED SO THAT IT DOESN'T REVERSE THE FLOW.

		double Cf = g * pow(GLOBAL_MANNING, 2.0) / pow(h_loc, 1.0 / 3.0);

		double expULoc = pow(u_loc, 2);
		double expVLoc = pow(v_loc, 2);

		double Sfx = -Cf * u_loc * sqrt(expULoc + expVLoc);
		double Sfy = -Cf * v_loc * sqrt(expULoc + expVLoc);

		double DDx = 1.0 + dt_loc * (Cf / h_loc * (2.0 * expULoc + expVLoc) / sqrt(expULoc + expVLoc));
		double DDy = 1.0 + dt_loc * (Cf / h_loc * (expULoc + 2.0 * expVLoc) / sqrt(expULoc + expVLoc));

		result.x = qx_loc + (dt_loc * (Sfx / DDx));
		result.y = qy_loc + (dt_loc * (Sfy / DDy));

	}

	return result;
}

inline __device__ void Friction_Implicit(xmachine_memory_FloodCell* agent, double dt)
{
	if (GLOBAL_MANNING > 0.0)
	{
		AgentFlowData flowData = GetFlowDataFromAgent(agent);

		if (flowData.z0 - flowData.zb0 <= TOL_H)
		{
			return;
		}

		double2 frict_Q0 = friction_2D(dt, flowData.z0, flowData.qx0, flowData.qy0, flowData.zb0);

		// Addition of the friction in a wet cell -- for the 1x-SLOPES discharge coefficients.
		double2 frict_Q1 = friction_2D(dt, flowData.z0 - flowData.z1x / sqrt3, flowData.qx0 - flowData.qx1x / sqrt3, flowData.qy0 - flowData.qy1x / sqrt3, flowData.zb0 - flowData.zb1x / sqrt3);

		double2 frict_Q2 = friction_2D(dt, flowData.z0 + flowData.z1x / sqrt3, flowData.qx0 + flowData.qx1x / sqrt3, flowData.qy0 + flowData.qy1x / sqrt3, flowData.zb0 + flowData.zb1x / sqrt3);

		double2 frict1 = sqrt3 / 2.0 * (frict_Q2 - frict_Q1);

		//Addition of the friction in a wet cell -- for the 1y-SLOPES discharge coefficients.
		double2 frictQ1 = friction_2D(dt, flowData.z0 - flowData.z1y / sqrt3, flowData.qx0 - flowData.qx1y / sqrt3, flowData.qy0 - flowData.qy1y / sqrt3, flowData.zb0 - flowData.zb1y / sqrt3);

		double2 frictQ2 = friction_2D(dt, flowData.z0 + flowData.z1y / sqrt3, flowData.qx0 + flowData.qx1y / sqrt3, flowData.qy0 + flowData.qy1y / sqrt3, flowData.zb0 + flowData.zb1y / sqrt3);

		double2 frict2 = sqrt3 / 2.0 * (frictQ2 - frictQ1);


		/*double2 frict_Q0 = make_double2( 0.0, 0.0 );
		double2 frict1 = make_double2( 0.0, 0.0 );
		double2 frict2 = make_double2( 0.0, 0.0 );*/

		if (!INTERMEDIATE_STAGE)
		{
			agent->qx0 = frict_Q0.x;
			agent->qy0 = frict_Q0.y;

			agent->qx1x = frict1.x;
			agent->qy1x = frict1.y;

			agent->qx1y = frict2.x;
			agent->qy1y = frict2.y;

		}
		else
		{

			agent->qx0_int = frict_Q0.x;
			agent->qy0_int = frict_Q0.y;

			agent->qx1x_int = frict1.x;
			agent->qy1x_int = frict1.y;

			agent->qx1y_int = frict2.x;
			agent->qy1y_int = frict2.y;

		}
	}

}

__FLAME_GPU_FUNC__ int PrepareWetDryFrontMessages(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages)
{
	if (agent->inDomain)
	{

		AgentFlowData flowData = GetFlowDataFromAgent(agent);

		double h0 = flowData.z0 - flowData.zb0;
		//double h1x = flowData.z1x - flowData.zb1x;
		//double h1y = flowData.z1y - flowData.zb1y;

		agent->minh_loc = h0;
		/*agent->minh_loc = min( h0, h0 - h1x/sqrt3 );
		agent->minh_loc = min( agent->minh_loc, h0 + h1x/sqrt3 );
		agent->minh_loc = min( agent->minh_loc, h0 - h1y/sqrt3 );
		agent->minh_loc = min( agent->minh_loc, h0 + h1y/sqrt3 );*/

		add_WetDryMessage_message<DISCRETE_2D>(WetDryMessage_messages, 1, agent->x, agent->y, agent->minh_loc);
	}
	else
	{
		add_WetDryMessage_message<DISCRETE_2D>(WetDryMessage_messages, 0, agent->x, agent->y, BIG_NUMBER);
	}

	return 0;

}

__FLAME_GPU_FUNC__ int ProcessWetDryFront(xmachine_memory_FloodCell* agent, xmachine_message_WetDryMessage_list* WetDryMessage_messages)
{
	if (agent->inDomain)
	{

		//looking up neighbours values for wet/dry tracking

		xmachine_message_WetDryMessage* msg = get_first_WetDryMessage_message<DISCRETE_2D>(WetDryMessage_messages, agent->x, agent->y);

		double maxHeight = agent->minh_loc;

		while (msg)
		{
			if (msg->inDomain)
			{
				//TEMP: maybe just 4 axis neighbours check here!
				//if ( ( msg->x == agent->x + 1 || msg->x == agent->x -1 ) &&  ( msg->y == agent->y + 1 || msg->y == agent->y - 1 ) )
				{
					agent->minh_loc = min(agent->minh_loc, msg->min_hloc);
				}

				if (msg->min_hloc > maxHeight)
				{
					maxHeight = msg->min_hloc;
				}
			}

			msg = get_next_WetDryMessage_message<DISCRETE_2D>(msg, WetDryMessage_messages);

		}

		agent->isDry = IsDry(maxHeight);

		if (!agent->isDry)
		{
			Friction_Implicit(agent, TIMESTEP);
		}
		else
		{
			//need to go high, so that it won't affect min calculation when it is tested again
			agent->minh_loc = BIG_NUMBER;
		}

	}

	return 0;
}

__FLAME_GPU_FUNC__ int PrepareLFVNeighbourMessages(xmachine_memory_FloodCell* agent, xmachine_message_LFVNeighbourMessage_list* LFVNeighbourMessage_messages)
{
	if (agent->inDomain)
	{
		if (agent->inflowHydrographIndex > -1)
		{

			float qXinflow = tex1D(QX_HydrographTexture0, agent->inflowHydrographIndex);
			float qYinflow = tex1D(QY_HydrographTexture0, agent->inflowHydrographIndex);
			float qZinflow = tex1D(Z_HydrographTexture0, agent->inflowHydrographIndex);

			//TEMP:  sort out this!!!
			//agent->waterLevel = 0.0; 
			agent->isDry = false;

			agent->z0 = qZinflow; //+ agent->zb0;
			agent->minh_loc = agent->z0;

			//agent->z1x = 0.0;
			//agent->z1y = 0.0;

			agent->qx0 = qXinflow; //* ( agent->z0 - agent->zb0 ); 
			agent->qx1x = 0.0;
			agent->qx1y = 0.0;

			agent->qx0_int = qXinflow; //* ( agent->z0 - agent->zb0 ); 
			agent->qx1x_int = 0.0;
			agent->qx1y_int = 0.0;

			agent->qy0 = qYinflow;// * ( agent->z0 - agent->zb0 ); 
			agent->qy1x = 0.0;
			agent->qy1y = 0.0;

			agent->qy0_int = qYinflow; //* ( agent->z0 - agent->zb0 ); 
			agent->qy1x_int = 0.0;
			agent->qy1y_int = 0.0;

			//If discharge is input on an original dry cell provide a 0.01m water depth to support the discharge !! GK(16/04/2012): problem-dependent
			if (fabs(agent->z0 - agent->zb0) < TOL_H
				&& (fabs(qXinflow) > emsmall || fabs(qYinflow) > emsmall || fabs(qZinflow) > emsmall))
			{
				agent->z0 = agent->zb0 + 0.01;
				agent->z0_int = agent->zb0 + 0.01;
			}
		}

		AgentFlowData flowData = GetFlowDataFromAgent(agent);

		//broadcast the to the surrounding cells
		add_LFVNeighbourMessage_message<DISCRETE_2D>(LFVNeighbourMessage_messages, 1, agent->x, agent->y, flowData.zb0, flowData.zb1x, flowData.zb1y, flowData.z0, flowData.z1x, flowData.z1y, flowData.qx0, flowData.qx1x, flowData.qx1y, flowData.qy0, flowData.qy1x, flowData.qy1y);
	}
	else
	{
		add_LFVNeighbourMessage_message<DISCRETE_2D>(LFVNeighbourMessage_messages, 0, agent->x, agent->y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	}


	return 0;
}

inline __device__ void FlowDataFromLFVNeighbourMessage(AgentFlowData& flowData, xmachine_message_LFVNeighbourMessage* msg)
{
	flowData.zb0 = msg->zb0;
	flowData.zb1x = msg->zb1x;
	flowData.zb1y = msg->zb1y;


	flowData.z1x = msg->z1x;
	flowData.z1y = msg->z1y;

	flowData.z0 = msg->z0;
	flowData.qx0 = msg->qx0;
	flowData.qy0 = msg->qy0;

	flowData.z1x = msg->z1x;
	flowData.qx1x = msg->qx1x;
	flowData.qy1x = msg->qy1x;

	flowData.z1y = msg->z1y;
	flowData.qx1y = msg->qx1y;
	flowData.qy1y = msg->qy1y;
}

inline __device__ double signe(double x)
{
	//This function computes the sign "s" of a real number "x"

	if (fabs(x) <= emsmall)
	{
		return 0.0;
	}
	else
	{
		return x / fabs(x);
	}
}

inline __device__ double minmod(double a1, double a2, double a3)
{
	//This function outputs the slope after limitation

	//* Output argument *!
	int s1 = (int)signe(a1);
	int s2 = (int)signe(a2);
	int s3 = (int)signe(a3);

	if (s1 == s2
		&&    s2 == s3)
	{
		return s1 * dmin(fabs(a1),
			dmin(fabs(a2), fabs(a3)));
	}
	else
	{
		return 0.0;
	}

}

inline __device__ double slope_limiting(double u0_loc,
	double u0_bwd,
	double u0_fwd,
	double u1_loc,
	double u1_bwd,
	double u1_fwd,
	double dx_loc,
	double dy_loc,
	double minh
)
{
	// purpose: limit the local slope coefficient under the TVD FV criterion.
	double u1_hat = u1_loc;

#ifndef FIRST_ORDER	

	// Local Limiting (LL) to a slope corresponding to a wet cell (i.e., minh > hmin_limiter)
	if (minh > MINHEIGHT_LIMITER)
	{

		//Preparative parameters for Krivodonova et al. (2004) interface discontinuty-detector
		double hh = sqrt(pow(dx_loc / 2.0, 2.0) + pow(dy_loc / 2.0, 2.0));

		double u_neg_loc = u0_loc - u1_loc;
		double u_pos_loc = u0_loc + u1_loc;
		double u_pos_bwd = u0_bwd + u1_bwd;
		double u_neg_fwd = u0_fwd - u1_fwd;

		double Ineg = fabs(u_neg_loc - u_pos_bwd);
		double Ipos = fabs(u_neg_fwd - u_pos_loc);

		double norm_h = max(fabs(u0_loc - u1_loc / sqrt3), fabs(u0_loc + u1_loc / sqrt3));

		double DS_neg = 0.0;
		double DS_pos = 0.0;

		if (norm_h <= emsmall)
		{
			DS_neg = KRIVO_THRESHOLD + 1.0;
			DS_pos = KRIVO_THRESHOLD + 1.0;
		}
		else
		{
			DS_neg = Ineg / (hh * norm_h);
			DS_pos = Ipos / (hh * norm_h);
		}


		// Troubled-slope detection and limiting        
		if (DS_neg > KRIVO_THRESHOLD
			|| DS_pos > KRIVO_THRESHOLD)
		{


			u1_hat = minmod(u1_loc, u0_loc - u0_bwd, u0_fwd - u0_loc);
		}
	}

#endif

	return u1_hat;
}


inline __device__ LFVResult LFV(const AgentFlowData& flowData,
	const double2& worldCellCentre,
	const double2& facePosition,
	double minh_loc,
	const AgentFlowData& eastNeighbourFlowData,
	const AgentFlowData& westNeighbourFlowData,
	const AgentFlowData& northNeighbourFlowData,
	const AgentFlowData& southNeighbourFlowData
)
{

	LFVResult result;

	// Purpose: Local face value evaluation (i.e. h_face,z_face,qx_face,qy_face)
	// of the slope limited approxiate plannar solution at the point (x_face,y_face).
	// 	The function also output the limited slope quantites for consistently accomodating the 
	// RK time update for the slope quantities. 

	// Disactivation criterion for the slope-limiting process at a cell stencil with a wet/dry front **!

	// Limiting of the "X-direction" slope  **!


	// Limiting of the local slope quantity: Component-wise*!

	// Free-surface elevation slope: z1x_hat
	result.xHat.x = slope_limiting(flowData.z0, westNeighbourFlowData.z0, eastNeighbourFlowData.z0, flowData.z1x, westNeighbourFlowData.z1x, eastNeighbourFlowData.z1x, DXL, DYL, minh_loc);

	// x-discharge component: qx1x_hat
	result.xHat.y = slope_limiting(flowData.qx0, westNeighbourFlowData.qx0, eastNeighbourFlowData.qx0, flowData.qx1x, westNeighbourFlowData.qx1x, eastNeighbourFlowData.qx1x, DXL, DYL, minh_loc);

	// y-discharge component: qy1x_hat
	result.xHat.z = slope_limiting(flowData.qy0, westNeighbourFlowData.qy0, eastNeighbourFlowData.qy0, flowData.qy1x, westNeighbourFlowData.qy1x, eastNeighbourFlowData.qy1x, DXL, DYL, minh_loc);

	// Depth slope: h1x_hat
	double h1x_hat = slope_limiting(flowData.z0 - flowData.zb0, westNeighbourFlowData.z0 - westNeighbourFlowData.zb0, eastNeighbourFlowData.z0 - eastNeighbourFlowData.zb0, flowData.z1x - flowData.zb1x, westNeighbourFlowData.z1x - westNeighbourFlowData.zb1x, eastNeighbourFlowData.z1x - eastNeighbourFlowData.zb1x, DXL, DYL, minh_loc);

	//** Limiting of the "Y-direction" slope  **!

	//* Finding "Northern" neighbour's variables
	//* Finding "Southern" neighbour's variables

	//* Limiting of the local slope quantity *!

	// Free-surface elevation slope: z1y_hat
	result.yHat.x = slope_limiting(flowData.z0, southNeighbourFlowData.z0, northNeighbourFlowData.z0, flowData.z1y, southNeighbourFlowData.z1y, northNeighbourFlowData.z1y, DXL, DYL, minh_loc);

	// x-discharge component: qx1y_hat
	result.yHat.y = slope_limiting(flowData.qx0, southNeighbourFlowData.qx0, northNeighbourFlowData.qx0, flowData.qx1y, southNeighbourFlowData.qx1y, northNeighbourFlowData.qx1y, DXL, DYL, minh_loc);

	// y-discharge component: qy1y_hat
	result.yHat.z = slope_limiting(flowData.qy0, southNeighbourFlowData.qy0, northNeighbourFlowData.qy0, flowData.qy1y, southNeighbourFlowData.qy1y, northNeighbourFlowData.qy1y, DXL, DYL, minh_loc);

	//Depth slope: h1y_hat
	double h1y_hat = slope_limiting(flowData.z0 - flowData.zb0, southNeighbourFlowData.z0 - southNeighbourFlowData.zb0, northNeighbourFlowData.z0 - northNeighbourFlowData.zb0, flowData.z1y - flowData.zb1y, southNeighbourFlowData.z1y - southNeighbourFlowData.zb1y, northNeighbourFlowData.z1y - northNeighbourFlowData.zb1y, DXL, DYL, minh_loc);

	//** Slope-limited approximated solution evaluated at the input point: (x_face,y_face) **!
	double xc = worldCellCentre.x;
	double yc = worldCellCentre.y;

#ifdef FIRST_ORDER

	result.z_face = flowData.z0;

	result.qFace.x = flowData.qx0;

	result.qFace.y = flowData.qy0;

	result.h_face = (flowData.z0 - flowData.zb0);

#else

	result.z_face = flowData.z0 + (2.0 / DXL) * (facePosition.x - xc) * result.xHat.x + (2.0 / DYL) * (facePosition.y - yc) * result.yHat.x;

	result.qFace.x = flowData.qx0 + (2.0 / DXL) * (facePosition.x - xc) * result.xHat.y + (2.0 / DYL) * (facePosition.y - yc) * result.yHat.y;

	result.qFace.y = flowData.qy0 + (2.0 / DXL) * (facePosition.x - xc) * result.xHat.z + (2.0 / DYL) * (facePosition.y - yc) * result.yHat.z;

	result.h_face = (flowData.z0 - flowData.zb0) + (2.0 / DXL) * (facePosition.x - xc) * h1x_hat + (2.0 / DYL) * (facePosition.y - yc) * h1y_hat;

#endif

	return result;
}

__FLAME_GPU_FUNC__ int ProcessLFVNeighbourMessages(xmachine_memory_FloodCell* agent, xmachine_message_LFVNeighbourMessage_list* LFVNeighbourMessage_messages)
{
	if (agent->inDomain
		&& !agent->isDry)
	{
		AgentFlowData flowData = GetFlowDataFromAgent(agent);

		AgentFlowData eastNeighbourFlowData;
		AgentFlowData westNeighbourFlowData;
		AgentFlowData northNeighbourFlowData;
		AgentFlowData southNeighbourFlowData;

		//initialise values as transmissive cells
		centbound(agent, flowData, eastNeighbourFlowData);
		centbound(agent, flowData, westNeighbourFlowData);
		centbound(agent, flowData, northNeighbourFlowData);
		centbound(agent, flowData, southNeighbourFlowData);

		//check for neighbour flow data
		xmachine_message_LFVNeighbourMessage* msg = get_first_LFVNeighbourMessage_message<DISCRETE_2D>(LFVNeighbourMessage_messages, agent->x, agent->y);

		while (msg)
		{
			if (msg->inDomain)
			{
				if (agent->x - 1 == msg->x
					&&   agent->y == msg->y)
				{
					FlowDataFromLFVNeighbourMessage(westNeighbourFlowData, msg);
				}
				else
					if (agent->x + 1 == msg->x
						&&   agent->y == msg->y)
					{
						FlowDataFromLFVNeighbourMessage(eastNeighbourFlowData, msg);
					}
					else
						if (agent->x == msg->x
							&&   agent->y + 1 == msg->y)
						{
							FlowDataFromLFVNeighbourMessage(northNeighbourFlowData, msg);
						}
						else
							if (agent->x == msg->x
								&&   agent->y - 1 == msg->y)
							{
								FlowDataFromLFVNeighbourMessage(southNeighbourFlowData, msg);
							}
			}

			msg = get_next_LFVNeighbourMessage_message<DISCRETE_2D>(msg, LFVNeighbourMessage_messages);

		}

		double2 cellPosition = GetWorldPosition(agent, make_double2(0.0, 0.0));
		double2 facePosition = GetWorldPosition(agent, make_double2(DXL * 0.5, 0.0));

		//EAST
		LFVResult faceLFV = LFV(flowData, cellPosition, facePosition, agent->minh_loc, eastNeighbourFlowData, westNeighbourFlowData, northNeighbourFlowData, southNeighbourFlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_E = faceLFV.h_face;
		agent->zFace_E = faceLFV.z_face;

		agent->qFace_E_X = faceLFV.qFace.x;
		agent->qFace_E_Y = faceLFV.qFace.y;

		/*agent->xHat_X = faceLFV.xHat.x;
		agent->xHat_Y = faceLFV.xHat.y;
		agent->xHat_Z = faceLFV.xHat.z;

		agent->yHat_X = faceLFV.yHat.x;
		agent->yHat_Y = faceLFV.yHat.y;
		agent->yHat_Z = faceLFV.yHat.z;*/

		//WEST
		facePosition = GetWorldPosition(agent, make_double2(-DXL * 0.5, 0.0));

		faceLFV = LFV(flowData, cellPosition, facePosition, agent->minh_loc, eastNeighbourFlowData, westNeighbourFlowData, northNeighbourFlowData, southNeighbourFlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_W = faceLFV.h_face;
		agent->zFace_W = faceLFV.z_face;

		agent->qFace_W_X = faceLFV.qFace.x;
		agent->qFace_W_Y = faceLFV.qFace.y;

		/*agent->xHat_X = faceLFV.xHat.x;
		agent->xHat_Y = faceLFV.xHat.y;
		agent->xHat_Z = faceLFV.xHat.z;

		agent->yHat_X = faceLFV.yHat.x;
		agent->yHat_Y = faceLFV.yHat.y;
		agent->yHat_Z = faceLFV.yHat.z; */

		//NORTH
		facePosition = GetWorldPosition(agent, make_double2(0.0, DYL * 0.5));

		faceLFV = LFV(flowData, cellPosition, facePosition, agent->minh_loc, eastNeighbourFlowData, westNeighbourFlowData, northNeighbourFlowData, southNeighbourFlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_N = faceLFV.h_face;
		agent->zFace_N = faceLFV.z_face;

		agent->qFace_N_X = faceLFV.qFace.x;
		agent->qFace_N_Y = faceLFV.qFace.y;

		/*agent->xHat_X = faceLFV.xHat.x;
		agent->xHat_Y = faceLFV.xHat.y;
		agent->xHat_Z = faceLFV.xHat.z;

		agent->yHat_X = faceLFV.yHat.x;
		agent->yHat_Y = faceLFV.yHat.y;
		agent->yHat_Z = faceLFV.yHat.z;*/

		//SOUTH 
		facePosition = GetWorldPosition(agent, make_double2(0.0, -DYL * 0.5));

		faceLFV = LFV(flowData, cellPosition, facePosition, agent->minh_loc, eastNeighbourFlowData, westNeighbourFlowData, northNeighbourFlowData, southNeighbourFlowData);

		//store for calculation in the Space_operator stage
		agent->hFace_S = faceLFV.h_face;
		agent->zFace_S = faceLFV.z_face;

		agent->qFace_S_X = faceLFV.qFace.x;
		agent->qFace_S_Y = faceLFV.qFace.y;

#ifdef FIRST_ORDER

		//store the neighbours levels for normal calc
		agent->xHat_X = eastNeighbourFlowData.z0;
		agent->xHat_Y = westNeighbourFlowData.z0;
		agent->xHat_Z = northNeighbourFlowData.z0;

		agent->yHat_X = southNeighbourFlowData.z0;

		agent->yHat_Y = 0.0;
		agent->yHat_Z = 0.0;

#else
		//store the local slopes
		agent->xHat_X = faceLFV.xHat.x;
		agent->xHat_Y = faceLFV.xHat.y;
		agent->xHat_Z = faceLFV.xHat.z;

		agent->yHat_X = faceLFV.yHat.x;
		agent->yHat_Y = faceLFV.yHat.y;
		agent->yHat_Z = faceLFV.yHat.z;

#endif
	}

	return 0;
}

__FLAME_GPU_FUNC__ int RKStage(xmachine_memory_FloodCell* agent, xmachine_message_RKStageMessage_list* RKStageMessage_messages)
{
	if (agent->inDomain
		&& !agent->isDry)
	{
		//broadcast internal LFV values to surrounding cells
		add_RKStageMessage_message<DISCRETE_2D>(RKStageMessage_messages,
			1,
			agent->x, agent->y,
			agent->hFace_E, agent->zFace_E, agent->qFace_E_X, agent->qFace_E_Y,
			agent->hFace_W, agent->zFace_W, agent->qFace_W_X, agent->qFace_W_Y,
			agent->hFace_N, agent->zFace_N, agent->qFace_N_X, agent->qFace_N_Y,
			agent->hFace_S, agent->zFace_S, agent->qFace_S_X, agent->qFace_S_Y
			);
	}
	else
	{
		//broadcast internal LFV values to surrounding cells
		add_RKStageMessage_message<DISCRETE_2D>(RKStageMessage_messages,
			0,
			agent->x, agent->y,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0
			);
	}

	return 0;
}

inline __device__ void WD(double h_L,
	double h_R,
	double et_L,
	double et_R,
	double qx_L,
	double qx_R,
	double qy_L,
	double qy_R,
	ECellDirection ndir,
	double& z_LR,
	double& et_L_star,
	double& et_R_star,
	double& qx_L_star,
	double& qx_R_star,
	double& qy_L_star,
	double& qy_R_star
)
{
	// This function provide a non-negative reconstruction of the Riemann-states.

	double z_L = et_L - h_L;
	double z_R = et_R - h_R;

	double u_L = 0.0;
	double v_L = 0.0;
	double u_R = 0.0;
	double v_R = 0.0;

	if (h_L > TOL_H)
	{
		u_L = qx_L / h_L;
		v_L = qy_L / h_L;
	}
	else
	{
		u_L = 0.0;
		v_L = 0.0;
	}


	if (h_R > TOL_H)
	{
		u_R = qx_R / h_R;
		v_R = qy_R / h_R;
	}
	else
	{
		u_R = 0.0;
		v_R = 0.0;
	}

	z_LR = max(z_L, z_R);

	double delta;

	switch (ndir)
	{
	case NORTH:
	case EAST:
	{
		delta = max(0.0, -(et_L - z_LR));
	}
	break;

	case WEST:
	case SOUTH:
	{
		delta = max(0.0, -(et_R - z_LR));
	}
	break;

	}

	double h_L_star = max(0.0, et_L - z_LR);
	et_L_star = h_L_star + z_LR;
	qx_L_star = h_L_star * u_L;
	qy_L_star = h_L_star * v_L;

	double h_R_star = max(0.0, et_R - z_LR);
	et_R_star = h_R_star + z_LR;
	qx_R_star = h_R_star * u_R;
	qy_R_star = h_R_star * v_R;

	if (delta > 0.0)
	{
		z_LR = z_LR - delta;
		et_L_star = et_L_star - delta;
		et_R_star = et_R_star - delta;
	}
}

inline __device__ double3 hll_x(double zb_LR, double z_L, double z_R, double qx_L, double qx_R, double qy_L, double qy_R)
{
	double3 F_face = make_double3(0.0, 0.0, 0.0);

	double h_L = z_L - zb_LR;
	double h_R = z_R - zb_LR;
	double u_L = 0.0;
	double v_L = 0.0;
	double u_R = 0.0;
	double v_R = 0.0;


	if ((h_L <= epsilon) && (h_R <= epsilon))
	{
		F_face.x = 0.0;
		F_face.y = pow(0.5 * (z_L + z_R), 2.0) - ((zb_LR + zb_LR) * ((z_L + z_R) * 0.5));
		F_face.y *= (0.5 * g);
		F_face.z = 0.0;

		return F_face;
	}

	if (h_L <= epsilon)
	{
		h_L = 0.0;
		u_L = 0.0;
		v_L = 0.0;
	}
	else
	{
		u_L = qx_L / h_L;
		v_L = qy_L / h_L;
	}

	if (h_R <= epsilon)
	{
		h_R = 0.0;
		u_R = 0.0;
		v_R = 0.0;
	}
	else
	{
		u_R = qx_R / h_R;
		v_R = qy_R / h_R;
	}

	double a_L = sqrt(g * h_L);
	double a_R = sqrt(g * h_R);

	double h_star = pow(((a_L + a_R) / 2.0 + (u_L - u_R) / 4.0), 2) / g;
	double u_star = (u_L + u_R) / 2.0 + a_L - a_R;
	double a_star = sqrt(g * h_star);

	double s_L = 0.0;

	if (h_L <= epsilon)
	{
		s_L = u_R - 2.0 * a_R;
	}
	else
	{
		s_L = min(u_L - a_L, u_star - a_star);
	}

	double s_R = 0.0;

	if (h_R <= epsilon)
	{
		s_R = u_L + 2.0 * a_L;
	}
	else
	{
		s_R = max(u_R + a_R, u_star + a_star);
	}

	double s_M = (s_L * h_R * (u_R - s_R) - s_R * h_L * (u_L - s_L)) / (h_R * (u_R - s_R) - h_L * (u_L - s_L));

	double3 F_L, F_R;

	F_L.x = qx_L;
	F_L.y = u_L * qx_L + 0.5 * g * (pow(z_L, 2.0) - (2.0 * zb_LR * z_L));
	F_L.z = u_L * qy_L;

	F_R.x = qx_R;
	F_R.y = u_R * qx_R + 0.5 * g * (pow(z_R, 2.0) - (2.0 * zb_LR * z_R));
	F_R.z = u_R * qy_R;

	if (s_L >= 0.0)
	{
		//F_face(1)=F_L(1)
		//F_face(2)=F_L(2)
		//F_face(3)=F_L(3)
		return F_L;
	}
	else
		if ((s_L < 0.0) && (s_R >= 0.0))
		{
			double F1_M = (s_R * F_L.x - s_L * F_R.x + s_L * s_R * (z_R - z_L)) / (s_R - s_L);
			double F2_M = (s_R * F_L.y - s_L * F_R.y + s_L * s_R * (qx_R - qx_L)) / (s_R - s_L);

			if ((s_L < 0.0) && (s_M >= 0.0))
			{
				F_face.x = F1_M;
				F_face.y = F2_M;
				F_face.z = F1_M * v_L;
			}
			else
				if ((s_M < 0.0) && (s_R >= 0.0))
				{
					F_face.x = F1_M;
					F_face.y = F2_M;
					F_face.z = F1_M * v_R;
				}
		}
		else
			if (s_R < 0.0)
			{
				//F_face(1)=F_R(1)
				//F_face(2)=F_R(2)
				//F_face(3)=F_R(3)
				return F_R;
			}

	return F_face;
}

inline __device__ double3 hll_y(double zb_SN, double z_S, double z_N, double qx_S, double qx_N, double qy_S, double qy_N)
{
	double3 G_face = make_double3(0.0, 0.0, 0.0);

	// This function calculates the interface fluxes in y-direction.
	double h_S = z_S - zb_SN;
	double h_N = z_N - zb_SN;

	double u_S = 0.0;
	double v_S = 0.0;
	double u_N = 0.0;
	double v_N = 0.0;

	if ((h_S <= TOL_H) && (h_N <= TOL_H))
	{
		G_face.x = 0.0;
		G_face.y = 0.0;
		G_face.z = pow(0.5 * (z_S + z_N), 2.0) - ((zb_SN + zb_SN) * ((z_S + z_N) * 0.5));
		G_face.z *= (g * 0.5);

		return G_face;
	}

	if (h_S <= epsilon)
	{
		h_S = 0.0;
		u_S = 0.0;
		v_S = 0.0;
	}
	else
	{
		u_S = qx_S / h_S;
		v_S = qy_S / h_S;
	}

	if (h_N <= epsilon)
	{
		h_N = 0.0;
		u_N = 0.0;
		v_N = 0.0;
	}
	else
	{
		u_N = qx_N / h_N;
		v_N = qy_N / h_N;
	}

	double a_S = sqrt(g * h_S);
	double a_N = sqrt(g * h_N);

	double h_star = pow(((a_S + a_N) / 2.0 + (v_S - v_N) / 4.0), 2.0) / g;
	double v_star = (v_S + v_N) / 2.0 + a_S - a_N;
	double a_star = sqrt(g * h_star);

	double s_S = 0.0;

	if (h_S <= epsilon)
	{
		s_S = v_N - 2.0 * a_N;
	}
	else
	{
		s_S = min(v_S - a_S, v_star - a_star);
	}

	double s_N = 0.0;

	if (h_N < epsilon)
	{
		s_N = v_S + 2.0 * a_S;
	}
	else
	{
		s_N = max(v_N + a_N, v_star + a_star);
	}


	double s_M = (s_S * h_N * (v_N - s_N) - s_N * h_S * (v_S - s_S)) / (h_N * (v_N - s_N) - h_S * (v_S - s_S));

	double3 G_S, G_N;

	G_S.x = qy_S;
	G_S.y = v_S * qx_S;
	G_S.z = v_S * qy_S + 0.5 * g * (pow(z_S, 2.0) - (2.0 * zb_SN * z_S));

	G_N.x = qy_N;
	G_N.y = v_N * qx_N;
	G_N.z = v_N * qy_N + 0.5 * g * (pow(z_N, 2.0) - (2.0 * zb_SN * z_N));

	if (s_S >= 0.0)
	{
		//G_face.x = G_S.x;
		//G_face.y = G_S.y;
		//G_face.z = G_S.z;
		return G_S;
	}
	else
		if ((s_S < 0.0) && (s_N >= 0.0))
		{
			double G1_M = (s_N * G_S.x - s_S * G_N.x + s_S * s_N * (z_N - z_S)) / (s_N - s_S);
			double G3_M = (s_N * G_S.z - s_S * G_N.z + s_S * s_N * (qy_N - qy_S)) / (s_N - s_S);

			if ((s_S < 0.0) && (s_M >= 0.0))
			{
				G_face.x = G1_M;
				G_face.y = G1_M * u_S;
				G_face.z = G3_M;
			}
			else
				if ((s_M < 0.0) && (s_N >= 0.0))
				{
					G_face.x = G1_M;
					G_face.y = G1_M * u_N;
					G_face.z = G3_M;
				}
		}
		else
			if (s_N < 0.0)
			{
				//G_face.x = G_N.x;
				//G_face.y = G_N.y;
				//G_face.z = G_N.z;
				return G_N;
			}

	return G_face;
}

inline __device__ void facebound(ECellDirection ndir, double hf_p, double zf_p, double2 qf_p, double& hf_b, double& zf_b, double2& qf_b)
{
	//Inputs the available Riemann face data : hf_p,zf_p,qxf_p,qyf_p
	//Outputs the missing Riemann face data  : hf_b,zf_b,qxf_b,qyf_b


	// Topogrpahy data *!
	//double zbf_p = zf_p-hf_p;

	//* Water Depth data - pre-initialize*!  
	zf_b = zf_p;
	hf_b = hf_p;

	// Discharges data  pre-initialize*!
	qf_b.x = qf_p.x;
	qf_b.y = qf_p.y;

	//TEMP: removed most solid bodies, etc as the model domain should all be modelled!?!
}

__FLAME_GPU_FUNC__ int ProcessRKStageMessages(xmachine_memory_FloodCell* agent, xmachine_message_RKStageMessage_list* RKStageMessage_messages)
{
	if (agent->inDomain
		&& !agent->isDry)
	{


		double3 FPlus = make_double3(0.0, 0.0, 0.0);
		double3 FMinus = make_double3(0.0, 0.0, 0.0);
		double3 GPlus = make_double3(0.0, 0.0, 0.0);
		double3 GMinus = make_double3(0.0, 0.0, 0.0);

		// Initialise Eastern face
		double zbf_E = agent->zb0;// TEMP: should be  + agent->zb1x;
		double zf_E = 0.0;
		double qxf_E = 0.0;
		double qyf_E = 0.0;

		double h_L = agent->hFace_E;
		double z_L = agent->zFace_E;
		double2 q_L = make_double2(agent->qFace_E_X, agent->qFace_E_Y);
		//double3 xL_loc = make_double3( agent->xHat_X, agent->xHat_Y, agent->xHat_Z );
		//double3 yL_loc = make_double3( agent->yHat_X, agent->yHat_Y, agent->yHat_Z );

		double h_R = h_L;
		double z_R = z_L;
		double2 q_R = -q_L;
		//double3 xR_loc = xL_loc;
		//double3 yR_loc = yL_loc;

		double zb_F = 0.0;
		double z_F_L = 0.0;
		double z_F_R = 0.0;
		double qx_F_L = 0.0;
		double qx_F_R = 0.0;
		double qy_F_L = 0.0;
		double qy_F_R = 0.0;

		//Wetting and drying "depth-positivity-preserving" reconstructions
		WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, EAST, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		// Flux across the cell(ic):
		FPlus = hll_x(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		//** INNER/LOCAL flow data restrictions at the eastern face**!
		zbf_E = zb_F;
		zf_E = z_F_L;
		qxf_E = qx_F_L;
		qyf_E = qy_F_L;

		//At the Western face
		double zbf_W = agent->zb0;
		double zf_W = 0.0;
		double qxf_W = 0.0;
		double qyf_W = 0.0;

		zb_F = 0.0;
		z_F_L = 0.0;
		z_F_R = 0.0;
		qx_F_L = 0.0;
		qx_F_R = 0.0;
		qy_F_L = 0.0;
		qy_F_R = 0.0;

		h_R = agent->hFace_W;
		z_R = agent->zFace_W;
		q_R = make_double2(agent->qFace_W_X, agent->qFace_W_Y);
		//xR_loc = make_double3( agent->xHat_W_X, agent->xHat_W_Y, agent->xHat_W_Z );
		//yR_loc = make_double3( agent->yHat_W_X, agent->yHat_W_Y, agent->yHat_W_Z );

		h_L = h_R;
		z_L = z_R;
		q_L = -q_R;
		//xL_loc = xR_loc;
		//yL_loc = yR_loc;

		//* Wetting and drying "depth-positivity-preserving" reconstructions
		WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, WEST, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		// Flux across the cell(ic):
		FMinus = hll_x(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		//** INNER/LOCAL flow data restrictions at the eastern face**!
		zbf_W = zb_F;
		zf_W = z_F_R;
		qxf_W = qx_F_R;
		qyf_W = qy_F_R;

		//At the Northern face
		double zbf_N = agent->zb0;
		double zf_N = 0.0;
		double qxf_N = 0.0;
		double qyf_N = 0.0;

		zb_F = 0.0;
		z_F_L = 0.0;
		z_F_R = 0.0;
		qx_F_L = 0.0;
		qx_F_R = 0.0;
		qy_F_L = 0.0;
		qy_F_R = 0.0;

		h_L = agent->hFace_N;
		z_L = agent->zFace_N;
		q_L = make_double2(agent->qFace_N_X, agent->qFace_N_Y);
		//xL_loc = make_double3( agent->xHat_N_X, agent->xHat_N_Y, agent->xHat_N_Z );
		//yL_loc = make_double3( agent->yHat_N_X, agent->yHat_N_Y, agent->yHat_N_Z );

		h_R = h_L;
		z_R = z_L;
		q_R = -q_L;
		//xR_loc = xL_loc;
		//yR_loc = yL_loc;

		//* Wetting and drying "depth-positivity-preserving" reconstructions
		WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, NORTH, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		// Flux across the cell(ic):
		GPlus = hll_y(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		//** INNER/LOCAL flow data restrictions at the eastern face**!
		zbf_N = zb_F;
		zf_N = z_F_L;
		qxf_N = qx_F_L;
		qyf_N = qy_F_L;


		//At the Southern face
		double zbf_S = agent->zb0;
		double zf_S = 0.0;
		double qxf_S = 0.0;
		double qyf_S = 0.0;

		zb_F = 0.0;
		z_F_L = 0.0;
		z_F_R = 0.0;
		qx_F_L = 0.0;
		qx_F_R = 0.0;
		qy_F_L = 0.0;
		qy_F_R = 0.0;

		h_R = agent->hFace_S;
		z_R = agent->zFace_S;
		q_R = make_double2(agent->qFace_S_X, agent->qFace_S_Y);
		//xR_loc = make_double3( agent->xHat_S_X, agent->xHat_S_Y, agent->xHat_S_Z );
		//yR_loc = make_double3( agent->yHat_S_X, agent->yHat_S_Y, agent->yHat_S_Z );

		h_L = h_R;
		z_L = z_R;
		q_L = -q_R;
		//xL_loc = xR_loc;
		//yL_loc = yR_loc;

		//* Wetting and drying "depth-positivity-preserving" reconstructions
		WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, SOUTH, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		// Flux across the cell(ic):
		GMinus = hll_y(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

		//** INNER/LOCAL flow data restrictions at the eastern face**!
		zbf_S = zb_F;
		zf_S = z_F_R;
		qxf_S = qx_F_R;
		qyf_S = qy_F_R;

		xmachine_message_RKStageMessage* msg = get_first_RKStageMessage_message<DISCRETE_2D>(RKStageMessage_messages, agent->x, agent->y);

		while (msg)
		{
			if (msg->inDomain)
			{
				if (msg->x + 1 == agent->x
					&& agent->y == msg->y)
				{
					//Local EAST, Neighbour WEST
					double& h_R = msg->hFace_W;
					double& z_R = msg->zFace_W;
					double2 q_R = make_double2(msg->qFace_X_W, msg->qFace_Y_W);

					double h_L = agent->hFace_E;
					double z_L = agent->zFace_E;
					double2 q_L = make_double2(agent->qFace_E_X, agent->qFace_E_Y);

					double zb_F = 0.0;
					double z_F_L = 0.0;
					double z_F_R = 0.0;
					double qx_F_L = 0.0;
					double qx_F_R = 0.0;
					double qy_F_L = 0.0;
					double qy_F_R = 0.0;

					//Wetting and drying "depth-positivity-preserving" reconstructions
					WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, EAST, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

					// Flux across the cell(ic):
					FPlus = hll_x(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

					//** INNER/LOCAL flow data restrictions at the eastern face**!
					zbf_E = zb_F;
					zf_E = z_F_L;
					qxf_E = qx_F_L;
					qyf_E = qy_F_L;
				}
				else
					if (msg->x - 1 == agent->x
						&& agent->y == msg->y)
					{
						//Local WEST, Neighbour EAST
						double& h_L = msg->hFace_E;
						double& z_L = msg->zFace_E;
						double2 q_L = make_double2(msg->qFace_X_E, msg->qFace_Y_E);

						double h_R = agent->hFace_W;
						double z_R = agent->zFace_W;
						double2 q_R = make_double2(agent->qFace_W_X, agent->qFace_W_Y);

						double zb_F = 0.0;
						double z_F_L = 0.0;
						double z_F_R = 0.0;
						double qx_F_L = 0.0;
						double qx_F_R = 0.0;
						double qy_F_L = 0.0;
						double qy_F_R = 0.0;

						//* Wetting and drying "depth-positivity-preserving" reconstructions
						WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, WEST, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

						// Flux across the cell(ic):
						FMinus = hll_x(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

						//** INNER/LOCAL flow data restrictions at the eastern face**!
						zbf_W = zb_F;
						zf_W = z_F_R;
						qxf_W = qx_F_R;
						qyf_W = qy_F_R;

					}
					else
						if (msg->x == agent->x
							&& agent->y == msg->y - 1)
						{
							//Local NORTH, Neighbour SOUTH

							double& h_R = msg->hFace_S;
							double& z_R = msg->zFace_S;
							double2 q_R = make_double2(msg->qFace_X_S, msg->qFace_Y_S);
							//double3 xR_loc = make_double3( msg->xHat_X, msg->xHat_Y, msg->xHat_Z );
							//double3 yR_loc = make_double3( msg->yHat_X, msg->yHat_Y, msg->yHat_Z );

							double h_L = agent->hFace_N;
							double z_L = agent->zFace_N;
							double2 q_L = make_double2(agent->qFace_N_X, agent->qFace_N_Y);
							//double3 xL_loc = make_double3( agent->xHat_N_X, agent->xHat_N_Y, agent->xHat_N_Z );
							//double3 yL_loc = make_double3( agent->yHat_N_X, agent->yHat_N_Y, agent->yHat_N_Z );;

							double zb_F = 0.0;
							double z_F_L = 0.0;
							double z_F_R = 0.0;
							double qx_F_L = 0.0;
							double qx_F_R = 0.0;
							double qy_F_L = 0.0;
							double qy_F_R = 0.0;

							//* Wetting and drying "depth-positivity-preserving" reconstructions
							WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, NORTH, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

							// Flux across the cell(ic):
							GPlus = hll_y(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

							//printf( "GPlus: %f, %f, %f\r\n", GPlus.x, GPlus.y, GPlus.z );

							//** INNER/LOCAL flow data restrictions at the eastern face**!
							zbf_N = zb_F;
							zf_N = z_F_L;
							qxf_N = qx_F_L;
							qyf_N = qy_F_L;

						}
						else
							if (msg->x == agent->x
								&& agent->y == msg->y + 1)
							{
								//Local SOUTH, Neighbour NORTH

								double& h_L = msg->hFace_N;
								double& z_L = msg->zFace_N;
								double2 q_L = make_double2(msg->qFace_X_N, msg->qFace_Y_N);
								//double3 xL_loc = make_double3( msg->xHat_X, msg->xHat_Y, msg->xHat_Z );
								//double3 yL_loc = make_double3( msg->yHat_X, msg->yHat_Y, msg->yHat_Z );

								double h_R = agent->hFace_S;
								double z_R = agent->zFace_S;
								double2 q_R = make_double2(agent->qFace_S_X, agent->qFace_S_Y);
								//double3 xR_loc = make_double3( agent->xHat_S_X, agent->xHat_S_Y, agent->xHat_S_Z );
								//double3 yR_loc = make_double3( agent->yHat_S_X, agent->yHat_S_Y, agent->yHat_S_Z );

								double zb_F = 0.0;
								double z_F_L = 0.0;
								double z_F_R = 0.0;
								double qx_F_L = 0.0;
								double qx_F_R = 0.0;
								double qy_F_L = 0.0;
								double qy_F_R = 0.0;


								//* Wetting and drying "depth-positivity-preserving" reconstructions
								WD(h_L, h_R, z_L, z_R, q_L.x, q_R.x, q_L.y, q_R.y, SOUTH, zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

								// Flux across the cell(ic):
								GMinus = hll_y(zb_F, z_F_L, z_F_R, qx_F_L, qx_F_R, qy_F_L, qy_F_R);

								//** INNER/LOCAL flow data restrictions at the eastern face**!
								zbf_S = zb_F;
								zf_S = z_F_R;
								qxf_S = qx_F_R;
								qyf_S = qy_F_R;

							}
			}

			msg = get_next_RKStageMessage_message<DISCRETE_2D>(msg, RKStageMessage_messages);
		}

		//calc average flux...
		double zb0x_mod = (zbf_E + zbf_W) * 0.5;
		double zb0y_mod = (zbf_N + zbf_S) * 0.5;

		double zb1x_mod = (zbf_E - zbf_W) * 0.5;
		double zb1y_mod = (zbf_N - zbf_S) * 0.5;

		double z0_mod = (zf_E + zf_W + zf_N + zf_S) * 0.25;

		double z1x_mod = (zf_E - zf_W) * 0.5;
		double z1y_mod = (zf_N - zf_S) * 0.5;

		double qx0_mod = (qxf_E + qxf_W + qxf_N + qxf_S) * 0.25;
		double qx1x_mod = (qxf_E - qxf_W) * 0.5;
		double qx1y_mod = (qxf_N - qxf_S) * 0.5;

		double qy0_mod = (qyf_E + qyf_W + qyf_N + qyf_S) * 0.25;
		double qy1x_mod = (qyf_E - qyf_W) * 0.5;
		double qy1y_mod = (qyf_N - qyf_S) * 0.5;

		double3 L0_temp = make_double3(0.0, 0.0, 0.0);
		double3 L1x_temp = make_double3(0.0, 0.0, 0.0);
		double3 L1y_temp = make_double3(0.0, 0.0, 0.0);

		/*printf( "agent: %d, %d\r\nFPlus: %f, %f, %f\r\nFMinus: %f, %f, %f\r\nGPlus: %f, %f, %f\r\nGMinus: %f, %f, %f\r\n",
		agent->x, agent->y,
		FPlus.x, FPlus.y, FPlus.z,
		FMinus.x, FMinus.y, FMinus.z,
		GPlus.x, GPlus.y, GPlus.z,
		GMinus.x, GMinus.y, GMinus.z
		); */

		DG2_2D(DXL,
			DYL,
			FPlus,
			FMinus,
			GPlus,
			GMinus,
			zb0x_mod,
			zb0y_mod,
			zb1x_mod,
			zb1y_mod,
			z0_mod,
			z1x_mod,
			z1y_mod,
			qx0_mod,
			qx1x_mod,
			qx1y_mod,
			qy0_mod,
			qy1x_mod,
			qy1y_mod,
			L0_temp,
			L1x_temp,
			L1y_temp
		);

		if (!INTERMEDIATE_STAGE)
		{
			//RK1 - store values in the intermediate variables

#ifdef FIRST_ORDER

			agent->z0 = agent->z0 + TIMESTEP * L0_temp.x;
			agent->qx0 = agent->qx0 + TIMESTEP * L0_temp.y;
			agent->qy0 = agent->qy0 + TIMESTEP * L0_temp.z;



			//double3 dx = make_double3( DXL, ( agent->xHat_X-agent->xHat_Y ) / (2.0*DXL), 0 );
			//double3 dz = make_double3( 0, ( agent->xHat_Z-agent->yHat_X ) /( 2.0*DYL), DYL ); 

			double3 dx = make_double3(DXL, (agent->z0 - agent->xHat_Y) / (DXL), 0);
			double3 dz = make_double3(0, (agent->z0 - agent->yHat_X) / (DYL), DYL);

			dz = normalize(cross(dz, dx));

			//store in these to transfer to visualisation rather than alloc 3 new members of the agent
			agent->hFace_S = dz.x;
			agent->zFace_S = dz.y;
			agent->qFace_S_X = dz.z;

			// Secure zero velocities at the wet/dry front
			double h0 = agent->z0 - agent->zb0;

			if (h0 <= TOL_H)
			{
				agent->qx0 = 0.0;
				agent->qx1x = 0.0;
				agent->qx1y = 0.0;

				agent->qy0 = 0.0;
				agent->qy1x = 0.0;
				agent->qy1y = 0.0;

				//this needs to be set high, so it is ignored in the timestep reduction stage
				agent->timeStep = BIG_NUMBER;
			}
			else
			{
				double up = agent->qx0 / h0;

				double vp = agent->qy0 / h0;

				//store for timestep calc
				double xStep = CFL * DXL / (fabs(up) + sqrt(g * h0));
				double yStep = CFL * DYL / (fabs(vp) + sqrt(g * h0));

				agent->timeStep = min(xStep, yStep);

			}

#else
			agent->z0_int = agent->z0 + TIMESTEP * L0_temp.x;
			agent->qx0_int = agent->qx0 + TIMESTEP * L0_temp.y;
			agent->qy0_int = agent->qy0 + TIMESTEP * L0_temp.z;

			agent->z1x_int = agent->xHat_X + TIMESTEP * L1x_temp.x;
			agent->qx1x_int = agent->xHat_Y + TIMESTEP * L1x_temp.y;
			agent->qy1x_int = agent->xHat_Z + TIMESTEP * L1x_temp.z;

			agent->z1y_int = agent->yHat_X + TIMESTEP * L1y_temp.x;
			agent->qx1y_int = agent->yHat_Y + TIMESTEP * L1y_temp.y;
			agent->qy1y_int = agent->yHat_Z + TIMESTEP * L1y_temp.z;

			// Conserve the limited slopes for the RK2 following step
			agent->z1x = agent->xHat_X;
			agent->qx1x = agent->xHat_Y;
			agent->qy1x = agent->xHat_Z;

			agent->z1y = agent->yHat_X;
			agent->qx1y = agent->yHat_Y;
			agent->qy1y = agent->yHat_Z;


			// Secure zero velocities at the wet/dry front
			double h0 = agent->z0_int - agent->zb0;

			if (h0 <= TOL_H)
			{
				agent->qx0 = 0.0;
				agent->qx1x = 0.0;
				agent->qx1y = 0.0;

				agent->qy0 = 0.0;
				agent->qy1x = 0.0;
				agent->qy1y = 0.0;
			}
#endif
		}
		else
		{
#ifndef FIRST_ORDER

			//RK2 - update the cell data
			agent->z0 = 0.5 * (agent->z0 + agent->z0_int + TIMESTEP * L0_temp.x);
			agent->qx0 = 0.5 * (agent->qx0 + agent->qx0_int + TIMESTEP * L0_temp.y);
			agent->qy0 = 0.5 * (agent->qy0 + agent->qy0_int + TIMESTEP * L0_temp.z);

			agent->z1x = 0.5 * (agent->z1x + agent->xHat_X + TIMESTEP * L1x_temp.x);
			agent->qx1x = 0.5 * (agent->qx1x + agent->xHat_Y + TIMESTEP * L1x_temp.y);
			agent->qy1x = 0.5 * (agent->qy1x + agent->xHat_Z + TIMESTEP * L1x_temp.z);

			agent->z1y = 0.5 * (agent->z1y + agent->yHat_X + TIMESTEP * L1y_temp.x);
			agent->qx1y = 0.5 * (agent->qx1y + agent->yHat_Y + TIMESTEP * L1y_temp.y);
			agent->qy1y = 0.5 * (agent->qy1y + agent->yHat_Z + TIMESTEP * L1y_temp.z);

			double3 dx = make_double3(DXL, (2.0 * agent->xHat_X) / DXL, 0);
			double3 dz = make_double3(0, (2.0 * agent->yHat_X) / DYL, DYL);

			dz = normalize(cross(dz, dx));

			//store in these to transfer to visualisation rather than alloc 3 new members of the agent
			agent->hFace_S = dz.x;
			agent->zFace_S = dz.y;
			agent->qFace_S_X = dz.z;

			// Secure zero velocities at the wet/dry front
			double h0 = agent->z0 - agent->zb0;

			if (h0 <= TOL_H)
			{
				agent->qx0 = 0.0;
				agent->qx1x = 0.0;
				agent->qx1y = 0.0;

				agent->qy0 = 0.0;
				agent->qy1x = 0.0;
				agent->qy1y = 0.0;

				//this needs to be set high, so it is ignored in the timestep reduction stage
				agent->timeStep = BIG_NUMBER;
			}
			else
			{
				double up = agent->qx0 / h0;

				double vp = agent->qy0 / h0;

				//store for timestep calc
				double xStep = CFL * DXL / (fabs(up) + sqrt(g * h0));
				double yStep = CFL * DYL / (fabs(vp) + sqrt(g * h0));

				agent->timeStep = min(xStep, yStep);

			}

#endif
		}
	}
	else

	{
#ifdef FIRST_ORDER

		/* agent->z0  = agent->zb0;
		agent->z1x = agent->z1x;
		agent->z1y = agent->z1y;

		agent->qx0	= agent->qx0;
		agent->qx1x = agent->qx1x;
		agent->qx1y = agent->qx1y;

		agent->qy0  = agent->qy0;
		agent->qy1x = agent->qy1x;
		agent->qy1y = agent->qy1y;*/

#else
		agent->z0_int = agent->z0;
		agent->z1x_int = agent->z1x;
		agent->z1y_int = agent->z1y;

		agent->qx0_int = agent->qx0;
		agent->qx1x_int = agent->qx1x;
		agent->qx1y_int = agent->qx1y;

		agent->qy0_int = agent->qy0;
		agent->qy1x_int = agent->qy1x;
		agent->qy1y_int = agent->qy1y;
#endif

		//this needs to be set high, so it is ignored in the timestep reduction stage
		agent->timeStep = BIG_NUMBER;

		return 0;

	}

	return 0;
}

/*
inline __device__ void centinterp(	xmachine_memory_FloodCell* agent,
ECellDirection ndir,
zb0_n,
zb1x_n,
zb1y_n,
z0_n,
z1x_n,
z1y_n,
qx0_n,
qx1x_n,
qx1y_n,
qy0_n,
qy1x_n,
qy1y_n)
{
// Purpose: To interpolate flow information for a uniform grid template
// Outputs: 1) zb0_n,zb1x_n,zb1y_n : Interpolated topography data coefficients.
//          2)  z0_n, z1x_n, z1y_n : Interpolated water elevation coefficients.
//          4) qx0_n,qx1x_n,qx1y_n : Interpolated qx-discharge coefficients.
//          4) qy0_n,qy1x_n,qy1y_n : Interpolated qx-discharge coefficients.
//
//N.B. The interpolation scheme is designed for quadrilateral cells.


if(icn.EQ.0)  //neighbour out of domain
!* Flow variables are extrapolated via 'centbound'
call centbound(ic,ndir,nmod,zb0_n,zb1x_n,zb1y_n,z0_n,z1x_n,z1y_n,qx0_n,qx1x_n,qx1y_n,qy0_n,qy1x_n,qy1y_n);
return;
end if

//** First neighbour data: neighbour #1 **!
!* Topography
zb0_n1= zb0(icn1);
zb1x_n1=zb1x(icn1);
zb1y_n1=zb1y(icn1);

!* Flow
if (nmod.EQ.0) then
!
z0_n1=  z0(icn1);  z1x_n1= z1x(icn1);  z1y_n1= z1y(icn1);
qx0_n1=qx0(icn1); qx1x_n1=qx1x(icn1); qx1y_n1=qx1y(icn1);
qy0_n1=qy0(icn1); qy1x_n1=qy1x(icn1); qy1y_n1=qy1y(icn1);
!
elseif (nmod.EQ.1) then
!
z0_n1=  z0_int(icn1);  z1x_n1= z1x_int(icn1);  z1y_n1= z1y_int(icn1);
qx0_n1=qx0_int(icn1); qx1x_n1=qx1x_int(icn1); qx1y_n1=qx1y_int(icn1);
qy0_n1=qy0_int(icn1); qy1x_n1=qy1x_int(icn1); qy1y_n1=qy1y_int(icn1);
!
end if
!
!
h0_n1=z0_n1-zb0_n1;
h1x_n1=z1x_n1-zb1x_n1;
h1y_n1=z1y_n1-zb1y_n1;
!
hmin_n1 = dmin1(h0_n1-h1x_n1/dsqrt(3.0d0),h0_n1-h1y_n1/dsqrt(3.0d0),h0_n1,h0_n1+h1x_n1/dsqrt(3.0d0),h0_n1+h1y_n1/dsqrt(3.0d0));

//* For a dry neighbour, directly take the central values
if(hmin_n1.LE.TOL_H)
{
zb0_n = zb0_n1; zb1x_n = zb1x_n1; zb1y_n = zb1y_n1;
z0_n =  z0_n1;  z1x_n =  z1x_n1;  z1y_n =  z1y_n1;
qx0_n = qx0_n1; qx1x_n = qx1x_n1; qx1y_n = qx1y_n1;
qy0_n = qy0_n1; qy1x_n = qy1x_n1; qy1y_n = qy1y_n1;
}

// * Wet neighbour cases
if(levn.EQ.lev)
{
zb0_n = zb0_n1; zb1x_n = zb1x_n1; zb1y_n = zb1y_n1;
z0_n =  z0_n1;  z1x_n =  z1x_n1;  z1y_n =  z1y_n1;
qx0_n = qx0_n1; qx1x_n = qx1x_n1; qx1y_n = qx1y_n1;
qy0_n = qy0_n1; qy1x_n = qy1x_n1; qy1y_n = qy1y_n1;
return;
}
}*/

inline __device__ double3 Sb(double etta, double zprime_x, double zprime_y)
{
	// This function outputs the bed slope source terms for a specific flow data
	double3 result;

	result.x = 0.0;
	result.y = -g * etta * zprime_x;
	result.z = -g * etta * zprime_y;

	return result;

}

inline __device__ double3 Sf_explicit(double et_loc, double z_loc, double qx_loc, double qy_loc)
{
	//This function outputs the bed slope source terms for a specific flow data

	//Initialize
	double3 Sf = make_double3(0.0, 0.0, 0.0);

	// Quit if Manning == 0        
	if (GLOBAL_MANNING > 0.0)
	{
		//Local Depth
		double h_loc = et_loc - z_loc;

		if (h_loc > TOL_H)
		{
			// Local velocities    
			double u_loc = qx_loc / h_loc;
			double v_loc = qy_loc / h_loc;

			double Cf = g * pow(GLOBAL_MANNING, 2.0) / pow(h_loc, 1.0 / 3.0);

			Sf.y = -Cf * u_loc * sqrt(pow(u_loc, 2) + pow(v_loc, 2));
			Sf.z = -Cf * v_loc * sqrt(pow(u_loc, 2) + pow(v_loc, 2));
		}
	}

	return Sf;
}

inline __device__ double3 Flux_F(double et, double qx, double qy, double zb)
{
	//This function evaluates the physical flux in the x-direction

	double h = et - zb;

	double3 FF = make_double3(0.0, 0.0, 0.0);

	if (h <= TOL_H)
	{
		FF.x = 0.0;
		FF.y = (g / 2.0) * (pow(et, 2.0) - (2.0 * et * zb));
		FF.z = 0.0;
	}
	else
	{
		FF.x = qx;
		FF.y = (pow(qx, 2) / h) + ((g / 2.0) * (pow(et, 2.0) - (2.0 * et * zb)));
		FF.z = qx * qy / h;
	}

	return FF;

}

inline __device__ double3 Flux_G(double et, double qx, double qy, double zb)
{
	//This function evaluates the physical flux in the y-direction

	double h = et - zb;

	double3 GG = make_double3(0.0, 0.0, 0.0);

	if (h <= TOL_H)
	{
		GG.x = 0.0;
		GG.y = 0.0;
		GG.z = (g / 2.0) * (pow(et, 2.0) - (2.0 * et * zb));
	}
	else
	{
		GG.x = qy;
		GG.y = qx * qy / h;
		GG.z = (pow(qy, 2) / h) + ((g / 2.0) * (pow(et, 2.0) - (2.0 * et * zb)));
	}

	return GG;

}

inline __device__ void DG2_2D(double dx_loc,
	double dy_loc,
	double3& F_pos_x,
	double3& F_neg_x,
	double3& G_pos_y,
	double3& G_neg_y,
	double z0x_loc,
	double z0y_loc,
	double z1x_loc,
	double z1y_loc,
	double et0_loc,
	double et1x_loc,
	double et1y_loc,
	double qx0_loc,
	double qx1x_loc,
	double qx1y_loc,
	double qy0_loc,
	double qy1x_loc,
	double qy1y_loc,
	double3& L0_loc,
	double3& L1x_loc,
	double3& L1y_loc
)
// This function is to update, in space, the degree of freedom.   
{

	double3 Source = Sb(et0_loc, 2.0 * z1x_loc / dx_loc, 2.0 * z1y_loc / dy_loc);

	L0_loc.x = -(F_pos_x.x - F_neg_x.x) / dx_loc - (G_pos_y.x - G_neg_y.x) / dy_loc + Source.x;
	L0_loc.y = -(F_pos_x.y - F_neg_x.y) / dx_loc - (G_pos_y.y - G_neg_y.y) / dy_loc + Source.y;
	L0_loc.z = -(F_pos_x.z - F_neg_x.z) / dx_loc - (G_pos_y.z - G_neg_y.z) / dy_loc + Source.z;

#ifndef FIRST_ORDER

	double3 Source_Q1 = Sb(et0_loc - et1x_loc / sqrt3, 2.0 * z1x_loc / dx_loc, 2.0 * z1y_loc / dy_loc);
	double3 Source_Q2 = Sb(et0_loc + et1x_loc / sqrt3, 2.0 * z1x_loc / dx_loc, 2.0 * z1y_loc / dy_loc);

	double3 Flux_Q1 = Flux_F(et0_loc - et1x_loc / sqrt3, qx0_loc - qx1x_loc / sqrt3, qy0_loc - qy1x_loc / sqrt3, z0x_loc - z1x_loc / sqrt3);
	double3 Flux_Q2 = Flux_F(et0_loc + et1x_loc / sqrt3, qx0_loc + qx1x_loc / sqrt3, qy0_loc + qy1x_loc / sqrt3, z0x_loc + z1x_loc / sqrt3);


	L1x_loc.x = -(3.0 / dx_loc) * (F_pos_x.x + F_neg_x.x - Flux_Q1.x - Flux_Q2.x - ((dx_loc * sqrt3 / 6.0) * (Source_Q2.x - Source_Q1.x)));
	L1x_loc.y = -(3.0 / dx_loc) * (F_pos_x.y + F_neg_x.y - Flux_Q1.y - Flux_Q2.y - ((dx_loc * sqrt3 / 6.0) * (Source_Q2.y - Source_Q1.y)));
	L1x_loc.z = -(3.0 / dx_loc) * (F_pos_x.z + F_neg_x.z - Flux_Q1.z - Flux_Q2.z - ((dx_loc * sqrt3 / 6.0) * (Source_Q2.z - Source_Q1.z)));

	Source_Q1 = Sb(et0_loc - et1y_loc / sqrt3, 2.0 * z1x_loc / dx_loc, 2.0 * z1y_loc / dy_loc);
	Source_Q2 = Sb(et0_loc + et1y_loc / sqrt3, 2.0 * z1x_loc / dx_loc, 2.0 * z1y_loc / dy_loc);

	Flux_Q1 = Flux_G(et0_loc - et1y_loc / sqrt3, qx0_loc - qx1y_loc / sqrt3, qy0_loc - qy1y_loc / sqrt3, z0y_loc - z1y_loc / sqrt3);
	Flux_Q2 = Flux_G(et0_loc + et1y_loc / sqrt3, qx0_loc + qx1y_loc / sqrt3, qy0_loc + qy1y_loc / sqrt3, z0y_loc + z1y_loc / sqrt3);

	L1y_loc.x = -(3.0 / dy_loc) * (G_pos_y.x + G_neg_y.x - Flux_Q1.x - Flux_Q2.x - ((dy_loc * sqrt3 / 6.0) * (Source_Q2.x - Source_Q1.x)));
	L1y_loc.y = -(3.0 / dy_loc) * (G_pos_y.y + G_neg_y.y - Flux_Q1.y - Flux_Q2.y - ((dy_loc * sqrt3 / 6.0) * (Source_Q2.y - Source_Q1.y)));
	L1y_loc.z = -(3.0 / dy_loc) * (G_pos_y.z + G_neg_y.z - Flux_Q1.z - Flux_Q2.z - ((dy_loc * sqrt3 / 6.0) * (Source_Q2.z - Source_Q1.z)));

	if (fabs(z0x_loc - z0y_loc) > emsmall)
	{
		if (fabs(L0_loc.x) <= emsmall)
		{
			L1x_loc.x = 0.0;
			L1y_loc.x = 0.0;
		}

		if (fabs(L0_loc.y) <= emsmall)
		{
			L1x_loc.y = 0.0;
			L1y_loc.y = 0.0;
		}

		if (fabs(L0_loc.z) <= emsmall)
		{
			L1x_loc.z = 0.0;
			L1y_loc.z = 0.0;
		}
	}

#endif

}

#endif 
