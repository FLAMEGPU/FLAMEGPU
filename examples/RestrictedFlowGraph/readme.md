# FLAME GPU Example: RestrictedFlowGraph

An example model showcasing the use of Edge based communication, and the use of a graph data structure. 

Agents travel along the edges of a predefined graph, at a given speed. 

When the end of a graph is reached, agents will attempt to move to a new edge, if there is available capacity as defined by the graph. 

If there is not sufficient capacity the agent will wait at the end of the current edge.

If there is only capacity for some of the agents which wish to make a transition to a given edge, conflict is resolved based on ID - the agents with the lowest id's will be allowed to proceed.


A python script is included to generate a grid based network. See `python tools/network-generator.py -h` for more information.

