The social particle swarm (sps) model is a self-driven particle systems used for the simulation of continuous dynamics of social systems in an abstract space. The model originally assumes agents interacting in a 2D Cartesian space which is an abstraction of the social or psychological space.



In our model, we considered interacting agents in a 3D abstract social space, each with an initial position $(\overrightarrow{x_i})$ and velocity $(\overrightarrow{v_i})$, and identical interaction ranges $(R)$. Through kinematics, agents express their social inclination by getting closer to groups offering social gain, or getting away from less fortunate ones, meaning that their positions represent their social relationship against others. Social gain is expressed with a prisoner's dilemma game for $(n)$ neighbors, using a generalized tit-for-tat strategy, each agent calculates the score she receives from her neighbors, weighted by the distance between them as follows: $total\_score_i = \sum_{j \in Neighbors_i}\frac{pay_{(i,j)}}{||\overrightarrow{d}_{i,j}||}$, where $pay_{(i,j)}$ is the payoff received from the game with her neighbors, and \overrightarrow{{d}_{i,j}} is the distance. 

Agents run three processes, (1)cooperators estimation, (2)strategy update and (3) social relationship change. The first one entails estimating the ratio of neighboring cooperators using a formula inspired partly from models of predictive coding in cognitive neuroscience, which depends on an information update rate about the neighbors $(I)$. The formula is as follows: $e(t+1) = e(t) + ( r(t)-e(t) \times I )$, where $e(t)$ is the estimate at instant $(t)$ and $r(t)$ is the actual ratio of cooperators at $(t)$, $I$ is the update rate $(I \in [0, 1])$. If $I$ is large, agents quickly minimize the prediction error about their neighbors, and conversely. In strategy update, $e$ is compared against a cooperation threshold $thresh$, if $e > thresh$ then the agent $i$ sets her strategy to cooperation, and to defection otherwise. The last process is expressed through steering forces, where an agent steers towards (or away form) the center of mass of her neighbors depending the sign of the $total\_score_i$. The generated steering force vector is obtained using Reynolds steering formula.

Simulation videos can be checked on:


It shows interesting phenomena of spherical cooperative clusters followed by their explosion due to overpopulation by defectors, and when the update rate $(I)$ was minimal $(0.01)$, cooperative clusters were few but consistent over time(has longer lifespan). 


Details on the model can be found in: 
