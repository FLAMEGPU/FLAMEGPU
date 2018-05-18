# FLAME GPU Example: StableMarriage

This example demonstrates how conflict resolution can be implemented within FLAME GPU simulations.
This is a common challange in ABM where individuals compete for limited resources.

The stable marriage problem defines two sided matching as a matching between two equal sized sets of `n` men and `n`
women, where each man and woman has a personalised ranking of all member of the opposite sex. 
The goal is to find a stable solution of matches, where stability is defined as a set of matches where there are no two couples that would prefer to swap with each others partners.

For further details on this model, please see:

Richmond, Paul. "Resolving conflicts between multiple competing agents in parallel simulations." European Conference on Parallel Processing. Springer, Cham, 2014.