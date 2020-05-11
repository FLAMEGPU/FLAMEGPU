import sys
sys.path.append('src')

import numpy as np

from objmodel import *
from ParticleGenerator import *

def printUsageInstructions():
	print "\n\nUsage: gen_initial_conditions.py filepath static"
	print "\tfilepath: the path to a triangulated obj model, whose vertices all lie within a 1.0 width cube centered at the origin"
	print "\tstatic: True or False, representing whether the model should be generated as static particles (i.e. a solid) or dynamic fluid particles\n"
	print "Multiple files can be processed at once by submitting more than one filepath static pair sequentially, e.g."
	print "\tgen_initial_conditions.py FLAME.obj False GPU.obj True"
	print "which will generate the provided 0.xml initial states file."
		
# Not enough arguments provided
if len(sys.argv) < 3:
	printUsageInstructions()
else:
	print sys.argv
	# Check we have valid number of args
	if len(sys.argv)%2 != 1:
		printUsageInstructions()
	else:
		# Safe to run normally - correct number of args given
		print "Starting..."
		numFiles = (len(sys.argv) - 1) / 2
		
		pg = ParticleGenerator()
		
		for item in range(numFiles):
			filePath = sys.argv[2*item + 1]
			print "Beginning processing of model " + filePath
			model = loadObjectModel(filePath)
			isStatic = sys.argv[2*item + 2] == 'True'
			pg.generateParticles(model, isStatic)
		pg.save()



