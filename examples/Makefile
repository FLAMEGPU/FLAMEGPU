################################################################################
#
# FLAME GPU Script for CUDA 7.5
#
# Copyright 2016 University of Sheffield.  All rights reserved.
#
# Authors : Dr Mozhgan Kabiri Chimeh, Dr Paul Richmond
# Contact : {m.kabiri-chimeh,p.richmond}@sheffield.ac.uk
#
# NOTICE TO USER:
#
# University of Sheffield retain all intellectual property and
# proprietary rights in and to this software and related documentation.
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation without an express license agreement from
# University of Sheffield is strictly prohibited.
#
# For terms of licence agreement please attached licence or view licence
# on www.flamegpu.com website.
#
################################################################################
# USAGE : make help
help:
	@echo "*********************************************************************"
	@echo "*  Copyright 2016 University of Sheffield.  All rights reserved.    *"
	@echo "*********************************************************************"
	@echo "make all -> Processes XML model and builds default modes per example*"
	@echo "           ------------------------------------------------         *"
	@echo "make build -> builds all executables in either release or debug     *"
	@echo "              only use this, if you already have all the .cu files  *"
	@echo "           ------------------------------------------------         *"
	@echo "All scripts are stored in bin/x64. To run, simple exectue the script*" 
	@echo "*********************************************************************"

################################################################################
################################################################################
## FLAMEGPU EXAMPLES
#################################################################################

TARGET_ARCH ?= $(shell uname -m)

# Project folders that contain FLAMEGPU examples
PROJECTS ?= $(shell find Boids_BruteForce Boids_Partitioning CirclesBruteForce_double CirclesBruteForce_float CirclesPartitioning_double CirclesPartitioning_float GameOfLife Keratinocyte MonteCarlo_BATCH MonteCarlo_MSMPR PedestrianLOD PedestrianNavigation StableMarriage Sugarscape Template -name Makefile)

%.ph_build :
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	+@$(MAKE) -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building FLAMEGPU examples"

# Generate .cu files first.
build: $(addsuffix .ph_build,$(PROJECTS))

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
