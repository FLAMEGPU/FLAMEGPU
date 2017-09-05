################################################################################
#
# FLAME GPU Makefile for CUDA 7.5 and above
#
# Copyright 2017 University of Sheffield.  All rights reserved.
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
	@echo "************************************************************************"
	@echo "* Copyright 2017 University of Sheffield.  All rights reserved.        *"
	@echo "************************************************************************"
	@echo "  Usage: "
	@echo "    make <target> <arguments>"
	@echo ""
	@echo "  Targets:"  
	@echo ""  
	@echo "    help         -> Shows this help documentation" 
	@echo "    all          -> For each example, validates XML, generates dynamic files" 
	@echo "                      & builds console and visualisation modes"  
	@echo "    clean        -> Deletes generated object files"
	@echo "    clobber      -> Deletes all generated files including executables"
	@echo ""
	@echo "  Arguments":
	@echo ""
	@echo "    debug=<arg> -> Builds target in 'Release' or 'Debug' mode"
	@echo "                     0 : Release (Default)"
	@echo "                     1 : Debug"
	@echo "                     I.e. 'make all debug=1'"
	@echo "    SMS=<arg>   -> Builds target for the specified CUDA architectures"
	@echo "                     I.e. 'make all SMS=60 61'"
	@echo "                     Defaults to: '$(SMS)'"
	@echo "                     Note: 'make clean' is required prior to using new values"
	@echo "************************************************************************"


################################################################################
################################################################################
## FLAMEGPU EXAMPLES
#################################################################################

TARGET_ARCH ?= $(shell uname -m)

# Project folders that contain FLAMEGPU examples
PROJECTS ?= $(shell find ./ -maxdepth 2 -mindepth 2  -name "Makefile")

.PHONY: help all tidy clean clobber %.ph_all %.ph_clean %.ph_clobber

%.ph_all :
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	+@$(MAKE) -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_all,$(PROJECTS))
	@echo "Finished building FLAMEGPU examples"


tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
