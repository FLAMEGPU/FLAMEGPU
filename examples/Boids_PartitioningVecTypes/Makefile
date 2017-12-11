################################################################################
#
# FLAME GPU Makefile for CUDA 7.5 and above
#
# Copyright 2017 University of Sheffield.  All rights reserved.
#
# Authors : Dr Mozhgan Kabiri Chimeh, Peter Heywood, Dr Paul Richmond
# Contact : {m.kabiri-chimeh,p.heywood,p.richmond}@sheffield.ac.uk
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
################################################################################

# The name of the example (i.e. the name of the parent directory.)
EXAMPLE := Boids_PartitioningVecTypes

# Flag indicating if a visualistion target should exist
HAS_VISUALISATION := 1

# Flag indicating if a custom visulisation is used. 0 = no, 1 = yes.
CUSTOM_VISUALISATION := 0

# Relative path to the root FLAME GPU directory. Must end with a trainling slash.
FLAMEGPU_ROOT := ../../

# Parent directory for generated executable files.
EXAMPLE_BIN_DIR := $(FLAMEGPU_ROOT)bin

# Directory for temprorary files
EXAMPLE_BUILD_DIR := ./build

# Custom gencode arguments for this model (if any), otherwise the defaults for FALME GPU are used from common.mk
# SMS ?= 30 35 37 50 60

# Include the comman makefile from the tools directory.
# If you wish to make a standalone project, you can simply replace this line with the contents of the common.mk file
include $(FLAMEGPU_ROOT)tools/common.mk
