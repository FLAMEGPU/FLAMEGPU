################################################################################
#
# FLAME GPU Script for CUDA 7.5, CUDA 8.0
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


################################################################################
# USAGE : make help
################################################################################

# The name of the example (i.e. the name of the parent directory.)
EXAMPLE := GameOfLife

# Flag indicating if a visualistion target should exist
HAS_VISUALISATION := 1
# Flag indicating if a custom visulisation is used. 0 = no, 1 = yes.
CUSTOM_VISUALISATION := 0

# Parent directory for generated executable files.
EXAMPLE_BIN_DIR := ../../bin

# Directory for temprorary files
EXAMPLE_BUILD_DIR := ./build

# Gencode arguments
SMS ?= 30 35 37 50 60

################################################################################


# OS Specific settings.
ifeq ($(OS),Windows_NT)
	OS_BIN_DIR := x64
	OS_BUILD_DIR := x64
	OS_LIB_DIR :=
	OS_BIN_EXT := .exe
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OS_BIN_DIR := linux-x64
		OS_BUILD_DIR := linux-x64
		OS_LIB_DIR := x86_64-linux-gnu
		OS_BIN_EXT :=
	endif
endif

# OS specific output location for executable files
BIN_DIR := $(EXAMPLE_BIN_DIR)/$(OS_BIN_DIR)
# Temporary directory for building
BUILD_DIR := $(EXAMPLE_BUILD_DIR)/$(OS_BUILD_DIR)

# Path to FLAME GPU include directory
INCLUDE_DIR := ../../include
# Path to FLAME GPU Lib directory (OS specific)
LIB_DIR := ../../lib/$(OS_LIB_DIR)
# Path to the library shard object files relative to the final bin directory location.
LD_RUN_PATH := '$$ORIGIN/../$(LIB_DIR)'

# Path to the FLAME GPU Templates directory
TEMPLATES_DIR := ../../FLAMEGPU/templates
# Path to FLAME GPU XSD Schema directory
XSD_SCHEMA_DIR := ../../FLAMEGPU/schemas

# Paths to various source file directories for this example
SRC_MODEL := src/model
SRC_VISUALISATION := src/visualisation
SRC_DYNAMIC := src/dynamic

# Full path to the XMLModelFile xml file for this model.
XML_MODEL_FILE:=$(SRC_MODEL)/XMLModelFile.xml

# Function Files(s) used within this model.
FUNCTIONS_FILES:= \
	$(SRC_MODEL)/functions.c

# Include directories for console and visualisation builds
# If a custom visualisation is used, this may also need to include $SRC_VISUALISATION because functions.c may include it.
INCLUDE_DIRS := \
	$(INCLUDE_DIR) \
	$(SRC_MODEL) \
	$(SRC_DYNAMIC) \
	$(SRC_VISUALISATION) 

# Include directories specific to visualisation builds
INCLUDE_DIRS_VISUALISATION := \
	$(INCLUDE_DIR)/GL


# Executables used for compilation
HOST_COMPILER ?= g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)

# Flags used for compilation.
NVCCFLAGS   := -m64 -lineinfo 
CCFLAGS     := -Wall
LDFLAGS     := -L$(LIB_DIR)

# Debug specific build flags
ifeq ($(debug),1)
	  NVCCFLAGS += -g -G -DDEBIG -D_DEBUG
	  Mode_TYPE := Debug
	  BUILD_DIR := $(BUILD_DIR)/$(Mode_TYPE)
else
	  Mode_TYPE := Release
	  BUILD_DIR := $(BUILD_DIR)/$(Mode_TYPE)
endif

# All flags to be passed to the compiler
ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

# All flags to be passed to the linker
ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Build the actual include compiler arguments.
# @todo - seperate vis and console 
CONSOLE_INCLUDES := $(addprefix -I ,$(INCLUDE_DIRS)) 
VISUALISATION_INCLUDES := $(addprefix -I ,$(INCLUDE_DIRS_VISUALISATION))

# List of linker archive commands for visualisation
LINK_ARCHIVES_VISUALISATION := -lglut -lGLEW -lGLU -lGL

# Verify that xsltlint and xsltproc are on the path.
XSLTPROC := $(shell command -v xsltproc 2> /dev/null)
XMLLINT := $(shell command -v xmllint 2> /dev/null)

# Generate the full path to target executable files
TARGET_VISUALISATION := $(BIN_DIR)/$(Mode_TYPE)_Visualisation/$(EXAMPLE)$(OS_BIN_EXT)
TARGET_CONSOLE := $(BIN_DIR)/$(Mode_TYPE)_Console/$(EXAMPLE)$(OS_BIN_EXT)

# Dependancies for the targets
CONSOLE_DEPENDANCIES := $(BUILD_DIR)/io.cu.o $(BUILD_DIR)/simulation.cu.o $(BUILD_DIR)/main_console.cu.o 

# If this is not a custom visualistaion
ifeq ($(CUSTOM_VISUALISATION), 0)
# List of Dynamic FLAME GPU files
XSLT_OUTPUT_FILES := \
	$(SRC_DYNAMIC)/header.h \
	$(SRC_DYNAMIC)/FLAMEGPU_kernals.cu\
	$(SRC_DYNAMIC)/io.cu \
	$(SRC_DYNAMIC)/simulation.cu \
	$(SRC_DYNAMIC)/main.cu \
	$(SRC_DYNAMIC)/visualisation.cu

# Dependancies for default visusliastion
VISUALISATION_DEPENDANCIES := $(BUILD_DIR)/io.cu.o $(BUILD_DIR)/simulation.cu.o $(BUILD_DIR)/main_visualisation.cu.o $(BUILD_DIR)/visualisation.cu.o
# Else this is a custom visualisation
else
# List of Dynamic FLAME GPU files
XSLT_OUTPUT_FILES := \
	$(SRC_DYNAMIC)/header.h \
	$(SRC_DYNAMIC)/FLAMEGPU_kernals.cu\
	$(SRC_DYNAMIC)/io.cu \
	$(SRC_DYNAMIC)/simulation.cu \
	$(SRC_DYNAMIC)/main.cu 
# Find all the cpp and cu files in the visusalistion direcotry

CUSTOM_VISUALISATION_C_FILES := $(wildcard $(SRC_VISUALISATION)/*.c)
CUSTOM_VISUALISATION_C_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CUSTOM_VISUALISATION_C_FILES:.c=.o)))
CUSTOM_VISUALISATION_CPP_FILES := $(wildcard $(SRC_VISUALISATION)/*.cpp)
CUSTOM_VISUALISATION_CPP_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CUSTOM_VISUALISATION_CPP_FILES:.cpp=.o)))
CUSTOM_VISUALISATION_CU_FILES := $(wildcard $(SRC_VISUALISATION)/*.cu)
CUSTOM_VISUALISATION_CU_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CUSTOM_VISUALISATION_CU_FILES:.cu=.cu.o)))

CUSTOM_VISUALISATION_OBJECTS := $(CUSTOM_VISUALISATION_C_OBJECTS) $(CUSTOM_VISUALISATION_CPP_OBJECTS) $(CUSTOM_VISUALISATION_CU_OBJECTS)

# Dependancies for custom visusliastion
VISUALISATION_DEPENDANCIES := $(BUILD_DIR)/io.cu.o $(BUILD_DIR)/simulation.cu.o $(BUILD_DIR)/main_visualisation.cu.o $(CUSTOM_VISUALISATION_OBJECTS)
endif

################################################################################

# Verify that atleast one SM value has been specified.
ifeq ($(SMS),)
$(error "Error - no SM architectures have been specified. Aborting.")
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

################################################################################

# Mark several targets as PHONY, i.e. they do not create a file of the target name
.PHONY: help all validate xslt visualisation console clean clobber makedirs

# When make all is called, the model is validated, all xslt is generated and then both console and visualisation targets are built
ifeq ($(HAS_VISUALISATION), 1)
all: validate xslt console visualisation
else
all: validate xslt console
endif

# Use XMLLint to validate the model (if installed and accessible on the path.)
validate: $(XML_MODEL_FILE) $(XSD_SCHEMA_DIR)/XMMLGPU.xsd
ifndef XMLLINT
	$(warning "Warning: xmllint is not available, please install libxml2-utils to enable validation")
else
	@xmllint --noout $(XML_MODEL_FILE) --schema $(XSD_SCHEMA_DIR)/XMMLGPU.xsd 
endif

# Target to use xsltproc to generate all dynamic files.
xslt: validate $(XSLT_OUTPUT_FILES)

# Create the console version of this application, inlcuding directory creation and validation of the XML Model
console: makedirs validate $(TARGET_CONSOLE)

# Create the visualisation version of this application, inlcuding directory creation and validation of the XML Model
ifeq ($(HAS_VISUALISATION), 1)
visualisation: makedirs validate $(TARGET_VISUALISATION)
endif

# Rule to create header.h from XSLT. Depends upon both header.xslt and the XML file, so if either is changed a re-build will occur.
$(SRC_DYNAMIC)/%.h: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE)
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif 
	xsltproc $< $(XML_MODEL_FILE) > $@

# Rule to create *.cu files in the dynamic folder, as requested by build dependencies.
$(SRC_DYNAMIC)/%.cu: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE)
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif 
	xsltproc $< $(XML_MODEL_FILE) > $@

# Explicit rules for object files in the dynamic folder.
# Explicit rules are used (rather than generic) as the dependancis are non-trivial (currently)
$(BUILD_DIR)/io.cu.o: $(SRC_DYNAMIC)/io.cu $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BUILD_DIR)/simulation.cu.o: $(SRC_DYNAMIC)/simulation.cu $(SRC_DYNAMIC)/FLAMEGPU_kernals.cu $(FUNCTIONS_FILES) $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BUILD_DIR)/main_console.cu.o: $(SRC_DYNAMIC)/main.cu $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ifeq ($(HAS_VISUALISATION), 1)
# Visualisation specific dynamic file rules.
$(BUILD_DIR)/main_visualisation.cu.o: $(SRC_DYNAMIC)/main.cu $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Only build visualistion.o if not a custom visuations
ifeq ($(CUSTOM_VISUALISATION), 0)
$(BUILD_DIR)/visualisation.cu.o: $(SRC_DYNAMIC)/visualisation.cu $(SRC_VISUALISATION)/visualisation.h $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
else
# Rules for custom visualistion compilation of c files
$(BUILD_DIR)/%.o: $(SRC_VISUALISATION)/%.c
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Rules for custom visualistion compilation of cpp files
$(BUILD_DIR)/%.o: $(SRC_VISUALISATION)/%.cpp
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Rules for custom visualistion compilation of cu files
$(BUILD_DIR)/%.cu.o: $(SRC_VISUALISATION)/%.cu
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
endif

# Rule to create the visualisation binary by linking the dependant object files.
$(TARGET_VISUALISATION): $(VISUALISATION_DEPENDANCIES)
	$(EXEC) LD_RUN_PATH=$(LD_RUN_PATH) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LINK_ARCHIVES_VISUALISATION) -o $@ $+
endif

# Rule to create the console binary by linking the dependant object files.
$(TARGET_CONSOLE): $(CONSOLE_DEPENDANCIES)
	$(EXEC) LD_RUN_PATH=$(LD_RUN_PATH) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

# Clean object files, but do not regenerate xslt. `|| true` is used to support the case where dirs do not exist.
clean:
	find $(EXAMPLE_BUILD_DIR)/ -name '*.o' -delete || true 
	find $(EXAMPLE_BUILD_DIR)/ -name '*.cu.o' -delete || true 

# Clobber all temporary files, including dynamic files and the target executable. `|| true` is used to support the case where dirs do not exist.
clobber: clean 
	find $(SRC_DYNAMIC)/ -name '*.cu' -delete || true 
	find $(SRC_DYNAMIC)/ -name '*.h' -delete || true 
	find $(BIN_DIR)/ -name '$(EXAMPLE)$(OS_BIN_EXT)' -delete || true 

# Create any required directories.
makedirs:
	@mkdir -p $(BIN_DIR)/$(Mode_TYPE)_Console
	@mkdir -p $(BIN_DIR)/$(Mode_TYPE)_Visualisation
	@mkdir -p $(BUILD_DIR)

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
	@echo "    all          -> Validates XML, generates dynamic files" 
	@echo "                      & builds console and visualisation modes"  
	@echo "    validate     -> Validates XMLModelFile using 'xmllint' (if installed)"
	@echo "                      install via '"apt install libxml2-utils"'"
	@echo "    xslt         -> Validates XML and generates dynamic files"
	@echo "                      depends on 'xsltproc'"
	@echo "    console      -> Builds console mode exectuable"
	@echo "    visualistion -> Builds visualisation mode executable, if it exists"
	@echo "    clean        -> Deletes generated object files"
	@echo "    clobber      -> Deletes all generated files including executables"
	@echo ""
	@echo "  Arguments":
	@echo ""
	@echo "    debug=<arg> -> Builds target in 'Release' or 'Debug' mode"
	@echo "                     0 : Release (Default)"
	@echo "                     1 : Debug"
	@echo "                     I.e. 'make console debug=1'"
	@echo "    SMS=<arg>   -> Builds target for the specified CUDA architectures"
	@echo "                     I.e. 'make console SMS=60 61'"
	@echo "                     Defaults to: '$(SMS)'"
	@echo "                     Note: 'make clean' is required prior to using new values"
	@echo "************************************************************************"
