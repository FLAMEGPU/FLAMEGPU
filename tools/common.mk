################################################################################
# FLAME GPU common Makefile rules for CUDA 7.5 and above
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
#
# This file contains common variables, rules and targets for compilation of
# the FLAME GPU example projects, but this depnds upon several variables being
# defined and set prior to the inclusion of this file.
# Additionally, several variables can be over-ridden by use, indicated by ?=
################################################################################

# Define the default values for SMS depedning on cuda version.
DEFAULT_SMS_CUDA_9 := 30 35 37 50 60 70
DEFAULT_SMS_CUDA_8 := 30 35 37 50 60
DEFAULT_SMS_CUDA_6 := 30 35 37 50

# Use .o as the default object extions.
OBJ_EXT := .o
# Set the default binary extension - no extension.
BIN_EXT :=

# OS Specific directoryes and file extensions.
ifeq ($(OS),Windows_NT)
	OS_BIN_DIR := x64
	OS_BUILD_DIR := x64
	OS_LIB_DIR :=
	# Override default values for windows.
	BIN_EXT := .exe
	OBJ_EXT := .obj
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OS_BIN_DIR := linux-x64
		OS_BUILD_DIR := linux-x64
		OS_LIB_DIR := x86_64-linux-gnu
	endif
endif

# OS specific output location for executable files
BIN_DIR := $(EXAMPLE_BIN_DIR)/$(OS_BIN_DIR)
# Temporary directory for building
BUILD_DIR := $(EXAMPLE_BUILD_DIR)/$(OS_BUILD_DIR)

# Path to FLAME GPU include directory
INCLUDE_DIR := $(FLAMEGPU_ROOT)include
# Path to FLAME GPU Lib directory (OS specific)
LIB_DIR := $(FLAMEGPU_ROOT)lib/$(OS_LIB_DIR)

# Path to the FLAME GPU Templates directory
TEMPLATES_DIR := $(FLAMEGPU_ROOT)FLAMEGPU/templates
# Path to FLAME GPU XSD Schema directory
XSD_SCHEMA_DIR := $(FLAMEGPU_ROOT)FLAMEGPU/schemas

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

# Executable used for compilation
NVCC := nvcc

# Get the NVCC verison
NVCC_MAJOR = $(shell ""$(NVCC)"" --version | sed -n -r 's/.*(V([0-9]+).([0-9]+).([0-9]+))/\2/p')
# NVCC_MINOR = $(shell ""$(NVCC)"" --version | sed -n -r 's/.*(V([0-9]+).([0-9]+).([0-9]+))/\3/p')
# NVCC_PATCH = $(shell ""$(NVCC)"" --version | sed -n -r 's/.*(V([0-9]+).([0-9]+).([0-9]+))/\4/p')
NVCC_GE_9_0 = $(shell [ $(NVCC_MAJOR) -ge 9 ] && echo true)
NVCC_GE_8_0 = $(shell [ $(NVCC_MAJOR) -ge 8 ] && echo true)


# For the appropriate CUDA version, assign default SMS if required.
ifeq ($(NVCC_GE_9_0),true)
SMS ?= $(DEFAULT_SMS_CUDA_9)
else ifeq ($(NVCC_GE_8_0),true)
SMS ?= $(DEFAULT_SMS_CUDA_8)
else
SMS ?= $(DEFAULT_SMS_CUDA_6)
endif

# Flags used for compilation.
# NVCC compiler flags
NVCCFLAGS   := -m64 -lineinfo
# Host compiler flags (gcc/cicc etc), passed to nvcc using -Xcompiler
CCFLAGS     := 
# NVCC linker flags
NVCCLDFLAGS :=
# Host linker flags (ld etc), passed to nvcc using -Xlinker
LDFLAGS     :=

# By default, the mode type is relase and build director
Mode_TYPE := Release

# Debug specific build flags and variables
ifeq ($(debug),1)
	  NVCCFLAGS += -g -G -DDEBIG -D_DEBUG
	  Mode_TYPE := Debug
endif

# Compute the actual build directory, by appending the mode type.
BUILD_DIR := $(BUILD_DIR)/$(Mode_TYPE)

# Os specigic tools, compiler and linker flags
ifeq ($(OS),Windows_NT)
	# Explicitly use the included xstl processor on windows.
	XSLTPROC := $(FLAMEGPU_ROOT)tools/XSLTProcessor.exe
	# by default, do not use a linter on windows
	XMMLLINT :=
	# Set windows specific host compiler flags
	CCFLAGS := -W3
	# Pass directory to lib files
	NVCCLDFLAGS += -L "$(LIB_DIR)"
	# Specify windows specific shared libraries to link against.
	LINK_ARCHIVES_VISUALISATION := -lglut32 -lglut64 -lglew64
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		# Verify that xsltlint and xsltproc are on the path.
		XSLTPROC := $(shell command -v xsltproc 2> /dev/null)
		XMLLINT := $(shell command -v xmllint 2> /dev/null)
		# Pass specific nvcc flags for linux
		NVCCFLAGS += -std=c++11
		CCFLAGS += -Wall
		# Pass directory to lib files
		NVCCLDFLAGS += -L$(LIB_DIR)
		# Path to the library shard object files relative to the final bin directory location.
		#@todo - this needs to be a path from the user specified bin directory, to the LIB directory for the OS.
		LD_RUN_PATH := LD_RUN_PATH='LD_RUN_PATH=$$ORIGIN/../$(LIB_DIR)'
		# Specify linux specific shared libraries to link against
		LINK_ARCHIVES_VISUALISATION := -lglut -lGLEW -lGLU -lGL
	endif
endif

# Build a single variable contianing all compiler flags, prefixed as appropriate
ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

# Build a single variable contianing all linker flags, prefixed as appropriate
ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Build the actual include compiler arguments, for both console and visualisation targets
CONSOLE_INCLUDES := $(addprefix -I ,$(INCLUDE_DIRS))
VISUALISATION_INCLUDES := $(addprefix -I ,$(INCLUDE_DIRS_VISUALISATION))

# Generate the full path to target executable files
TARGET_VISUALISATION := $(BIN_DIR)/$(Mode_TYPE)_Visualisation/$(EXAMPLE)$(BIN_EXT)
TARGET_CONSOLE := $(BIN_DIR)/$(Mode_TYPE)_Console/$(EXAMPLE)$(BIN_EXT)

# Dependancies for the targets
CONSOLE_DEPENDANCIES := $(BUILD_DIR)/io.cu$(OBJ_EXT) $(BUILD_DIR)/simulation.cu$(OBJ_EXT) $(BUILD_DIR)/main_console.cu$(OBJ_EXT)

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
VISUALISATION_DEPENDANCIES := $(BUILD_DIR)/io.cu$(OBJ_EXT) $(BUILD_DIR)/simulation.cu$(OBJ_EXT) $(BUILD_DIR)/main_visualisation.cu$(OBJ_EXT) $(BUILD_DIR)/visualisation.cu$(OBJ_EXT)
# Else this is a custom visualisation
else
# List of Dynamic FLAME GPU files
XSLT_OUTPUT_FILES := \
	$(SRC_DYNAMIC)/header.h \
	$(SRC_DYNAMIC)/FLAMEGPU_kernals.cu\
	$(SRC_DYNAMIC)/io.cu \
	$(SRC_DYNAMIC)/simulation.cu \
	$(SRC_DYNAMIC)/main.cu

# Find all the c, cpp and cu files in the visusalistion direcotry
# @todo support manually specified custom visualisation files / directoeies being passed in.
CUSTOM_VISUALISATION_C_FILES ?= $(wildcard $(SRC_VISUALISATION)/*.c)
CUSTOM_VISUALISATION_C_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CUSTOM_VISUALISATION_C_FILES:.c=$(OBJ_EXT))))
CUSTOM_VISUALISATION_CPP_FILES ?= $(wildcard $(SRC_VISUALISATION)/*.cpp)
CUSTOM_VISUALISATION_CPP_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CUSTOM_VISUALISATION_CPP_FILES:.cpp=$(OBJ_EXT))))
CUSTOM_VISUALISATION_CU_FILES ?= $(wildcard $(SRC_VISUALISATION)/*.cu)
CUSTOM_VISUALISATION_CU_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CUSTOM_VISUALISATION_CU_FILES:.cu=.cu$(OBJ_EXT))))

# Build the list of custom visualisation object files
CUSTOM_VISUALISATION_OBJECTS := $(CUSTOM_VISUALISATION_C_OBJECTS) $(CUSTOM_VISUALISATION_CPP_OBJECTS) $(CUSTOM_VISUALISATION_CU_OBJECTS)

# Dependancies for custom visusliastion
VISUALISATION_DEPENDANCIES := $(BUILD_DIR)/io.cu$(OBJ_EXT) $(BUILD_DIR)/simulation.cu$(OBJ_EXT) $(BUILD_DIR)/main_visualisation.cu$(OBJ_EXT) $(CUSTOM_VISUALISATION_OBJECTS)
endif

XLST_FUNCTIONS_C := $(SRC_DYNAMIC)/functions.c

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
.PHONY: help all validate xslt visualisation console clean clobber makedirs functions.c

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
	@$(XMLLINT) --noout $(XML_MODEL_FILE) --schema $(XSD_SCHEMA_DIR)/XMMLGPU.xsd
endif

# Target to use xsltproc to generate all dynamic files.
xslt: validate $(XSLT_OUTPUT_FILES)

# Create functions.c prototypes as a differently named file, to avoid overwriting user code.
functions.c: validate $(XLST_FUNCTIONS_C)

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
ifeq ($(OS),Windows_NT)
	$(XSLTPROC) $(XML_MODEL_FILE) $< $@
else
	$(XSLTPROC) $< $(XML_MODEL_FILE) > $@
endif

# Rule to create *.cu files in the dynamic folder, as requested by build dependencies.
$(SRC_DYNAMIC)/%.cu: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE)
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif
ifeq ($(OS),Windows_NT)
	$(XSLTPROC) $(XML_MODEL_FILE) $< $@
else
	$(XSLTPROC) $< $(XML_MODEL_FILE) > $@
endif

# Rule to create functsion.c file in the dynamic folder.
$(SRC_DYNAMIC)/%.c: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE)
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif
ifeq ($(OS),Windows_NT)
	$(XSLTPROC) $(XML_MODEL_FILE) $< $@
else
	$(XSLTPROC) $< $(XML_MODEL_FILE) > $@
endif

# Explicit rules for object files in the dynamic folder.
# Explicit rules are used (rather than generic) as the dependancis are non-trivial (currently)
$(BUILD_DIR)/io.cu$(OBJ_EXT): $(SRC_DYNAMIC)/io.cu $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BUILD_DIR)/simulation.cu$(OBJ_EXT): $(SRC_DYNAMIC)/simulation.cu $(SRC_DYNAMIC)/FLAMEGPU_kernals.cu $(FUNCTIONS_FILES) $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BUILD_DIR)/main_console.cu$(OBJ_EXT): $(SRC_DYNAMIC)/main.cu $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ifeq ($(HAS_VISUALISATION), 1)
# Visualisation specific dynamic file rules.
$(BUILD_DIR)/main_visualisation.cu$(OBJ_EXT): $(SRC_DYNAMIC)/main.cu $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Only build visualistion$(OBJ_EXT) if not a custom visuations
ifeq ($(CUSTOM_VISUALISATION), 0)
$(BUILD_DIR)/visualisation.cu$(OBJ_EXT): $(SRC_DYNAMIC)/visualisation.cu $(SRC_VISUALISATION)/visualisation.h $(SRC_DYNAMIC)/header.h
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
else
# Rules for custom visualistion compilation of c files
$(BUILD_DIR)/%$(OBJ_EXT): $(SRC_VISUALISATION)/%.c
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Rules for custom visualistion compilation of cpp files
$(BUILD_DIR)/%$(OBJ_EXT): $(SRC_VISUALISATION)/%.cpp
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Rules for custom visualistion compilation of cu files
$(BUILD_DIR)/%.cu$(OBJ_EXT): $(SRC_VISUALISATION)/%.cu
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
endif

# Rule to create the visualisation binary by linking the dependant object files.
$(TARGET_VISUALISATION): $(VISUALISATION_DEPENDANCIES)
	$(EXEC) $(LD_RUN_PATH) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LINK_ARCHIVES_VISUALISATION) -o $@ $+
endif

# Rule to create the console binary by linking the dependant object files.
$(TARGET_CONSOLE): $(CONSOLE_DEPENDANCIES)
	$(EXEC) $(LD_RUN_PATH) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

# Clean object files, but do not regenerate xslt. `|| true` is used to support the case where dirs do not exist.
clean:
	find $(EXAMPLE_BUILD_DIR)/ -name '*$(OBJ_EXT)' -delete || true
	find $(EXAMPLE_BUILD_DIR)/ -name '*.cu$(OBJ_EXT)' -delete || true

# Clobber all temporary files, including dynamic files and the target executable. `|| true` is used to support the case where dirs do not exist.
clobber: clean
	find $(SRC_DYNAMIC)/ -name '*.c' -delete || true
	find $(SRC_DYNAMIC)/ -name '*.cu' -delete || true
	find $(SRC_DYNAMIC)/ -name '*.h' -delete || true
	find $(BIN_DIR)/ -name '$(EXAMPLE)$(BIN_EXT)' -delete || true

# Create any required directories.
makedirs:
	@mkdir -p $(BIN_DIR)/$(Mode_TYPE)_Console
	@mkdir -p $(BIN_DIR)/$(Mode_TYPE)_Visualisation
	@mkdir -p $(BUILD_DIR)

# Help target, printing usage information to stdout
help:
	@echo "************************************************************************"
	@echo "* Copyright 2017 University of Sheffield.  All rights reserved.        *"
	@echo "************************************************************************"
	@echo " Usage: "
	@echo "   make <target> <arguments>"
	@echo ""
	@echo " Targets:"
	@echo ""
	@echo "   help          Shows this help documentation"
	@echo "   all           Validates XML, generates dynamic files"
	@echo "                   & builds console and visualisation modes"
	@echo "   validate      Validates XMLModelFile using 'xmllint' (if installed)"
	@echo "                   install via '"apt install libxml2-utils"'"
	@echo "   xslt          Validates XML and generates dynamic files"
	@echo "                    depends on 'xsltproc'"
	@echo "   console       Builds console mode exectuable"
	@echo "   visualistion  Builds visualisation mode executable, if it exists"
	@echo "   clean         Deletes generated object files"
	@echo "   clobber       Deletes all generated files including executables"
	@echo "   functions.c   Generates functions.c in the dynamic folder using xslt"
	@echo "                   This file is for reference only. Not used in build."
	@echo ""
	@echo " Arguments:"
	@echo ""
	@echo "   debug=<arg>   Builds target in 'Release' or 'Debug' mode"
	@echo "                   0 : Release (Default)"
	@echo "                   1 : Debug"
	@echo "                   I.e. 'make console debug=1'"
	@echo "   SMS=<arg>     Builds target for the specified CUDA architectures"
	@echo "                   I.e. 'make console SMS=60 61'"
	@echo "                   Defaults to: '$(SMS)'"
	@echo "                   'make clean' is required prior to using new values"
	@echo "************************************************************************"
