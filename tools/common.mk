################################################################################
# FLAME GPU common Makefile rules
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
DEFAULT_SMS_CUDA_11 := 52 60 70 75 80
DEFAULT_SMS_CUDA_10 := 30 35 50 60 70 75
DEFAULT_SMS_CUDA_9  := 30 35 37 50 60 70
DEFAULT_SMS_CUDA_8  := 30 35 37 50 60
DEFAULT_SMS_CUDA_6  := 30 35 37 50

# Use .o as the default object extions.
OBJ_EXT := .o
# Set the default binary extension - no extension.
BIN_EXT :=

# OS Specific directoryes and file extensions.
ifeq ($(OS),Windows_NT)
	OS_BIN_DIR := x64
	OS_BUILD_DIR := x64
	# Override default values for windows.
	BIN_EXT := .exe
	OBJ_EXT := .obj
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OS_BIN_DIR := linux-x64
		OS_BUILD_DIR := linux-x64
	endif
endif

# OS specific output location for executable files
BIN_DIR := $(EXAMPLE_BIN_DIR)/$(OS_BIN_DIR)
# Temporary directory for building
BUILD_DIR := $(EXAMPLE_BUILD_DIR)/$(OS_BUILD_DIR)

# Path to FLAME GPU include directory
INCLUDE_DIR := $(FLAMEGPU_ROOT)include
INCLUDE_DIR_CUB := $(FLAMEGPU_ROOT)include_cub
# Path to FLAME GPU Lib directory (OS specific)
LIB_DIR := $(FLAMEGPU_ROOT)lib/

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
NVCC_MINOR = $(shell ""$(NVCC)"" --version | sed -n -r 's/.*(V([0-9]+).([0-9]+).([0-9]+))/\3/p')
NVCC_PATCH = $(shell ""$(NVCC)"" --version | sed -n -r 's/.*(V([0-9]+).([0-9]+).([0-9]+))/\4/p')
NVCC_GE_11_0 = $(shell [ $(NVCC_MAJOR) -ge 11 ] && echo true)
NVCC_GE_10_0 = $(shell [ $(NVCC_MAJOR) -ge 10 ] && echo true)
NVCC_GE_9_0  = $(shell [ $(NVCC_MAJOR) -ge 9 ] && echo true)
NVCC_GE_8_0  = $(shell [ $(NVCC_MAJOR) -ge 8 ] && echo true)


# If SM has been specified in place of SMS, set the value appropriately.
ifneq ($(SM),)
ifeq ($(SMS),)
SMS = $(SM)
$(warning "Warning - 'SMS' should be specified rather than 'SM'. Using '$(SM)'.")
endif
endif

# For the appropriate CUDA version, assign default SMS if required.
ifeq ($(NVCC_GE_11_0),true)
SMS ?= $(DEFAULT_SMS_CUDA_11)
else ifeq ($(NVCC_GE_10_0),true)
SMS ?= $(DEFAULT_SMS_CUDA_10)
else ifeq ($(NVCC_GE_9_0),true)
SMS ?= $(DEFAULT_SMS_CUDA_9)
else ifeq ($(NVCC_GE_8_0),true)
SMS ?= $(DEFAULT_SMS_CUDA_8)
else
SMS ?= $(DEFAULT_SMS_CUDA_6)
endif

# Incase SMS is comma separated, split the commas.
comma:= ,
empty:=
space:= $(empty) $(empty)
TMP_SMS:= $(subst $(comma),$(space),$(SMS))
override SMS = $(TMP_SMS)


# CUDA 11.0+ requires local versions of CUB to not be included (as it ships with CUDA), so only add it for CUDA < 11.0.
# Unsure how to do this for visual studio.
ifneq ($(NVCC_GE_11_0),true)
INCLUDE_DIRS += \
	$(INCLUDE_DIR_CUB)
endif

# Flags used for compilation.
# NVCC compiler flags
NVCCFLAGS   := -m64
# Host compiler flags (gcc/cicc etc), passed to nvcc using -Xcompiler
CCFLAGS     := 
# NVCC linker flags
NVCCLDFLAGS :=
# Host linker flags (ld etc), passed to nvcc using -Xlinker
LDFLAGS     :=

# By default, the mode type is relase and build director
Mode_TYPE := Release

# If the user specified any flags, pass them through.
ifneq ($(DEFINES),)
$(foreach DEF,$(DEFINES),$(eval NVCCFLAGS += -D$(DEF) ))
endif

# Release/Debug specific build flags and variables
ifeq ($(debug),1)
	NVCCFLAGS += -g -G -DDEBIG -D_DEBUG
	Mode_TYPE := Debug
else
	NVCCFLAGS += -lineinfo
endif

# Enable / Disable profiling
ifeq ($(profile),1)
	NVCCFLAGS += -DPROFILE -D_PROFILE
	ifeq ($(OS),Windows_NT)
	NVCCFLAGS+=-I"$(NVTOOLSEXT_PATH)include"
	NVCCLDFLAGS +=-L"$(NVTOOLSEXT_PATH)lib/x64" nvToolsExt64_1.lib
	else
		UNAME_S := $(shell uname -s)
		ifeq ($(UNAME_S),Linux)
			LDFLAGS +=-lnvToolsExt -L/usr/local/cuda-$(NVCC_MAJOR).$(NVCC_MINOR)/bin/../lib64
		endif
	endif
else
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
	LINK_ARCHIVES_VISUALISATION := -lfreeglut -lglew64
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		# Verify that xsltlint and xsltproc are on the path.
		XSLTPROC := $(shell command -v xsltproc 2> /dev/null)
		XMLLINT := $(shell command -v xmllint 2> /dev/null)
		# Pass specific nvcc flags for linux
		ifeq ($(NVCC_GE_11_0),true)
			# CUDA 11 Thrust requires c++14
			NVCCFLAGS += -std=c++14
			# CUDA 11.0RC adds deprecation annotations to old texture introp methods.
			NVCCFLAGS += -Xcompiler "-Wno-deprecated-declarations"
		else
			# Use c++11 otherwise to avoid breaking older compilers.
			NVCCFLAGS += -std=c++11
		endif
		CCFLAGS += -Wall
		# On linux we generate a runpath via -rpath and --enable-new-dtags. This enables a simple location for users who cannot install system wide dependencies a sensible place to put lib files.
		# Library files are looked for in LD_LIBRARY_PATH, the LIB_DIR, then system paths.
		# .so's can also be placed next to the binary file at runtime (but not compilation)
		NVCCLDFLAGS += -L$(LIB_DIR)
		LDFLAGS += --enable-new-dtags,-rpath,"\$$ORIGIN/../$(LIB_DIR)",-rpath,"\$$ORIGIN"
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

DISABLED_XSLT_TRANSFORMS :=

ifeq ($(TRANSFORM_HEADER_XSLT_DISABLED),1)
DISABLED_XSLT_TRANSFORMS += $(SRC_DYNAMIC)/header.h
endif
ifeq ($(TRANSFORM_FLAMEGPU_KERNALS_XSLT_DISABLED),1)
DISABLED_XSLT_TRANSFORMS += $(SRC_DYNAMIC)/FLAMEGPU_kernals.cu
endif
ifeq ($(TRANSFORM_IO_XSLT_DISABLED),1)
DISABLED_XSLT_TRANSFORMS += $(SRC_DYNAMIC)/io.cu
endif
ifeq ($(TRANSFORM_SIMULATION_XSLT_DISABLED),1)
DISABLED_XSLT_TRANSFORMS += $(SRC_DYNAMIC)/simulation.cu
endif
ifeq ($(TRANSFORM_MAIN_XSLT_DISABLED),1)
DISABLED_XSLT_TRANSFORMS += $(SRC_DYNAMIC)/main.cu
endif
ifeq ($(TRANSFORM_VISUALISTION_XSLT_DISABLED),1)
DISABLED_XSLT_TRANSFORMS += $(SRC_DYNAMIC)/visualistion.cu
endif

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

XSLT_FUNCTIONS_C := $(SRC_DYNAMIC)/functions.c.tmp
XSLT_COMMON_TEMPLATES := $(TEMPLATES_DIR)/_common_templates.xslt

# Build target for less xmllint validation
LAST_VALID_AT := $(BUILD_DIR)/.last_valid_at

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
validate: makedirs $(LAST_VALID_AT)

$(LAST_VALID_AT) : $(XML_MODEL_FILE) $(XSD_SCHEMA_DIR)/XMMLGPU.xsd $(XSD_SCHEMA_DIR)/XMML.xsd
ifndef XMLLINT
	$(warning "Warning: xmllint is not available, please install libxml2-utils to enable validation")
else
	$(XMLLINT) --noout $(XML_MODEL_FILE) --schema $(XSD_SCHEMA_DIR)/XMMLGPU.xsd && (touch $(LAST_VALID_AT))
endif

# Target to use xsltproc to generate all dynamic files.
xslt: validate $(XSLT_OUTPUT_FILES)

# Create functions.c prototypes as a differently named file, to avoid overwriting user code.
functions.c: validate $(XSLT_FUNCTIONS_C) $(MAKEFILE_LIST)

# Create the console version of this application, inlcuding directory creation and validation of the XML Model
console: makedirs validate $(TARGET_CONSOLE)

# Create the visualisation version of this application, inlcuding directory creation and validation of the XML Model
ifeq ($(HAS_VISUALISATION), 1)
visualisation: makedirs validate $(TARGET_VISUALISATION)
endif

# Rule to create header.h from XSLT. Depends upon both header.xslt and the XML file, so if either is changed a re-build will occur.
$(SRC_DYNAMIC)/%.h: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE) $(XSLT_COMMON_TEMPLATES) $(MAKEFILE_LIST)
# Error if XSLTPROC is not available
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif
# If the target file is not in the list of disabled targets, use the platform dependant method of generating the xlst or not.
ifeq ($(OS),Windows_NT)
	@if [ ! $(findstring $@, $(DISABLED_XSLT_TRANSFORMS)) ]; then \
		$(XSLTPROC) $(XML_MODEL_FILE) $< $@ ;\
		echo "$(XSLTPROC) $(XML_MODEL_FILE) $< $@";\
	else \
		echo "  Transformation of $@ is disabled.";\
    fi;
else
	@if [ ! $(findstring $@, $(DISABLED_XSLT_TRANSFORMS)) ]; then \
		$(XSLTPROC) $< $(XML_MODEL_FILE) > $@ ;\
		echo "$(XSLTPROC) $< $(XML_MODEL_FILE) > $@";\
	else \
		echo "  Transformation of $@ is disabled.";\
    fi;
endif

# Rule to create *.cu files in the dynamic folder, as requested by build dependencies.
$(SRC_DYNAMIC)/%.cu: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE) $(XSLT_COMMON_TEMPLATES) $(MAKEFILE_LIST)
# Error if XSLTPROC is not available
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif
# If the target file is not in the list of disabled targets, use the platform dependant method of generating the xlst or not. 
ifeq ($(OS),Windows_NT)
	@if [ ! $(findstring $@, $(DISABLED_XSLT_TRANSFORMS)) ]; then \
		$(XSLTPROC) $(XML_MODEL_FILE) $< $@ ;\
		echo "$(XSLTPROC) $(XML_MODEL_FILE) $< $@";\
	else \
		echo "  Transformation of $@ is disabled.";\
    fi;
else
	@if [ ! $(findstring $@, $(DISABLED_XSLT_TRANSFORMS)) ]; then \
		$(XSLTPROC) $< $(XML_MODEL_FILE) > $@ ;\
		echo "$(XSLTPROC) $< $(XML_MODEL_FILE) > $@";\
	else \
		echo "  Transformation of $@ is disabled.";\
    fi;
endif

# Rule to create functsion.c file in the dynamic folder.
$(SRC_DYNAMIC)/%.c.tmp: $(TEMPLATES_DIR)/%.xslt $(XML_MODEL_FILE) $(XSLT_COMMON_TEMPLATES) $(MAKEFILE_LIST)
# Error if XSLTPROC is not available
ifndef XSLTPROC
	$(error "xsltproc is not available, please install xlstproc")
endif
# If the target file is not in the list of disabled targets, use the platform dependant method of generating the xlst or not. 
ifeq ($(OS),Windows_NT)
	@if [ ! $(findstring $@, $(DISABLED_XSLT_TRANSFORMS)) ]; then \
		$(XSLTPROC) $(XML_MODEL_FILE) $< $@ ;\
		echo "$(XSLTPROC) $(XML_MODEL_FILE) $< $@";\
	else \
		echo "  Transformation of $@ is disabled.";\
    fi;
else
	@if [ ! $(findstring $@, $(DISABLED_XSLT_TRANSFORMS)) ]; then \
		$(XSLTPROC) $< $(XML_MODEL_FILE) > $@ ;\
		echo "$(XSLTPROC) $< $(XML_MODEL_FILE) > $@";\
	else \
		echo "  Transformation of $@ is disabled.";\
    fi;
endif

# Explicit rules for object files in the dynamic folder.
# Explicit rules are used (rather than generic) as the dependancis are non-trivial (currently)
$(BUILD_DIR)/io.cu$(OBJ_EXT): $(SRC_DYNAMIC)/io.cu $(SRC_DYNAMIC)/header.h $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BUILD_DIR)/simulation.cu$(OBJ_EXT): $(SRC_DYNAMIC)/simulation.cu $(SRC_DYNAMIC)/FLAMEGPU_kernals.cu $(FUNCTIONS_FILES) $(SRC_DYNAMIC)/header.h $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
$(BUILD_DIR)/main_console.cu$(OBJ_EXT): $(SRC_DYNAMIC)/main.cu $(SRC_DYNAMIC)/header.h $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ifeq ($(HAS_VISUALISATION), 1)
# Visualisation specific dynamic file rules.
$(BUILD_DIR)/main_visualisation.cu$(OBJ_EXT): $(SRC_DYNAMIC)/main.cu $(SRC_DYNAMIC)/header.h $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Only build visualistion$(OBJ_EXT) if not a custom visuations
ifeq ($(CUSTOM_VISUALISATION), 0)
$(BUILD_DIR)/visualisation.cu$(OBJ_EXT): $(SRC_DYNAMIC)/visualisation.cu $(SRC_VISUALISATION)/visualisation.h $(SRC_DYNAMIC)/header.h $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
else
# Rules for custom visualistion compilation of c files
$(BUILD_DIR)/%$(OBJ_EXT): $(SRC_VISUALISATION)/%.c $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Rules for custom visualistion compilation of cpp files
$(BUILD_DIR)/%$(OBJ_EXT): $(SRC_VISUALISATION)/%.cpp $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
# Rules for custom visualistion compilation of cu files
$(BUILD_DIR)/%.cu$(OBJ_EXT): $(SRC_VISUALISATION)/%.cu $(MAKEFILE_LIST)
	$(EXEC) $(NVCC) $(CONSOLE_INCLUDES) $(VISUALISATION_INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c -DVISUALISATION $<
endif

# Rule to create the visualisation binary by linking the dependant object files.
$(TARGET_VISUALISATION): $(VISUALISATION_DEPENDANCIES)
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LINK_ARCHIVES_VISUALISATION) -o $@ $+
endif

# Rule to create the console binary by linking the dependant object files.
$(TARGET_CONSOLE): $(CONSOLE_DEPENDANCIES)
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

# Clean object files, but do not regenerate xslt. `|| true` is used to support the case where dirs do not exist.
clean:
	@find $(EXAMPLE_BUILD_DIR)/ -name '*$(OBJ_EXT)' -delete 2> /dev/null || true
	@find $(EXAMPLE_BUILD_DIR)/ -name '*.cu$(OBJ_EXT)' -delete 2> /dev/null || true
	@find $(LAST_VALID_AT) -delete 2> /dev/null || true

# Clobber all temporary files, including dynamic files and the target executable. `|| true` is used to support the case where dirs do not exist.
clobber: clean
	@find $(SRC_DYNAMIC)/ -name '*.c' -delete 2> /dev/null || true
	@find $(SRC_DYNAMIC)/ -name '*.cu' -delete 2> /dev/null || true
	@find $(SRC_DYNAMIC)/ -name '*.h' -delete 2> /dev/null || true
	@find $(BIN_DIR)/ -name '$(EXAMPLE)$(BIN_EXT)' -delete 2> /dev/null || true

# Create any required directories.
makedirs:
	@mkdir -p $(BIN_DIR)/$(Mode_TYPE)_Console
	@mkdir -p $(BIN_DIR)/$(Mode_TYPE)_Visualisation
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(SRC_DYNAMIC)

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
	@echo "   functions.c   Generates functions.c.tmp in the dynamic folder using xslt"
	@echo "                   This file is for reference only. Not used in build."
	@echo ""
	@echo "  Arguments":
	@echo "    On first modifcation of values using this method ensure that files are"
	@echo "    rebuilt by using 'make -B <target> <args>'."
	@echo ""
	@echo "   debug=<arg>   Builds target in 'Release' or 'Debug' mode"
	@echo "                   0 : Release (Default)"
	@echo "                   1 : Debug"
	@echo "                   I.e. 'make console debug=1'"
	@echo "   profile=<arg> Includes NVTX ranges for a more detailed timeline"
	@echo "                   0 : Off (Default)"
	@echo "                   1 : On"
	@echo "                   I.e. 'make console profile=1'"
	@echo "   SMS=<arg>     Builds target for the specified CUDA architectures"
	@echo "                   I.e. 'make console SMS=\"60 61\"'"
	@echo "                   Defaults to: '$(SMS)'"
	@echo "************************************************************************"
