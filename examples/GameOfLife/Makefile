################################################################################
#
# FLAME GPU Script for CUDA 7.5, CUDA 8.0
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
################################################################################
#
# Makefile project only supported on Linux Platforms
#
################################################################################
# Folders containing FLAMEGPU example files and templates
BIN_DIR := ../../bin/linux-x64/
 
IDIR := ../../include/
# For now this will only work for x64 linux. 32x is uninportant as deprecated in CUDA 8.0 Other systems are currently not possible to test.
LDIR := ../../lib/x86_64-linux-gnu/
TEMPLATE := ../../FLAMEGPU/templates/
XSD_SCHEMA := ../../FLAMEGPU/schemas/

INPUT_DATA:=iterations/0.xml

SRC_ := src/model/
SRC_VIZ := src/visualisation/
SRC_CUDA := src/dynamic/

OPENGL_FLAGS := -lglut -lGLEW -lGLU -lGL
FLAMELIB := -I $(IDIR) -I $(SRC_) -I $(SRC_CUDA) -I $(SRC_VIZ) -I $(IDIR)GL/

################################################################################
#Generating Dynamic Code from FLAMEGPU Templates

XML_MODEL:=$(SRC_)XMLModelFile.xml

all: XSLTPREP build

XML_Validate:
	xmllint --noout $(XML_MODEL) --schema $(XSD_SCHEMA)XMMLGPU.xsd 

XSLTPREP: XML_Validate
XSLTPREP:
	xsltproc $(TEMPLATE)header.xslt  $(XML_MODEL)> $(SRC_CUDA)header.h 
	xsltproc $(TEMPLATE)FLAMEGPU_kernals.xslt $(XML_MODEL) > $(SRC_CUDA)FLAMEGPU_kernals.cu
	xsltproc $(TEMPLATE)io.xslt $(XML_MODEL) > $(SRC_CUDA)io.cu 
	xsltproc $(TEMPLATE)simulation.xslt $(XML_MODEL) > $(SRC_CUDA)simulation.cu 
	xsltproc $(TEMPLATE)main.xslt $(XML_MODEL) > $(SRC_CUDA)main.cu
	xsltproc $(TEMPLATE)visualisation.xslt $(XML_MODEL) > $(SRC_CUDA)visualisation.cu

################################################################################

HOST_COMPILER ?= g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m64
CCFLAGS     :=
LDFLAGS     := -L$(LDIR)

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      Mode_TYPE := Debug
else
      Mode_TYPE := Release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc

################################################################################

SAMPLE_ENABLED := 1

# Gencode arguments
SMS ?= 30 35 37 50 60

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
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

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
help:
	@echo "************************************************************************"
	@echo "*  Copyright 2017 University of Sheffield.  All rights reserved.       *"
	@echo "************************************************************************"
	@echo "make all -> validate &processes XSL model and generates all .cu files  *" 
	@echo "                                                                       *"  
	@echo "           ------------------------------------------------            *"
	@echo "make XSLTPREP -> Validate and preprocesses the xml model               *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"
	@echo "make XML_Validate -> To validate the XML file                          *"
	@echo "run "sudo apt install libxml2-utils" to install xmllint                *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"
	@echo "make build -> builds all executables in either release or debug        *"
	@echo "                   -Debug_Console                                      *"
	@echo "                   -Debug_Visualisation                                *"
	@echo "                          OR                                           *"
	@echo "                   -Release_Console                                    *"
	@echo "                   -Release_Visualisation                              *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"
	@echo "make visualisation_mode -> builds executables in visualisation mode    *"
	@echo "                                                                       *"
	@echo "make console_mode -> builds executables in console mode                *"
	@echo "                                                                       *"
	@echo "make < .. > dbg='arg' -> builds in Release/Debug only                  *"
	@echo "                                'arg' -> 0 or 1 value                  *"
	@echo "                                                                       *"
	@echo "To run executables for console mode, run below command:                *"
	@echo "make run_console iter='arg'                                            *"
	@echo "           Note that without the 'arg', it only runs for 1 iteration   *"
	@echo "           ------------------------------------------------            *"   
	@echo "To run executables for visualisation mode, run below command:          *"
	@echo "make run_vis                                                           *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"                
	@echo "Alternatively, run the bash script stored in bin/linux-x64.            *"
	@echo "The iteration default value in console mode is 10. You can simple      *"
	@echo "change it by entering a new value while running the ./*.sh file.       *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"
	@echo "Note: You can manualy change the location/name of the INPUT_DATA       *"
	@echo "                                                                       *" 
	@echo "************************************************************************"
 
build: Visualisation_mode Console_mode 

Visualisation_mode:  GameOfLife_vis

Console_mode:  GameOfLife_console

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

io.o: $(SRC_CUDA)io.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

simulation.o: $(SRC_CUDA)simulation.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

visualisation.o: $(SRC_CUDA)visualisation.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c -DVISUALISATION $<

main.o: $(SRC_CUDA)main.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

main_.o: $(SRC_CUDA)main.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c -DVISUALISATION $<


GameOfLife_vis:	BUILD_TYPE=$(Mode_TYPE)_Visualisation
GameOfLife_vis: io.o simulation.o main_.o visualisation.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(OPENGL_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(BUILD_TYPE)
	$(EXEC) mv $@ $(BIN_DIR)$(BUILD_TYPE)
	@echo "LD_LIBRARY_PATH=../../lib/x86_64-linux-gnu/ ./$(BUILD_TYPE)/GameOfLife_vis ../../examples/GameOfLife/$(INPUT_DATA)" > $(BIN_DIR)GameOfLife_vis.sh
	chmod +x $(BIN_DIR)GameOfLife_vis.sh


GameOfLife_console: BUILD_TYPE=$(Mode_TYPE)_Console
GameOfLife_console: io.o simulation.o  main.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(BUILD_TYPE)
	$(EXEC) mv $@ $(BIN_DIR)$(BUILD_TYPE)
	@echo ./$(BUILD_TYPE)/GameOfLife_console ../../examples/GameOfLife/$(INPUT_DATA) '$$'{1:-1}> $(BIN_DIR)GameOfLife_console.sh
	chmod +x $(BIN_DIR)GameOfLife_console.sh


run_console: GameOfLife_console
	cd $(BIN_DIR) && ./GameOfLife_console.sh $(iter)

run_vis: GameOfLife_vis
	cd $(BIN_DIR) && ./GameOfLife_vis.sh

clean:
	find . -name '*.o' -delete

clobber: clean 
	find . -name '*.cu' -delete
	rm -rf $(BIN_DIR)

