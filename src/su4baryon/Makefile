# Set the main target
MAIN := main

# Paths
CHROMA := chroma_Nc4
BARYON := su4baryon
CONFIG := ../../install/$(CHROMA)/bin/chroma-config
IEIGEN := ../../src/eigen
INCLUDE_DIR := ./include
INSTALL_DIR := ../../install/$(BARYON)
BIN_DIR := bin

# Compiler and flags
CXX := $(shell $(CONFIG) --cxx)
CXXFLAGS := $(shell $(CONFIG) --cxxflags)
LDFLAGS := $(shell $(CONFIG) --ldflags)
LIBS := $(shell $(CONFIG) --libs)

# Additional flags and libraries
GPU_ARCH_FLAGS := --offload-arch=gfx90a
INCLUDE_DIRS := -I$(INCLUDE_DIR) -I$(IEIGEN)
AMD_LIBS := -lamdhip64 -lhipblas -lrocblas

# GPU executable name
GPU_EXE := $(BIN_DIR)/$(BARYON)

# Compilation command template for HIP files
COMPILE_HIP_CMD := $(CXX) $(GPU_ARCH_FLAGS) $(INCLUDE_DIRS) $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(AMD_LIBS) -c

# Compilation command template for CPP files
COMPILE_CPP_CMD := $(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -c


# Object files
OBJECTS := src/utils.o src/define_diagrams_gpu_original_Sungwoo.o src/main.o src/tfunc.o src/compute_t_gpu.o src/compute_Bprop_gpu.o src/diagram_utils.o

.PHONY: all clean

all: $(GPU_EXE)
	@echo "Compilation complete."

# Rule to link all object files into the GPU executable
$(GPU_EXE): $(OBJECTS)
	@echo "Linking $@"
	@mkdir -p $(BIN_DIR)
	$(CXX) $(GPU_ARCH_FLAGS) $(INCLUDE_DIRS) $(CXXFLAGS) $(LDFLAGS) $(LIBS) $(AMD_LIBS) $^ -o $@

# General rule for compiling .hip files to .o object files
src/%.o: src/%.hip
	@echo "Compiling $<"
	$(COMPILE_HIP_CMD) $< -o $@

# General rule for compiling .cpp files to .o object files
src/%.o: src/%.cpp
	@echo "Compiling $<"
	$(COMPILE_CPP_CMD) $< -o $@

# Rule for compiling main.cc specifically
src/main.o: src/main.cc
	@echo "Compiling $<"
	$(COMPILE_CPP_CMD) $< -o $@

install:
	@mkdir -p $(INSTALL_DIR)
	@mv $(BIN_DIR) $(INSTALL_DIR)/.


# Print out the configuration
info:
	@echo "CXX=$(CXX)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
	@echo "LIBS=$(LIBS)"

# Clean up object files
clean:
	@echo "Cleaning up..."
	rm -f $(OBJECTS) $(BIN_DIR)/$(notdir $(GPU_EXE))
