CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++-12
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

# OpenGL dependencies
GLFW_LIBS = -lglfw
GLEW_LIBS = -lGLEW
OPENGL_LIBS = -lGL

# ImGui dependencies
IMGUI_DIR = 3rdparty/imgui
IMGUI_SOURCES = $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
IMGUI_OBJS = $(IMGUI_SOURCES:.cpp=.o)

# Update include paths to use relative paths for ImGui
HOST_COMPILER_FLAGS = -m64 -std=c++17 -I$(CUDA_PATH)/include -I./3rdparty/imgui -I./3rdparty/imgui/backends
NVCCFLAGS = $(NVCC_DBG) -m64 -std=c++17 -allow-unsupported-compiler -I./3rdparty/imgui -I./3rdparty/imgui/backends
GENCODE_FLAGS  = -gencode arch=compute_86,code=sm_86

INCS = bbox.cuh bvh.cuh 3rdparty/stb_image.h 3rdparty/stb_image_write.h camera.cuh compute_normals.h frame.h flexception.h intersection.h light.h material.h matrix.h parse_obj.h parse_ply.h parse_scene.h parse_serialized.h print_scene.h radiance.cuh texture.h transform.h cutil_math.h ray.h torrey.cuh scene.h 3rdparty/miniz.h 3rdparty/pugiconfig.hpp 3rdparty/pugixml.hpp 3rdparty/stb_image.h 3rdparty/tinyexr.h 3rdparty/tinyply.h shape.cuh imgui_impl.h
SRCS = compute_normals.cpp parse_obj.cpp parse_ply.cpp parse_scene.cpp parse_serialized.cpp print_scene.cpp scene.cpp transform.cpp 3rdparty/pugixml.cpp 3rdparty/miniz.c $(IMGUI_SOURCES)

CU_SRCS = main.cu bvh.cu
CU_OBJS := $(patsubst %.cu, %.o, $(CU_SRCS))

# .cpp to .o
CPP_OBJS := $(patsubst %.cpp, %.o, $(filter %.cpp, $(SRCS)))

%.o: %.cpp
	$(HOST_COMPILER) $(HOST_COMPILER_FLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

torrey: $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o torrey $(CU_OBJS) $(CPP_OBJS) $(GLFW_LIBS) $(GLEW_LIBS) $(OPENGL_LIBS)

out.ppm: torrey
	rm -f out.ppm
	./torrey > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: torrey
	nvprof ./torrey > out.ppm

# use nvprof --query-metrics
profile_metrics: torrey
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./torrey > out.ppm

clean:
	rm -f torrey *.o out.ppm out.jpg
