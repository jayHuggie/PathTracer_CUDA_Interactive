## Interactive Real-Time GPU Path Tracer with CUDA and OpenGL
<img src="/sample_images/Dragon_1000.png" alt="dragon_1000" title="Dragon example" width="500"/>
This 512 x 512 Dragon in a Cornell Box with 1000 samples per pixel took 20.74 seconds to render.

(The Dragon has 871,306 triangles!)

# Project Overview

Thanks to CUDA-based GPU acceleration and BVH, I was able to reduce the render time of a path-traced image from **42 seconds** to just **0.88 seconds** in my previous project ([PathTracer_CUDA](https://github.com/jayHuggie/PathTracer_CUDA)).

Building on this achievement, this project extends the original path tracer into an **interactive, real-time renderer**:

- **CUDA** handles the path tracing computations on the GPU.
- **OpenGL** displays the rendered image in real time.
- **GLFW** manages the application window and user input.
- A simple GUI is built using **Dear ImGui**.

# Features:

* Parses a scene in Mitsuba XML file format.
* GPU Path Tracer (with global illumination) using CUDA.
* Implements a BVH acceleration structure.
* Support for spheres and triangle meshes.
* Support for diffuse, mirror, plastic, Phong materials (with Fresnel reflection).
* Anti-aliasing, Russian Roulette termination.
* Exports the scene into a png file.

# Running the code

1. To compile, use the command:

```
make
```

in the home directory. (If the program cannot find CUDA, adjust the CUDA_PATH in the Makefile)

2. After compiling, use the command:
```
./torrey scenes/spheres/scene1.xml
```
'scenes/spheres/scene1.xml' after ./torrey is just an example. Additional scenes can also be rendered. The xml file can be located in various folders within the ‘scenes’ folder like:
```
./torrey scenes/cbox/cbox.xml
```
or like:
```
./torrey scenes/budda/buddha.xml
```
The resolution and sample size can be adjusted by changing the values of the xml files.  <br /> <br />
3. The rendered image can be found in the '**output**' folder under the name '**img.png**'!

# Tested on:
* System: Ubuntu 24.04.1 LTS
* CPU: Intel i7-13700k
* GPU: RTX 3080
* Nvidia Driver Version: 560.35.05
* CUDA Version: 12.6

## Dependencies

This project requires the following libraries and tools:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (>= 11.0)
- [OpenGL](https://www.opengl.org/) (provided by NVIDIA drivers)
- [GLFW](https://www.glfw.org/) (window and input management)
- [Dear ImGui](https://github.com/ocornut/imgui) (GUI; included as a submodule or source)
- [GLEW](http://glew.sourceforge.net/) (OpenGL extension loading, if required)
- [CMake](https://cmake.org/) (build system)
- C++17 compatible compiler (e.g., g++ 9+ or clang 10+)

### Installing Dependencies on Ubuntu

```bash
sudo apt update
sudo apt install build-essential cmake libglfw3-dev libglew-dev
```

- **CUDA Toolkit:**  
  Download and install from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit) to match your GPU and driver version.

- **Dear ImGui:**  
  This project includes Dear ImGui as a submodule. If you did not clone with submodules, run:
  ```bash
  git submodule update --init --recursive
  ```

# More sample images!

### Cornell Box

<img src="/sample_images/cbox.png" alt="cbox" title="Cornell Box example" width="330"/>
This 512 x 512 Cornell Box with 500 samples per pixel took 0.88 seconds to render. (A significant change compared to the previous version which took 42 seconds!)

### Spheres_Scene1 (Diffuse and Phong spheres)

<img src="/sample_images/scene1_phong.png" alt="scene1" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 0.15 seconds! (Previously took 10 seconds)

### Buddha

<img src="/sample_images/buddha.png" alt="buddha" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 16.5 seconds (The Buddha has 1,087,474 triangles, Prev. 44 sec)

### Bunny

<img src="/sample_images/bunny.png" alt="bunny" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 5.23 seconds  <br />
(The bunny has 144,046 triangles)

# Coming Soon..
Image Textures!
