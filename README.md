## A GPU Path Tracer written in CUDA C++ with BVH acceleration.
<img src="/sample_images/cbox.png" alt="cbox" title="Cornell Box example" width="500"/>
This 512 x 512 Cornell Box with 500 samples per pixel took 109 seconds to render.

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

in the home directory. (If the program could not find CUDA, adjust the CUDA_PATH in the Makefile)

2. After compiling, use the command:
```
./torrey scenes/spheres/scene3.xml
```
'scenes/spheres/scene3.xml' after ./torrey is just an example. Additional scenes can also be rendered. The xml file can be located in various folders within the ‘scenes’ folder like:
```
./torrey scenes/cbox/cbox.xml
```
The resolution and sample size can be adjusted by changing the values of the xml files.

# Tested on:
* System: Ubuntu 23.04
* CPU: Intel i7-13700k
* GPU: RTX 3080
* Nvidia Driver Version: 535.86.10
* CUDA Version: 12.2

# More sample images!

### Spheres_Scene1 (Diffuse and Phong spheres)

<img src="/sample_images/scene1_phong.png" alt="scene1" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 10 seconds

### Buddha

<img src="/sample_images/buddha.png" alt="buddha" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 44 seconds (The Buddha has 1087474 triangles)

### Party (Bunny, Buddha, Armadillo, and Dragon)

<img src="/sample_images/party.png" alt="party" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 310 seconds  <br />
(The bunny has 144046 triangles, the dragon has 871306, and the armadillo has 212574)
