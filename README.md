## A GPU Path Tracer written in CUDA C++ with BVH acceleration. (2025 Updated Version)
<img src="/sample_images/Dragon_1000.png" alt="dragon_1000" title="Dragon example" width="500"/>
This 512 x 512 Dragon in a Cornell Box with 1000 samples per pixel took 20.74 seconds to render. 

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

# More sample images!

### Cornell Box

<img src="/sample_images/cbox.png" alt="cbox" title="Cornell Box example" width="330"/>
This 512 x 512 Cornell Box with 500 samples per pixel took 0.88 seconds to render. (A significant change compared to the previous version which took 42 seconds!)

### Spheres_Scene1 (Diffuse and Phong spheres)

<img src="/sample_images/scene1_phong.png" alt="scene1" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 0.15 seconds! (Previously took 10 seconds)

### Buddha

<img src="/sample_images/buddha.png" alt="buddha" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 16.5 seconds (The Buddha has 1087474 triangles, Prev. 44 sec)

### Bunny

<img src="/sample_images/bunny.png" alt="bunny" width="330"/>
640 x 480 pixels with 500 samples per pixel. Render time: 5.23 seconds  <br />
(The bunny has 144046 triangles)

# Coming Soon..
Image Textures!
