# ADMIRE (Aperture Domain Model Image REconstruction)

## Table of Contents
1. [Overview](#Overview)
2. [CPU Implementation Setup](#CPU-Implementation-Setup)
3. [GPU Implementation Setup](#GPU-Implementation-Setup)
4. [License](#License)
5. [Acknowledgements](#Acknowledgements)

## Overview
ADMIRE (Aperture Domain Model Image REconstruction) is a model-based approach to ultrasound beamforming. The overview of the method is that ultrasound channel data is first collected and time-delayed. Following this, the short-time Fourier transform (STFT) is taken through depth for each channel, and the aperture domain data for several frequencies within each STFT window are fit using models. The frequencies that are typically fit correspond to the bandwidth of the ultrasound pulse. Each model consists of the modeled wavefronts, localized in time and frequency, that return to the aperture from different scattering locations. Linear regression with elastic-net regularization is utilized to perform the fits, and the objective function is shown below. Essentially, the purpose of the model fits is to estimate how each scattering location contributes to a given set of aperture domain frequency data.

![objective function](https://latex.codecogs.com/svg.latex?%5Cboldsymbol%7B%5Chat%5Cbeta%7D%20%3D%20%5Cunderset%7B%5Cboldsymbol%7B%5Cbeta%7D%7D%7B%5Cmathrm%7Bargmin%7D%7D%5Cfrac%7B1%7D%7B2N%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cleft%28%5Cboldsymbol%7By%7D_%7Bi%7D%20-%20%5Csum_%7Bj%3D1%7D%5E%7BP%7D%20%5Cboldsymbol%7BX%7D_%7Bij%7D%5Cboldsymbol%7B%5Cbeta%7D_%7Bj%7D%5Cright%29%5E%7B2%7D%20&plus;%20%5Clambda%20%5Cleft%28%20%5Calpha%20%5Cleft%5C%7C%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%5C%7C_%7B1%7D%20&plus;%20%5Cfrac%7B%20%5Cleft%281%20-%20%5Calpha%20%5Cright%29%5Cleft%5C%7C%20%5Cboldsymbol%7B%5Cbeta%7D%20%5Cright%5C%7C_%7B2%7D%5E%7B2%7D%7D%7B2%7D%20%5Cright%29)

Once the models are fit, the decluttered aperture domain data for each frequency is reconstructed by only using the scattering locations that do not contribute to multipath or off-axis scattering. The inverse short-time Fourier transform (ISTFT) is then taken to obtain the decluttered channel data.

## CPU Implementation Setup
In order to utilize the CPU implementation of ADMIRE, an available release of MATLAB is required. Moreover, a C/C++ compiler that is compatible with the installed release of MATLAB must be installed in order to compile MEX-files containing C/C++ code. The compiler compatibility can be found at https://www.mathworks.com/support/requirements/supported-compilers.html. Note that the code was evaluated on both Windows and Linux OS. For Windows, the free community edition of Microsoft Visual Studio 2017 was used as the C/C++ compiler. To download this older version, go to https://visualstudio.microsoft.com/vs/older-downloads/ and create a free Dev Essentials program account with Microsoft. When installing Microsoft Visual Studio 2017, make sure to also check the box for the VC++ 2015 toolset (the 2015 will most likely be followed by a version number). For Linux, the GNU Compiler Collection (GCC) was used as the C/C++ compiler.

## GPU Implementation Setup 
In order to utilize the GPU implementation of ADMIRE, a CUDA-capable NVIDIA GPU along with an available release of MATLAB is required. The speedup that is obtained using the GPU implementation versus the CPU implementation can vary depending on the GPU that is used. The code was tested using an NVIDIA GeForce GTX 1080 Ti GPU, an NVIDIA GeForce GTX 2080 Ti GPU, and an NVIDIA GeForce GTX 1660 Ti laptop GPU. The MATLAB Parallel Computing Toolbox must also be installed if not already installed in order to allow for the compilation of MEX-files containing CUDA code. Moreover, a C/C++ compiler that is compatible with the installed release of MATLAB must be installed in order to compile MEX-files containing C/C++ code. The compiler compatibility can be found at https://www.mathworks.com/support/requirements/supported-compilers.html. Note that the code was evaluated on both Windows and Linux OS. For Windows, the free community edition of Microsoft Visual Studio 2017 was used as the C/C++ compiler. To download this older version, go to https://visualstudio.microsoft.com/vs/older-downloads/ and create a free Dev Essentials program account with Microsoft. When installing Microsoft Visual Studio 2017, make sure to also check the box for the VC++ 2015 toolset (the 2015 will most likely be followed by a version number). For Linux, the GNU Compiler Collection (GCC) was used as the C/C++ compiler. In addition to a C/C++ compiler, a CUDA toolkit version that is compatible with the installed release of MATLAB must be installed. To determine compatibility, refer to https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html. Once the compatibility is determined, go to https://developer.nvidia.com/cuda-toolkit-archive and install the particular CUDA toolkit version. Note that the installation process for the toolkit will also allow for the option to install a new graphics driver. If you do not desire to install a new driver, then you must ensure that your current driver supports the toolkit version that is being installed. For driver and toolkit compatability, refer to page 4 of https://docs.nvidia.com/pdf/CUDA_Compatibility.pdf. 

## License
Copyright 2020 Christopher Khan (christopher.m.khan@vanderbilt.edu)

This is free software made available under the Apache License, Version 2.0. For details, refer to the [LICENSE](LICENSE) file. 

## Acknowledgements
This work was supported by NIH grants R01EB020040 and S10OD016216-01 and NAVSEA grant N0002419C4302.
