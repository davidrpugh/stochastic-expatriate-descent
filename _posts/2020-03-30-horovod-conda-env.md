---
toc: true
comments: true
layout: post
description: Getting started with distributed training of DNNs using Horovod.
categories: [python, conda, deep-learning, pytorch, tensorflow, nvidia, horovod]
title: Building a Conda environment for distributed training with Horovod
---
# Getting Started with Horovod

<p align="center">
   <img alt="Horovod + NVIDIA FTW" src="{{ site.baseurl }}/images/horovod-nvidia.jpg" width="500">
</p>
[Horovod]() is an open-source distributed training framework for 
[TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), 
[PyTorch](https://pytorch.org/), and 
[Apache MXNet](https://mxnet.incubator.apache.org/). Horovod improves the speed, 
scale, and resource utilization of deep learning training.

In this post I describe how I build Conda environments for my deep learning 
projects where I plan to use Horovod to enable distributed training across 
multiple GPUs (either on the same node or spread across multuple nodes). If 
you like my approach then you can make use of the template repository on 
[GitHub](https://github.com/kaust-vislab/horovod-gpu-data-science-project) to 
get started with you rnext Horovod data science project!

## Installing the NVIDIA CUDA Toolkit

First thing you need to do is to install the 
[appropriate version](https://developer.nvidia.com/cuda-toolkit-archive) 
of the NVIDIA CUDA Toolkit on your workstation. For this blog post I am using 
[NVIDIA CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2) 
[(documentation)](https://docs.nvidia.com/cuda/archive/10.1/) which works with 
all three deep learning frameworks that are currently supported by Horovod.

### Why not just use the `cudatoolkit` package from `conda-forge`?

Typically when installing PyTorch, TensorFlow, or Apache MXNet with GPU support 
using Conda you simply add the appropriate version 
[`cudatoolkit`](https://anaconda.org/anaconda/cudatoolkit) package to your 
`environment.yml` file. 

Unfortunately, the `cudatoolkit` package available from 
[`conda-forge`](https://conda-forge.org/) does not include 
[NVCC](https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html) 
and in order to use Horovod with either PyTorch, TensorFlow, or MXNet you need 
to compile extensions.

### But what about the `cudatoolkit-dev` package from `conda-forge`?

While there are 
[`cudatoolkit-dev`](https://anaconda.org/conda-forge/cudatoolkit-dev) packages 
available from `conda-forge` that do include NVCC, I have had difficult getting 
these packages to consistently install properly. The most robust approach to 
obtain NVCC and still use Conda to manage all the other dependencies is to 
install the NVIDIA CUDA Toolkit on your system and then install a meta-package 
[`nvcc_linux-64`](https://anaconda.org/nvidia/nvcc_linux-64) from `conda-forge` 
which configures your Conda environment to use the NVCC installed on the system 
together with the other CUDA Toolkit components installed inside the Conda 
environment.

## The `environment.yml` File

Check the [official installation guide](https://horovod.readthedocs.io/en/latest/install_include.html) for Horovod more details.

### Channel Priority

```yaml
name: null

channels:
  - pytorch
  - conda-forge
  - defaults
```

### Dependencies

Below are the core required dependencies. Few things to note. Even though I have installed the NVIDIA CUDA Toolkit manually I still use Conda to manage the other required CUDA components such as `cudnn` and `nccl` (and the optional `cupti`). I use two meta-pacakges, `cxx-compiler` and `nvcc_linux-64`, to make sure that suitable C, and C++ compilers are installed and that the resulting Conda environment is aware of the manually installed CUDA Toolkit. Horovod also requires some controller library to coordinate work between the various Horovod processes. Typically this will be some MPI implementation such as [OpenMPI](). However, rather than specifying `openmpi` directly I instead opt for [mpi4py]() Conda package which provides a cuda-aware build of OpenMPI for your OS (where possible). Horovo also support that [Gloo]() collective communications library that can be used in place of MPI. I include `cmake` in order to insure that the Horovod extensions for Gloo are built.

```yaml
dependencies:
  - bokeh=1.4
  - cmake=3.16 # insures that the Gloo library extensions will be built
  - cudnn=7.6
  - cupti=10.1
  - cxx-compiler=1.0 # meta-pacakge that insures suitable C and C++ compilers are available
  - jupyterlab=1.2
  - mpi4py=3.0 # installs cuda-aware openmpi
  - nccl=2.5
  - nodejs=13
  - nvcc_linux-64=10.1 # meta-package that configures environment to be "cuda-aware"
  - pip=20.0
  - pip:
    - mxnet-cu101mkl==1.6.* # makes sure MXNET is installed prior to horovod
    - -r file:requirements.txt
  - python=3.7
  - pytorch=1.4
  - tensorboard=2.1
  - tensorflow-gpu=2.1
  - torchvision=0.5 
```

The complete `environment.yml` file is available on [GitHub](https://github.com/kaust-vislab/horovod-gpu-data-science-project/blob/master/environment.yml).

## The `requirements.txt` File

The `requirements.txt` file is where all of the `pip` dependencies, including 
Horovod itself, are listed for installation. In addition to Horovod I 
typically will also use `pip` to install JupyterLab extensions to enable GPU and 
CPU resource monitoring via [`jupyterlab-nvdashboard`]() 
and Tensorboard support via [`jupyter-tensorboard`]().

```bash
horovod==0.19.*
jupyterlab-nvdashboard==0.2.* # server-side component; client-side component installed in postBuild
jupyter-tensorboard==0.2.*

# make sure horovod is re-compiled if environment is re-built
--no-binary=horovod
```

Note the use of the `--no-binary` option at the end of the file. Including this 
option insures that Horovod will be re-built whenever the Conda environment is 
re-built.

The complete `requirements.txt` file is available on [GitHub](https://github.com/kaust-vislab/horovod-gpu-data-science-project/blob/master/requirements.txt).


## Building Conda Environment 

After adding any necessary dependencies that should be downloaded via `conda` 
to the `environment.yml` file and any dependencies that should be downloaded 
via `pip` to the `requirements.txt` file you create the Conda environment in a 
sub-directory `./env`of your project directory by running the following 
commands.

```bash
export ENV_PREFIX=$PWD/env
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL
conda env create --prefix $ENV_PREFIX --file environment.yml --force
```

By default Horovod will try and build extensions for all detected frameworks. 
See the Horovod documentation on 
[environment variables](https://horovod.readthedocs.io/en/latest/install_include.html#environment-variables) 
for the details on additional environment variables that can be set prior to 
building Horovod.

Once the new environment has been created you can activate the environment 
with the following command.

```bash
conda activate $ENV_PREFIX
```

## The `postBuild` File

If you wish to use any JupyterLab extensions included in the `environment.yml` 
and `requirements.txt` files then you need to rebuild the JupyterLab 
application using the following commands to source the `postBuild` script.

```bash
conda activate $ENV_PREFIX # optional if environment already active
. postBuild
```

## Wrapping it all up in a Bash script

I typically wrap these commands into a shell script `./bin/create-conda-env.sh`. 
Running the shell script will set the Horovod build variables, create the 
Conda environment, activate the Conda environment, and built JupyterLab with 
any additional extensions. 

```bash
#!/bin/bash --login

set -e

export ENV_PREFIX=$PWD/env
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL

conda env create --prefix $ENV_PREFIX --file environment.yml --force
conda activate $ENV_PREFIX
. postBuild
```

The script should be run from the project root directory as follows.

```bash
./bin/create-conda-env.sh # assumes that $CUDA_HOME is set properly
```

## Verifying the Conda environment

After building the Conda environment you can check that Horovod has been built 
with support for the deep learning frameworks TensorFlow, PyTorch, Apache 
MXNet, and the contollers MPI and Gloo with the following command.

```bash
conda activate $ENV_PREFIX # optional if environment already active
horovodrun --check-build
```

You should see output similar to the following.

```bash
Horovod v0.19.1:

Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [X] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo  
```

## Listing the full contents of the Conda environment

To see the full list of packages installed into the environment run the following command.

```bash
conda list --prefix $ENV_PREFIX
```

## Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file or the `requirements.txt` file 
after the environment has already been created, then you can re-create the environment with the 
following command.

```bash
conda env create --prefix $ENV_PREFIX --file environment.yml --force
```

However, whenever I add new dependencies I prefer to re-run the Bash script which will re-build both the Conda environment and JupyterLab.

```bash
./bin/create-conda-env.sh
```
