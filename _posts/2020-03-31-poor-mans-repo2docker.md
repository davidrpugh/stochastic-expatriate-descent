---
toc: true
comments: true
layout: post
description: A solution to the environment and package management problems that plague data science projects.
categories: [python, conda, docker, data-science]
title: Conda, pip, and Docker FTW!
---
# Conda, pip, and docker FTW!

## Environment management problem

## Package management problem

## Why not just use Conda (+ pip)?

# The `Dockerfile`

For the parent image I use [Ubuntu 16.04](http://releases.ubuntu.com/16.04/) 
which is one of the most commonly used flavor of Linux in the data science 
community (and also happens to be the same OS installed on my workstation).

```
FROM ubuntu:16.04
```

### Make `bash` the default shell

The default shell used to run `Dockerfile` commands when building Docker 
images is `/bin/sh`. Unfortunately `/bin/sh` is currently not one of the shells 
supported by the `conda init` command. Fortunately it is possible to change the 
default shell used to run `Dockerfile` commands using the 
[`SHELL`](https://docs.docker.com/engine/reference/builder/#shell) instruction.

```
SHELL [ "/bin/bash", "--login", "-c" ]
```

Note the use of the `--login` flag which insures that both `~/.profile` and 
`~/.bashrc` are sourced properly. Proper sourcing of both `~/.profile` and 
`~/.bashrc` is necessary in order to use various `conda` commands to build the 
Conda environment inside the Docker image.

### Create a non-root user

It is a 
[Docker security "best practice"](https://snyk.io/blog/10-docker-image-security-best-practices/) 
to create a non-root user inside your Docker images. My preferred approach to 
create a non-root user uses build arguments to customize the `username`, `uid`, 
and `gid`the non-root user. I use standard defaults for the `uid` and `gid`; 
the default username is set to `al-khawarizmi` in honor of the 9th-century 
Iranian mathematician 
[Muhammad ibn Musa al-Khwarizmi](https://en.wikipedia.org/wiki/Muhammad_ibn_Musa_al-Khwarizmi)

```
# Create a non-root user
ARG username=al-khawarizmi
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER
```

### Copy over the config files for the Conda environment

After creating the non-root user I copy over all of the config files that I 
will need to create the Conda environment (i.e., `environment.yml`, 
`requirements.txt`, `postBuild`). I also copy over a Bash script that I will 
use as the Docker `ENTRYPOINT` (more on this below).

```
COPY environment.yml requirements.txt /tmp/
RUN chown $UID:$GID /tmp/environment.yml /tmp/requirements.txt

COPY postBuild /usr/local/bin/postBuild.sh
RUN chown $UID:$GID /usr/local/bin/postBuild.sh && \
    chmod u+x /usr/local/bin/postBuild.sh

COPY docker/entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh
```

Newer versions of Docker support copying files as a non-root user, however 
the version of Docker available on DockerHub does not yet support copying 
as a non-root user so if you want to setup 
[automated builds](https://docs.docker.com/docker-hub/builds/) for your Git 
repositories you will need to copy everything as root.

### Install Miniconda as the non-root user.

After copying over the config files as root, I switch over to the non-root 
user and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

```
USER $USER

# install miniconda
ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
```

### Create a project directory

Next I create a project directory inside the non-root user home directory. The 
Conda environment will be created in a `env` sub-directory inside the project 
directory and all other project files and directories can then be mounted into 
this directory. 

```
# create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR
```

### Build the Conda environment

Now I am ready to build the Conda environment. Note that I can use nearly the 
same sequence of `conda` commands that I would use to build a Conda environment 
for a project on my laptop or workstation.

```
# build the conda environment
ENV ENV_PREFIX $PWD/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

# run the postBuild script to install any JupyterLab extensions
RUN conda activate $ENV_PREFIX && \
    /usr/local/bin/postBuild.sh && \
    conda deactivate
```

### Insure Conda environment is properly activated at runtime

Almost finished! Second to last step is to use an 
[`ENTRYPOINT`](https://docs.docker.com/engine/reference/builder/#entrypoint) 
script to insure that the Conda environment is properly activated at runtime.

```
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
```

Here is the `/usr/local/bin/entrypoint.sh` script for reference.

```bash
#!/bin/bash --login
set -e

conda activate $ENV_PREFIX
exec "$@"
```

### Specify a default command for the Docker container

Finally, I use the [`CMD`](https://docs.docker.com/engine/reference/builder/#cmd) 
instruction to specify a default command to run when a Docker container is 
launched. Since I install JupyerLab in all of my Conda environments I tend to 
launch a JupyterLab server by default when executing containers.

```
# default command will be to launch JupyterLab server for development
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
```

