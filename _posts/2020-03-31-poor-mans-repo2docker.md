---
toc: true
comments: true
layout: post
description: A solution to the environment and package management problems that plague data science projects.
categories: [python, conda, docker, data-science]
title: Conda, pip, and Docker FTW!
---
# Conda, pip, and docker FTW!

[Conda](https://docs.conda.io/en/latest/) is an open source package and 
environment management system that runs on Windows, Mac OS and Linux.

* Conda can quickly install, run, and update packages and their dependencies.
* Conda can create, save, load, and switch between project specific software 
  environments on your local computer.
* Although Conda was created for Python programs, Conda can package and 
  distribute software for any language such as R, Ruby, Lua, Scala, Java, 
  JavaScript, C, C++, FORTRAN.

Conda as a package manager helps you find and install packages. If you need a 
package that requires a different version of Python, you do not need to switch 
to a different environment manager, because Conda is also an environment 
manager. With just a few commands, you can set up a totally separate 
environment to run that different version of Python, while continuing to run 
your usual version of Python in your normal environment. 

While Conda is my default package and environment management solution, not 
every Python package that I might need to use is available via Conda. 
Fortunately, Conda plays nicely with [`pip`](https://pip.pypa.io/en/stable/) 
which is the default Python package management tool.

## Why not just use Conda (+ pip)?

While Conda (+ pip) solves most of my day-to-day data science environment and 
package management issues, incorporating [Docker](https://www.docker.com/) 
into my Conda (+ pip) development workflow has made it much easier to port my 
data science workflows from from my laptop/workstation to remote cloud 
computing resources.

Getting Conda (+ pip) to work as expected inside Docker containers turned out 
to be much more challenging that I expected. This blog post shows how I 
eventually combined Conda (+ pip) and Docker. In the following I assume that 
you have organized your project directory similar to my 
[Python data science project template](https://github.com/kaust-vislab/python-data-science-project). 
In particular, I will assume that you store all Docker related files in a 
`docker` sub-directory within your project root directory.

## Writing the `Dockerfile`

The trick to getting Conda (+ pip) and Docker to work smoothly together is to 
write a good `Dockerfile`. In this section I will take you step by step 
through the various pieces of the `Dockerfile` that I developed. Hopefully you 
can use this `Dockerfile` without modification on you next data science 
project.

Every `Dockefile` has a base or parent image. For the parent image I use 
[Ubuntu 16.04](http://releases.ubuntu.com/16.04/) 
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

## Building the Docker image

The following command builds a new image for your project with a custom `$USER` 
(and associated `$UID` and `$GID`) as well as a particular `$IMAGE_NAME` and 
`$IMAGE_TAG`. This command should be run within the `docker` sub-directory of 
the project as the Docker build context is set to `../` which should be the 
project root directory.

```bash
docker image build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --file Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  ../
```

## Running a Docker container

Once the image is built, the following command will run a container based on 
the image `$IMAGE_NAME:$IMAGE_TAG`. This command should be run from within the 
project's root directory.

```bash
docker container run \
  --rm \
  --tty \
  --volume ${pwd}/bin:/home/$USER/app/bin \
  --volume ${pwd}/data:/home/$USER/app/data \ 
  --volume ${pwd}/doc:/home/$USER/app/doc \
  --volume ${pwd}/notebooks:/home/$USER/app/notebooks \
  --volume ${pwd}/results:/home/$USER/app/results \
  --volume ${pwd}/src:/home/$USER/app/src \
  --publish 8888:8888 \
  $IMAGE_NAME:$IMAGE_TAG
```

## Using Docker Compose

It is quite easy to make typos whilst writing the above docker commands by hand. 
A less error-prone approach is to use 
[Docker Compose](https://docs.docker.com/compose/). The above docker commands can 
be encapsulated into the `docker-compose.yml` configuration file as follows.

```yaml
version: "3.7"

services:
  jupyterlab-server:
    build:
      args:
        - username=${USER}
        - uid=${UID}
        - gid=${GID}
      context: ../
      dockerfile: docker/Dockerfile
    ports:
      - "8888:8888"
    volumes:
      - ../bin:/home/${USER}/app/bin
      - ../data:/home/${USER}/app/data
      - ../doc:/home/${USER}/app/doc
      - ../notebooks:/home/${USER}/app/notebooks
      - ../results:/home/${USER}/app/results
      - ../src:/home/${USER}/app/src
    init: true
    stdin_open: true
    tty: true
```

The above `docker-compose.yml` file relies on 
[variable substitution](https://docs.docker.com/compose/environment-variables/#the-env-file).
to obtain the values for `$USER`, `$UID`, and `$GID`. These values can be 
stored in an a file called `.env` as follows.

```
USER=$USER
UID=$UID
GID=$GID
```

You can test your `docker-compose.yml` file by running the following command in 
the `docker` sub-directory of the project.

```bash
docker-compose config
```

This command takes the `docker-compose.yml` file and substitutes the values 
provided in the `.env` file and then returns the result.

Once you are confident that values in the `.env` file are being substituted 
properly into the `docker-compose.yml` file, the following command can be used 
to bring up a container based on your project's Docker image and launch the 
JupyterLab server. This command should also be run from within the `docker` 
sub-directory of the project.

```bash
docker-compose up --build
```

When you are done developing and have shutdown the JupyterLab server, the following command tears down the networking infrastructure for the running container.

```bash
docker-compose down
```
