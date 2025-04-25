# some of the structure of this Dockerfile builds upon ideas in
# https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html

# This was the latest version on 20250424
#FROM condaforge/mambaforge:22.9.0-3
FROM mambaorg/micromamba:2.0.8-ubuntu24.04

USER root

# we need to libgl1 installed of the cuda stuff fails
RUN apt-get update && apt-get install libgl1 build-essential -y

USER $MAMBA_USER

WORKDIR /h3prompt

COPY Dataset Dataset
COPY checkpoint-2470 Model

# use the dependencies file to install a version of python we can use
# along with the nvidia GPU stuff etc.
COPY dependencies.yaml .
RUN --mount=type=cache,target=$MAMBA_ROOT_PREFIX/pkgs micromamba env create -p /h3prompt/env --file dependencies.yaml

# now we install the python packages we need
COPY requirements.txt .
RUN --mount=type=cache,target=/home/$MAMBA_USER/.cache,uid=$MAMBA_USER_ID,gid=$MAMBA_USER_ID micromamba run -p /h3prompt/env pip install -r requirements.txt

COPY demo.py README.md .

# This  is a hack for testing so the container doesn't immediately exit
CMD [ "sleep", "6000" ]
