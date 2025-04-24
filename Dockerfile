# some of the structure of this Dockerfile builds upon ideas in
# https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html

# This was the latest version on 20250424
#FROM condaforge/mambaforge:22.9.0-3
FROM mambaorg/micromamba:2.0.8-ubuntu24.04

USER root

# we need to libgl1 installed of the cuda stuff fails
RUN apt-get update && apt-get install libgl1 build-essential -y

USER mambauser

# use the lock file to install the none python dependencies
COPY dependencies.yaml .
RUN --mount=type=cache,target=/opt/conda/pkgs micromamba env create -p /tmp/env --file dependencies.yaml

# now we install the python packages we need
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache micromamba run -p /tmp/env pip install -r requirements.txt

COPY Dataset Dataset

COPY demo.py README.md .

CMD [ "sleep", "6000" ]
