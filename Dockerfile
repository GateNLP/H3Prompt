# some of the structure of this Dockerfile builds upon ideas in
# https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html

# This was the latest version on 20250424
FROM condaforge/mambaforge:24.9.2-0

# we need to libgl1 installed of the cuda stuff fails
RUN apt-get update && apt-get install libgl1 -y

# use the lock file to install the none python dependencies
COPY dependencies.yaml .
RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -p /env --file dependencies.yaml

# now we install the python packages we need
RUN --mount=type=cache,target=/root/.cache conda run -p /env pip install unsloth
