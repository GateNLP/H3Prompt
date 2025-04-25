#!/bin/bash

echo docker buildx build -f Dockerfile -t elg.docker.gate.ac.uk/h3prompt:latest "$@" ..
