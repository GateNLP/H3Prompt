#!/bin/sh

# with conda we were using --no-capture-output but
# this doesn't seem to be supported in micromamba

# uvicorn.workers.UvicornWorker
# custom_worker.CustomeUvicornWorker

$MAMBA_EXE run -p ./env gunicorn -k sync -b 0.0.0.0:8000 --workers $WORKERS --timeout 120 service:app
