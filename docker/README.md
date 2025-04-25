# Building Docker image and GATE Cloud app

The narrative classifier is deployed on GATE Cloud via a two step process, first the classifier itself is deployed as a container that exposes an ELG-compliant API endpoint using the Flask python HTTP framework, then the GATE Cloud API endpoint is a simple GATE app that contains an ELG client PR configured to call the Python endpoint.

## Building the Python classifier images

The Python-based classifier can be built using the `./build.sh` script in this directory.  To publish on the public GATE Cloud the images should be pushed to the `elg.docker.gate.ac.uk` registry:

```
./build.sh
```

Any arguments will be passed through to the `docker buildx` command, allowing you to specify things like `--push`, `--platform`, etc. as required.

