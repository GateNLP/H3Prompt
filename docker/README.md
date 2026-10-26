# Building Docker image and GATE Cloud app

The narrative classifier is deployed on GATE Cloud via a two step process, first the classifier itself is deployed as a container that exposes an ELG-compliant API endpoint using the Flask python HTTP framework, then the GATE Cloud API endpoint is a simple GATE app that contains an ELG client PR configured to call the Python endpoint.

## Building the Python classifier images

The Python-based classifier can be built using the `./build.sh` script in this directory.  To publish on the public GATE Cloud the images should be pushed to the `elg.docker.gate.ac.uk` registry:

```
./build.sh
```

Any arguments will be passed through to the `docker buildx` command, allowing you to specify things like `--push`, `--platform`, etc. as required.

## Building the GATE Cloud application

To build the wrapper application for GATE Cloud, you need an installation of Java, as the build process uses Gradle.  In the `cloud-application` folder, run

```
./gradlew zip
```

This will download the necessary plugins and assemble the final zip file in `build/distributions`.  If you need to override the default endpoint for the ELG service, use `-PelgEndpoint`, e.g.

```
./gradlew zip -PelgEndpoint=http://example.com/process
```