# Docker

## Using the Docker Image

A Docker image is available [here](https://hub.docker.com/repository/docker/kundajelab/finemo_gpu/general). This is useful for running in containerized environments, such as in a Kubernetes cluster. To pull the image, run the following command:

```
docker pull kundajelab/finemo_gpu
```

To run the image, use the following command:

```
docker run -it --rm kundajelab/finemo_gpu finemo <args>
```

## Rebuilding the Docker Image

```
docker build -t kundajelab/finemo_gpu:dev  - < Dockerfile
```

Test the image by running the following command. You should see the help message for `finemo`:

```
$ docker run -it --rm kundajelab/finemo_gpu:dev finemo
usage: finemo [-h]
              {extract-regions-bw,extract-regions-chrombpnet-h5,extract-regions-h5,extract-regions-bpnet-h5,extract-regions-modisco-fmt,call-hits,report}
              ...
finemo: error: the following arguments are required: cmd
```
