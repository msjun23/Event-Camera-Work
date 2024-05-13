# Event-Camera-Work
General starting code for Event camera works

# Environment

## Run docker container
- Docker image: [pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/2.1.2-cuda11.8-cudnn8-devel/images/sha256-66b41f1755d9644f6341cf4053cf2beaf3948e2573acf24c3b4c49f55e82f578?context=explore)
```bash
docker run \
    -v /home/user/your/repo/path:/root/code \
    -v /path/to/your/datasets:/root/data \
    -it --gpus=all --ipc=host --name=event_general \
    pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
```

# Training
## Build deformable convolution
```bash
cd /root/code/models/deform_conv/ && bash build.sh
```