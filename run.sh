#!/bin/bash
entry_point="torchserve --start --foreground --model-store /model_store"

docker run --gpus all -t --rm -p 8082:8080 -p 8083:8081 \
    -v /models:/models \
    -v /home/vikas/model_store:/model_store \
    -v /tmp:/tmp \
    torchserve_vikas \
    -c "${entry_point}"