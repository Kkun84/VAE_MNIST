#!/bin/bash
docker exec -itd vae_mnist tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
