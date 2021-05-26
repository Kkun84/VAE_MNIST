#!/bin/bash
docker exec -itd vae_test tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
