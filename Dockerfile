FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Timezone setting
RUN apt-get update && apt-get install -y --no-install-recommends tzdata

# Install something
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-add-repository -y ppa:fish-shell/release-3
RUN apt-get update && apt-get install -y --no-install-recommends fish

RUN apt-get update && apt-get install -y --no-install-recommends nano git sudo curl

# OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev

# Install Python library
COPY requirements.txt /
RUN pip install -r /requirements.txt

ARG UID
ARG USER
ARG PASSWORD
RUN groupadd -g 1000 ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd
USER ${USER}

ENV PATH $PATH:/home/${USER}/.local/bin

# # Build: docker build -t project_name .
# # Run: docker run --gpus all -it --rm project_name

# # Build from official Nvidia PyTorch image
# # GPU-ready with Apex for mixed-precision support
# # https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# # https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
# FROM nvcr.io/nvidia/pytorch:21.03-py3


# # Copy all files
# ADD . /workspace/project
# WORKDIR /workspace/project


# # Create myenv
# RUN conda env create -f conda_env_gpu.yaml -n myenv
# RUN conda init bash


# # Set myenv to default virtual environment
# RUN echo "source activate myenv" >> ~/.bashrc
