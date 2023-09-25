FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive


RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y  bash \
                        build-essential \
                        git \
                        curl \
                        wget \
                        zsh \
                        ca-certificates 

# instal conda 
RUN mkdir /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh 

COPY .devcontainer/environment.yml /workspace/environment.yml
RUN --mount=type=cache,target=/root/miniconda3/pkgs \
    /root/miniconda3/bin/conda env create -f /workspace/environment.yml

# zsh plugin
ENV SHELL /bin/zsh
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.zsh_history" \
    && echo "$SNIPPET" >> "/root/.zsh_history"
COPY .devcontainer/zsh-plugins/.p10k.zsh /root/


# install cmake
ARG CMAKE_VERSION=3.27.3
RUN curl -O -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    yes | sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh


