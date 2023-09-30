FROM lashi08/pytorch-build
ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y zsh
 
# zsh plugin
ENV SHELL /bin/zsh
COPY .devcontainer/zsh-plugins/.p10k.zsh .devcontainer/zsh-plugins/ohmyzsh.sh /root/
RUN chmod +x /root/ohmyzsh.sh && sh -c /root/ohmyzsh.sh

# this disable due to pytorch 2.0 replace the pytorch2.2 build version
# COPY .devcontainer/environment.yml /workspace/environment.yml
# RUN --mount=type=cache,target=/root/miniconda3/pkgs \
#     conda-env update -n nlp -f /workspace/environment.yml
    # /root/miniconda3/bin/conda env create -f /workspace/environment.yml

COPY .devcontainer/requirements.txt /workspace/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip conda run -n nlp pip install -r /workspace/requirements.txt

