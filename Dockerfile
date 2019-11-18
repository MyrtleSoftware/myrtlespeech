FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM continuumio/miniconda3

RUN apt-get install locate && locate cuda | grep /cuda$
RUN git

# fix https://github.com/conda/conda/issues/7267
RUN chown -R 1000:1000 /opt/conda/

# install headers to build regex as part of Black
# https://github.com/psf/black/issues/1112
RUN apt update && apt install -y build-essential python3-dev

# create non-root user
RUN useradd --create-home --shell /bin/bash user
USER user
WORKDIR /home/user

# setup Conda environment
COPY --chown=user:user environment.yml myrtlespeech/
RUN conda env create --quiet --file myrtlespeech/environment.yml

# ensure conda env is loaded
RUN echo "source /opt/conda/bin/activate myrtlespeech" > ~/.bashrc
RUN echo "force_color_prompt=no" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# install myrtlespeech package
COPY --chown=user:user . myrtlespeech/
WORKDIR /home/user/myrtlespeech
RUN pip install -e .
RUN protoc --proto_path src/ --python_out src/ src/myrtlespeech/protos/*.proto --mypy_out src/
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 880ab925bce9f817a93988b021e12db5f67f7787 && \
    pip install -v --no-cache-dir ./ && \
    cd .. && \
    rm -rf apex

USER root
RUN apt-get update && apt-get install make cmake build-essential -y #&& make deps/warp-transducer

USER user
RUN git clone https://github.com/HawkAaron/warp-transducer build/warp-transducer; exit 0
RUN cd build/warp-transducer && \
    git reset --hard c6d12f9e1562833c2b4e7ad84cb22aa4ba31d18c && \
		mkdir build && \
		cd build && \
		cmake .. && \
		make
ENV WARP_RNNT_PATH=/usr/local/lib
ENV CUDA_HOME=/usr/local/cuda
RUN echo "export WARP_RNNT_PATH=/usr/local/lib" >> ~/.bashrc && \
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
USER root
RUN cp build/warp-transducer/build/libwarprnnt.so $WARP_RNNT_PATH
USER user
RUN export CUDA_HOME=/usr/local/cuda && export WARP_RNNT_PATH=/usr/local/lib && \
  cd build/warp-transducer/pytorch_binding && \
		python3 setup.py install --user

# use CI Hypothesis profile, see ``tests/__init__.py``
ENV HYPOTHESIS_PROFILE ci

ENTRYPOINT ["/bin/bash", "--login", "-c"]

CMD ["pytest"]
