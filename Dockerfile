FROM continuumio/miniconda3

# fix https://github.com/conda/conda/issues/7267
RUN chown -R 1000:1000 /opt/conda/

# install headers to build regex as part of Black
# https://github.com/psf/black/issues/1112
RUN apt update && apt install -y build-essential python3-dev

# install cmake and make and ggc-6 for warp-transducer build
RUN apt-get update && apt-get install cmake make -y && \
    apt-get install software-properties-common -y && \
    apt-get update && \
    echo "deb  http://deb.debian.org/debian  stretch main" >> /etc/apt/sources.list && \
    echo "deb-src  http://deb.debian.org/debian  stretch main" >> /etc/apt/sources.list && \
    apt update && apt-get install gcc-6 g++-6 -y

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

# Install warp-transducer
ENV CXX=/usr/bin/g++-6
ENV CC=/usr/bin/gcc-6
RUN git clone https://github.com/HawkAaron/warp-transducer.git && \
    cd /home/user/myrtlespeech/warp-transducer && \
    git checkout c6d12f9e1562833c2b4e7ad84cb22aa4ba31d18c && \
    mkdir build && \
    cd build && \
    export WARP_RNNT_PATH=`pwd` && \
    cmake -D CMAKE_C_COMPILER=gcc-6 -D CMAKE_CXX_COMPILER=g++-6 .. && \
    make CC=gcc-6 CPP=g++-6 CXX=g++-6 LD=g++-6 && \
    cd ../pytorch_binding && \
    python3 setup.py install --user && \
    cd .. && rm -rf pytorch_binding/test tensorflow_binding && \
    echo "export WARP_RNNT_PATH=$WARP_RNNT_PATH" >> ~/.bashrc

# use CI Hypothesis profile, see ``tests/__init__.py``
ENV HYPOTHESIS_PROFILE ci

ENTRYPOINT ["/bin/bash", "--login", "-c"]

CMD ["pytest tests src"]
