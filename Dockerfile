FROM continuumio/miniconda3

# fix https://github.com/conda/conda/issues/7267
RUN chown -R 1000:1000 /opt/conda/

# install headers to build regex as part of Black
# https://github.com/psf/black/issues/1112
# and also install required components for
# warp-transducer build (make, cmake gcc-6)
RUN apt-get update && apt-get install -y \
    cmake \
    g++-7 \
    gcc-7 \
    make \
    python3-dev

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

# install warp-transducer
ENV CXX=/usr/bin/g++-7
ENV CC=/usr/bin/gcc-7
RUN make deps/warp-transducer

# use CI Hypothesis profile, see ``tests/__init__.py``
ENV HYPOTHESIS_PROFILE ci

ENTRYPOINT ["/bin/bash", "--login", "-c"]

CMD ["pytest"]
