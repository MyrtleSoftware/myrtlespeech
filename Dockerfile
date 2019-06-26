FROM continuumio/miniconda3

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

# use CI Hypothesis profile, see ``tests/__init__.py``
ENV HYPOTHESIS_PROFILE ci

ENTRYPOINT ["/bin/bash", "--login", "-c"]

CMD ["pytest"]
