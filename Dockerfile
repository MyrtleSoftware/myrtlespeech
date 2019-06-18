FROM continuumio/miniconda3

# create non-root user
RUN useradd --create-home --shell /bin/bash user
USER user
WORKDIR /home/user

# setup Conda environment
COPY --chown=user:user environment.yml repaper/
RUN conda env create --quiet --file repaper/environment.yml

# ensure conda env is loaded
RUN echo "source /opt/conda/bin/activate repaper" > ~/.bashrc
RUN echo "force_color_prompt=no" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# install repaper package
COPY --chown=user:user . repaper/
WORKDIR /home/user/repaper
RUN pip install -e .

# use CI Hypothesis profile, see ``tests/__init__.py`
ENV HYPOTHESIS_PROFILE ci

ENTRYPOINT ["/bin/bash", "--login", "-c"]

CMD ["pytest"]
