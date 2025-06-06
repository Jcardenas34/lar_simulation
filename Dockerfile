# Start with a Miniconda base image
FROM continuumio/miniconda3

# Set a working directory
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Copy the environment.yml separately first to leverage Docker caching
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Make sure the environment is activated in all subsequent layers
SHELL ["conda", "run", "-n", "lar_simulation", "/bin/bash", "-c"]

# Install your package (assuming a src layout and pyproject.toml)
RUN pip install -e .

# Optionally set environment path so it's default
ENV PATH=/opt/conda/envs/lar_simulation/bin:$PATH

# Set default command (can be overridden)
CMD ["conda", "run", "-n", "lar_simulation", "python", "scripts/main.py"]
