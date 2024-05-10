# Use a Miniconda base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Clone the git repository and checkout the required branch
RUN git clone https://github.com/tsa87/TacoGFN-SBDD.git && \
    cd TacoGFN-SBDD && \
    git checkout multi-objective

# Navigate to the project directory
WORKDIR /app/TacoGFN-SBDD

# Install gdown and other pip dependencies (if any)
RUN pip install gdown

# Download and extract the dataset
# RUN gdown --id 1Mdg3eIhXube6TpctjPBUN5JVFDvkxNRO && \
#     tar -xvzf tacogfn_data.tar.gz

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the conda environment - Needs to be activated in an entry script as Docker does not support direct activation in build
# For building purposes, we install the package outside the environment, but ideally this should be inside the environment.
WORKDIR /app/TacoGFN-SBDD/src/molvoxel
RUN pip install -e .

# Setting up the entry point script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]