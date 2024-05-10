#!/bin/bash

# Activate the conda environment
echo "Activating conda environment..."
source activate $(head -1 /app/TacoGFN-SBDD/environment.yml | cut -d' ' -f2)

# Execute the passed command
exec "$@"