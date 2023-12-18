#!/bin/bash
# Setup ADFR Suite - to prepare protein-ligand docking program

# Function to display error message and exit with a non-zero status
function die {
    echo "Error: $1"
    exit 1
}

# Download URL for the tarball
download_url="https://ccsb.scripps.edu/adfr/download/1038/ADFRsuite_x86_64Linux_1.0.tar.gz"
tools_directory="tools/"

# Create the tools directory if it doesn't exist
mkdir -p "$tools_directory"

# Full path for the tarball in the tools directory
tarball_path="${tools_directory}ADFRsuite_x86_64Linux_1.0.tar.gz"

# Download the tarball
echo "Downloading ADFRsuite tarball..."
wget -q -O "$tarball_path" "$download_url" || die "Failed to download ADFRsuite tarball."

# Extract the tarball in the tools directory
echo "Extracting ADFRsuite..."
tar -xzf "$tarball_path" -C "$tools_directory" || die "Failed to extract ADFRsuite tarball."

# Step into the ADFRsuite folder
cd "${tools_directory}ADFRsuite_x86_64Linux_1.0" || die "Failed to enter ADFRsuite folder."

# Run the installation script
echo "Running ADFRsuite installer..."
./install.sh -d $(pwd) -c 0 || die "Failed to install ADFRsuite."

# Set permissions for the ADFRsuite binaries
chmod -R a+x bin || die "Failed to set permissions for ADFRsuite binaries."

# Add ADFRsuite to the PATH
echo "export PATH=\$PATH:$(pwd)/bin" >> ~/.bashrc
source ~/.bashrc

echo "ADFRsuite installation completed successfully."