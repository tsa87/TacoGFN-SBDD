
## Installation

1. Setup up Conda Environment
```bash
(Optional for remote machine that has nothing)
sudo apt-get -y install build-essential
sudo apt-get -y install libxau6
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11

conda install -n base conda-libmamba-solver --yes
conda config --set solver libmamba

conda env create -f environment.yml
```

2. Download dataset
```bash
pip install gdown
gdown --id 1Mdg3eIhXube6TpctjPBUN5JVFDvkxNRO
tar -xvzf tacogfn_data.tar.gz
```

3. Compute pharmacophore models
```bash
```

4. Install MolVoxel
```bash
(Changing this because an error)
cd src/molvoxel
pip install -e .
```

4. Install ADFR for Docking
```bash
chmod +x scripts/setup_adfr_suite.sh
./setup_adfr.sh
```

5. Export relevant paths 
```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Change directory to repo root
export PATH="$(pwd)/tools:$PATH" 
export PATH="$(pwd)/tools/ADFRsuite_x86_64Linux_1.0/bin:$PATH"
```
