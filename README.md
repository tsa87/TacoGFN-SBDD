
## Installation

1. Setup up Conda Environment
```bash
sudo apt-get install build-essential
sudo apt-get install libxau6

conda env create -f environment.yml
```

2. Download and unzip CrossDock Dataset
TODO: add instructions; move all protein.pdb, ligand.pdb into folder
TODO: add instructions on getting docking score from type files 
```bash 
```

3. Compute pharmacophore models
```bash
```

4. Install MolVoxel
```bash
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