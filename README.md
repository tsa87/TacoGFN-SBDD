
## Installation

1. Setup up Conda Environment
```bash
(Optional for remote machine that has nothing)
sudo apt-get -y install build-essential
sudo apt-get -y install libxau6
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11
sudo apt-get install -y libsm6 libxrender1 libfontconfig1

conda install -n base conda-libmamba-solver --yes
conda config --set solver libmamba

conda env create -f environment.yml
```

2. Download dataset
```bash
(if creating the compressed file)
tar -czvf tacogfn_data.tar.gz dataset/split_by_name.pt model_weights/ dataset/pocket_to_avg_zinc_vina_score.pt misc/pharmacophores_db.lmdb/ dataset/affinity_prediction_pharmacophores/

(otherwise)
cd tacogfn
pip install gdown
gdown --id 1Mdg3eIhXube6TpctjPBUN5JVFDvkxNRO
tar -xvzf tacogfn_data.tar.gz
```

3. Install MolVoxel
```bash
(Changing this because an error)
cd src/molvoxel
pip install -e .
```

4. Train
```bash
conda activate tacogfn
python3 src/tacogfn/tasks/pharmaco_frag.py --hps_path hps/???
```


6. Install ADFR for Docking (Optional)
```bash
chmod +x scripts/setup_adfr_suite.sh
./setup_adfr.sh
```

6. Export relevant paths 
```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Change directory to repo root
export PATH="$(pwd)/tools:$PATH" 
export PATH="$(pwd)/tools/ADFRsuite_x86_64Linux_1.0/bin:$PATH"
```
