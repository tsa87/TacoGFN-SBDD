sudo apt-get -y install build-essential
sudo apt-get -y install libxau6
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11
sudo apt-get install -y libsm6 libxrender1 libfontconfig1

conda install -n base conda-libmamba-solver --yes
conda config --set solver libmamba
conda env create -f environment.yml

pip install gdown
gdown --id 1Mdg3eIhXube6TpctjPBUN5JVFDvkxNRO
tar -xvzf tacogfn_data.tar.gz