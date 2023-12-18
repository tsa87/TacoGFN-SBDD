
## Installation

1. Setup up Conda Environment
```bash
sudo apt-get install build-essential
sudo apt-get install libxau6

conda env create -f environment.yml
```

2. Download and unzip dataset
```bash
sh scripts/download_and_extract_crossdocked.sh
```
3. Install ADFR for Docking
```bash
chmod +x scripts/setup_adfr_suite.sh
./setup_adfr.sh
```

5. Export relevant paths
```bash
cd TacoGFN/
export PYTHONPATH=$(pwd)/src:$(pwd)/src/gflownet/E3Bind/src:$PYTHONPATH
export PATH="$(pwd)/ADFRsuite_x86_64Linux_1.0/bin:$PATH"
```

5. Run Active Learning
```bash
cd TacoGFN/src
python3 gflownet/tasks/run_pocket_cond.py
```