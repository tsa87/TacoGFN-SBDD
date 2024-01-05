wget -O dataset/ http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz


# TODO move into this folder, unzip into crossdock
# and flatten the folder structure
tar -xzvf dataset/CrossDocked2020_v1.3.tgz --wildcards '*.pdb'