from test_time_numpy import main

if __name__ == '__main__' :
    from molvoxel.voxelizer.numba import Voxelizer
    resolution = 0.5
    dimension = 48
    voxelizer = Voxelizer(resolution, dimension)
    main(voxelizer)
