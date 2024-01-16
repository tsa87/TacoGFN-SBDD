# https://github.com/mattragoza/LiGAN
def write_grid_to_dx_file(dx_path, values, center, resolution):
    '''
    Write a grid with the provided values,
    center, and resolution to a .dx file.
    '''
    assert len(values.shape) == 3
    # assert values.shape[0] == values.shape[1] == values.shape[2]
    assert len(center) == 3

    size_x, size_y, size_z = values.shape
    center_x, center_y, center_z = center
    # origin = center - resolution * (size - 1) / 2.
    origin = (center_x - resolution * (size_x - 1) / 2.), \
             (center_y - resolution * (size_y - 1) / 2.), \
             (center_z - resolution * (size_z - 1) / 2.)

    lines = [
        'object 1 class gridpositions counts {:d} {:d} {:d}\n'.format(
            size_x, size_y, size_z
        ),
        'origin {:.5f} {:.5f} {:.5f}\n'.format(*origin),
        'delta {:.5f} 0 0\n'.format(resolution),
        'delta 0 {:.5f} 0\n'.format(resolution),
        'delta 0 0 {:.5f}\n'.format(resolution),
        'object 2 class gridconnections counts {:d} {:d} {:d}\n'.format(
            size_x, size_y, size_z
        ),
        'object 3 class array type double rank 0 items '
        + '[ {:d} ] data follows\n'.format(size_x * size_y * size_z),
    ]
    line = ''
    values = values.reshape(-1).tolist()
    for i, value in enumerate(values):
        if i % 3 == 2:
            line += f'{value:.5f}\n'
        else:
            line += f'{value:.5f} '
    lines.append(line)

    with open(dx_path, 'w') as f:
        f.write(''.join(lines))
