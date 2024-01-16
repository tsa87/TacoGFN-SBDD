from setuptools import setup

setup(
    name='molvoxel',
    version='0.1.0',
    description='MolVoxel: Python Library to Voxelize 3D Molecular Structure',
    author='Seonghwan Seo',
    author_email='shwan0106@gmail.com',
    url='https://github.com/SeonghwanSeo/molvoxel',
    packages=['molvoxel/'],
    install_requires=['numpy'],
    extras_require = {
            'numpy': ['scipy'],
            'numba': ['numba'],
            'torch': ['torch'],
            'rdkit': ['rdkit-pypi'],
        }
)
