import numpy as np

from typing import Tuple, Union, Optional
from numpy.typing import NDArray
from ._quaternion import random_quaternion, apply_quaternion, Q
from ..base.transform import BaseRandomTransform, BaseT

NDArrayFloat = NDArray[np.float_]
class T(BaseT) :
    def __init__(self, translation: Optional[NDArrayFloat], quaternion: Optional[Q]) :
        self.translation = translation
        self.quaternion = quaternion

    def __call__(self, coords, center) :
        return do_transform(coords, center, self.translation, self.quaternion)

    @classmethod
    def create(
        cls,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ) :
        if random_translation > 0.0:
            translation = np.random.uniform(-random_translation, random_translation, size=(1,3)).astype(np.float32)
        else :
            translation = None
        if random_rotation :
            quaternion = random_quaternion()
        else :
            quaternion = None
        return cls(translation, quaternion)

class RandomTransform(BaseRandomTransform) :
    class_T=T
    def forward(self, coords: NDArrayFloat, center: Optional[NDArrayFloat]) -> NDArrayFloat:
        return do_random_transform(coords, center, self.random_translation, self.random_rotation)

def do_transform(
    coords: NDArrayFloat,
    center: Optional[NDArrayFloat] = None,
    translation: Optional[NDArrayFloat] = None,
    quaternion: Optional[Q] = None,
) -> NDArrayFloat:
    if quaternion is not None :
        if center is not None :
            center = center.reshape(1,3)
            coords = apply_quaternion(coords - center, quaternion)
            coords += center
        else :
            coords = apply_quaternion(coords, quaternion)
        if translation is not None :
            coords += translation
    if translation is not None :
        coords = coords + translation
    return coords

def do_random_transform(
    coords: NDArrayFloat,
    center: Optional[NDArrayFloat] = None,
    random_translation: Optional[float] = 0.0,
    random_rotation: bool = False,
) -> NDArray:

    if random_rotation :
        quaternion = random_quaternion()
    else :
        quaternion = None

    if (random_translation is not None) and (random_translation > 0.0) :
        translation = np.random.uniform(-random_translation, random_translation, size=(1,3)).astype(np.float32)
    else :
        translation = None
    
    return do_transform(coords, center, translation, quaternion)
