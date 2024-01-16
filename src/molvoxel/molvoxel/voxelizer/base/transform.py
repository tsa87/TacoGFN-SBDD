from abc import ABCMeta, abstractmethod
from numpy.typing import ArrayLike


class BaseT(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, coords: ArrayLike, center: ArrayLike) -> ArrayLike:
        pass

    @classmethod
    @abstractmethod
    def create(cls, random_translation: float = 0.0, random_rotation: bool = False):
        pass


class BaseRandomTransform(metaclass=ABCMeta):
    class_T = BaseT

    def __init__(
        self,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ):
        self.random_translation = random_translation
        self.random_rotation = random_rotation

    @abstractmethod
    def forward(self, coords: ArrayLike, center: ArrayLike) -> ArrayLike:
        pass

    def get_transform(self) -> BaseT:
        return self.class_T.create(self.random_translation, self.random_rotation)
