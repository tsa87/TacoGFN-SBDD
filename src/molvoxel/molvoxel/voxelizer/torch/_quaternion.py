import math
import numpy as np
import torch
from torch import FloatTensor

from typing import Tuple, Union

# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

Q = Union[
    Tuple[float, float, float, float],
    Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]
]

pi2 = 2 * math.pi


def random_quaternion() -> Q:
    u1, u2, u3 = np.random.rand(3)
    sq1 = math.sqrt(1 - u1)
    sqr = math.sqrt(u1)
    q0 = sq1 * math.sin(pi2 * u2)
    q1 = sq1 * math.cos(pi2 * u2)
    q2 = sqr * math.sin(pi2 * u3)
    q3 = sqr * math.cos(pi2 * u3)
    return q0, q1, q2, q3


def inverse_quaternion(quaternion: Q):
    return quaternion[0], quaternion[1] * -1, quaternion[2] * -1, quaternion[3] * -1


def multiply_quaternion(Q1: Q, Q2: Q) -> Q:
    q10, q11, q12, q13 = Q1
    q20, q21, q22, q23 = Q2
    q0 = q10 * q20 - q11 * q21 - q12 * q22 - q13 * q23
    q1 = q10 * q21 + q11 * q20 + q12 * q23 - q13 * q22
    q2 = q10 * q22 - q11 * q23 + q12 * q20 + q13 * q21
    q3 = q10 * q23 + q11 * q22 - q12 * q21 + q13 * q20
    return q0, q1, q2, q3


def position_to_quaternion(pos: FloatTensor) -> Q:
    x, y, z = torch.unbind(pos, dim=-1)
    q1, q2, q3 = x, y, z
    q0 = torch.zeros_like(q1)
    return q0, q1, q2, q3


def apply_quaternion(pos: FloatTensor, quaternion: Q) -> FloatTensor:
    pos_quaternion = position_to_quaternion(pos)
    qp = multiply_quaternion(quaternion, pos_quaternion)
    qpqinv = multiply_quaternion(qp, inverse_quaternion(quaternion))
    _, x, y, z = qpqinv
    return torch.stack([x, y, z], dim=-1)
