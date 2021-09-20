
"""Core brax structs and some conversion and slicing functions."""

from flax import struct
import jax
import jax.numpy as jnp
from xpdb_brax.physics import math

@struct.dataclass
class Q(object):
  """Coordinates: position and rotation.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
  """
  pos: jnp.ndarray
  rot: jnp.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return QP(self.pos, self.rot, o.vel, o.ang)
    elif isinstance(o, Q):
      return Q(self.pos + o.pos, self.rot + o.rot)
    elif isinstance(o, QP):
      return QP(self.pos + o.pos, self.rot + o.rot, o.vel, o.ang)
    else:
      raise ValueError("add only supported for P, Q, QP")

  def __neg__(self):
    return Q(-self.pos, - self.rot)

@struct.dataclass
class P(object):
  """Time derivatives: velocity and angular velocity.

  Attributes:
    vel: Velocity.
    ang: Angular velocity about center of mass.
  """
  vel: jnp.ndarray
  ang: jnp.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return P(self.vel + o.vel, self.ang + o.ang)
    elif isinstance(o, Q):
      return QP(o.pos, o.rot, self.vel, self.ang)
    elif isinstance(o, QP):
      return QP(o.pos, o.rot, self.vel + o.vel, self.ang + o.ang)
    else:
      raise ValueError("add only supported for P, Q, QP")

  def __mul__(self, o):
    return P(self.vel * o, self.ang * o)

  def __neg__(self):
    return P(-self.vel, -self.ang)


@struct.dataclass
class QP(object):
  """A coordinate and time derivative frame for a brax body.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
    vel: Velocity.
    ang: Angular velocity about center of mass.
  """
  pos: jnp.ndarray
  rot: jnp.ndarray
  vel: jnp.ndarray
  ang: jnp.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return QP(self.pos, self.rot, self.vel + o.vel, self.ang + o.ang)
    elif isinstance(o, Q):
      return QP(self.pos + o.pos, self.rot + o.rot, self.vel, self.ang)
    elif isinstance(o, QP):
      return QP(self.pos + o.pos, self.rot + o.rot, self.vel + o.vel,
                self.ang + o.ang)
    else:
      raise ValueError("add only supported for P, Q, QP")

  def __neg__(self):
    return QP(-self.pos, - self.rot, -self.vel, -self.ang)

  def __mul__(self, o):
    return QP(self.pos * o, self.rot * o, self.vel * o, self.ang * o)

  @classmethod
  def zero(cls):
    return cls(
        pos=jnp.zeros(3),
        rot=jnp.array([1., 0., 0., 0]),
        vel=jnp.zeros(3),
        ang=jnp.zeros(3))


def vec_to_np(v):
  return jnp.array([v.x, v.y, v.z])


def quat_to_np(q):
  return jnp.array([q.x, q.y, q.z, q.w])


def euler_to_quat(v):
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = jnp.cos(jnp.array([v.x, v.y, v.z]) * jnp.pi / 360)
  s1, s2, s3 = jnp.sin(jnp.array([v.x, v.y, v.z]) * jnp.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return jnp.array([x, y, z, w])


def take(objects, i: jnp.ndarray, axis=0):
  """Returns objects sliced by i."""
  flat_data, py_tree_def = jax.tree_flatten(objects)
  sliced_data = [jnp.take(k, i, axis=axis, mode="clip") for k in flat_data]
  return jax.tree_unflatten(py_tree_def, sliced_data)

