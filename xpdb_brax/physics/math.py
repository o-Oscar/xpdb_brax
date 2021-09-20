
import jax
import jax.numpy as jnp


matrix_vector_dot = lambda a, b: jnp.sum(a * jnp.expand_dims(b, axis=0), axis=1)
vector_dot = lambda a, b: jnp.sum(a * b, axis=0, keepdims=True)

def safe_normalize (vec):
	norm = jnp.linalg.norm(vec)

	is_zero = jnp.allclose(norm, 0.)
	safe_norm = jnp.where(is_zero, jnp.ones_like(norm), norm)  # replace d with ones if is_zero

	return vec/safe_norm, norm, is_zero


def to_world(qp, rpos: jnp.ndarray):
  """Returns world information about a point relative to a part.

  Args:
    qp: Part from which to offset point.
    rpos: Point relative to center of mass of part.

  Returns:
    A 2-tuple containing:
      * World-space coordinates of rpos
      * World-space velocity of rpos
  """
  rpos_off = rotate(rpos, qp.rot)
  rvel = jnp.cross(qp.ang, rpos_off)
  return (qp.pos + rpos_off, qp.vel + rvel)

def qmult(u, v):
	"""Multiplies two quaternions.

	Args:
	u: jnp.ndarray (4) (w,x,y,z)
	v: jnp.ndarray (4) (w,x,y,z)

	Returns:
	A quaternion u*v.
	"""
	return jnp.stack([
		u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
		u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
		u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
		u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
	], axis=0)

def qmult_old(u, v):
	"""Multiplies two quaternions.

	Args:
	u: jnp.ndarray (4) (w,x,y,z)
	v: jnp.ndarray (4) (w,x,y,z)

	Returns:
	A quaternion u*v.
	"""
	return jnp.stack([
		u[:,0] * v[:,0] - u[:,1] * v[:,1] - u[:,2] * v[:,2] - u[:,3] * v[:,3],
		u[:,0] * v[:,1] + u[:,1] * v[:,0] + u[:,2] * v[:,3] - u[:,3] * v[:,2],
		u[:,0] * v[:,2] - u[:,1] * v[:,3] + u[:,2] * v[:,0] + u[:,3] * v[:,1],
		u[:,0] * v[:,3] + u[:,1] * v[:,2] - u[:,2] * v[:,1] + u[:,3] * v[:,0],
	], axis=1)

def rotate(vec: jnp.ndarray, quat: jnp.ndarray):
	"""Rotates a vector vec by a unit quaternion quat.

	Args:
	vec: jnp.ndarray (?,3)
	quat: jnp.ndarray (?,4) (w,x,y,z)

	Returns:
	A jnp.ndarry(3) containing vec rotated by quat.
	"""

	u = quat[1:]
	s = quat[0:1]
	return 2 * (vector_dot(u, vec) * u) + (s * s - vector_dot(u, u)) * vec + 2 * s * jnp.cross(u, vec)


def inv_rotate(vec: jnp.ndarray, quat: jnp.ndarray):
	"""Rotates a vector by the inverse of a unit quaternion.

	Args:
	vec: jnp.ndarray
	quat: jnp.ndarray

	Returns:
	A vector rotated by quat^{-1}
	"""
	u = -1. * quat[1:]
	s = quat[0:1]
	return 2 * (vector_dot(u, vec) * u) + (s * s - vector_dot(u, u)) * vec + 2 * s * jnp.cross(u, vec)

def qinv (quat: jnp.ndarray):
	return quat * jnp.array([1, -1, -1, -1])

def skew (a):
	return jnp.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


