

from flax import struct
import jax
import jax.numpy as jnp
import functools

from xpdb_brax.physics.config import FlaxBody

from xpdb_brax.physics.base import QP, Q
from xpdb_brax.physics import math


def take(objects, i: jnp.ndarray, axis=0):
	"""Returns objects sliced by i."""
	flat_data, py_tree_def = jax.tree_flatten(objects)
	sliced_data = [jnp.take(k, i, axis=axis, mode="clip") for k in flat_data]
	return jax.tree_unflatten(py_tree_def, sliced_data)

def set(full_qp, idx, qp):
	"""Returns objects sliced by i."""
	return QP(
		jax.ops.index_update(full_qp.pos, idx, qp.pos),
		jax.ops.index_update(full_qp.rot, idx, qp.rot),
		jax.ops.index_update(full_qp.vel, idx, qp.vel),
		jax.ops.index_update(full_qp.ang, idx, qp.ang),
		)

class PlaneSphere:
	def __init__ (self, idx_p, idx_c, offset, radius):
		self.idx_p = idx_p
		self.idx_c = idx_c

		self.offset = offset

		self.radius = radius


	def to_flax (self):
		return FlaxPlaneSphere(
			self.idx_p,
			self.idx_c,
			jnp.array(self.offset),
			self.radius
		)


@struct.dataclass
class FlaxPlaneSphere:
	idx_p : int
	idx_c : int

	offset : jnp.ndarray

	radius : float

	def apply (self, qp, qp_prev, bodies):
		qp_p = take(qp, self.idx_p)
		qp_c = take(qp, self.idx_c)
		qp_c_prev = take(qp_prev, self.idx_c)

		body_p = take(bodies, self.idx_p)
		body_c = take(bodies, self.idx_c)

		qp_p, qp_c = self._apply(qp_p, qp_c, qp_c_prev, body_p, body_c)
		# qp_p, qp_c = self._apply(qp_p, qp_c, body_p, body_c)

		qp = set(qp, self.idx_p, qp_p)
		qp = set(qp, self.idx_c, qp_c)
		
		return qp
	
	def _apply (self, qp_p, qp_s, qp_s_prev, body_p, body_s):

		off_world = math.rotate(self.offset, qp_s.rot)
		s_pos = qp_s.pos + off_world

		pen_depth = qp_p.pos[2]- s_pos[2]+self.radius
		is_pen = jnp.where(jnp.greater(pen_depth, 0), 1, 0)

		# penetration_vec = jnp.array([0, 0, ])
		# n, penetration = math.safe_normalize(penetration_vec)


		delta_x = -jnp.array([0, 0, qp_p.pos[2]- s_pos[2]+self.radius])
		n, c, _ = math.safe_normalize(delta_x)
		
		off_s = jnp.array([0, 0, -self.radius]) + off_world
		rs_n = math.inv_rotate(jnp.cross(off_s, n), qp_s.rot)
		ic = (math.vector_dot(rs_n, math.matrix_vector_dot(body_s.inv_inertia, rs_n)))
		ws = 1. / body_s.mass + (math.vector_dot(rs_n, math.matrix_vector_dot(body_s.inv_inertia, rs_n)))

		delta_lamb = -c/ws
		p = delta_lamb * n

		ps = math.inv_rotate(jnp.cross(off_s, p), qp_s.rot)
		temp_s = math.matrix_vector_dot(body_s.inv_inertia, ps)
		drot_s = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(temp_s, qp_s.rot))), qp_s.rot)# * 0.3

		dqs = Q(p/body_s.mass * body_s.is_active * is_pen, drot_s * body_s.is_active * is_pen)

		return qp_p, qp_s + dqs
