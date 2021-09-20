


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

@struct.dataclass
class Actuator (object):

	idx_p : int
	idx_c : int

	# body_p : FlaxBody
	# body_c : FlaxBody

	off_p: jnp.ndarray
	off_c: jnp.ndarray

	axis: jnp.ndarray

	def apply (self, qp, bodies, h):
		qp_p = take(qp, self.idx_p)
		qp_c = take(qp, self.idx_c)

		body_p = take(bodies, self.idx_p)
		body_c = take(bodies, self.idx_c)

		dang_p, dang_c = self._apply(qp_p, qp_c, body_p, body_c)
		# qp_p, qp_c = self._apply(qp_p, qp_c, body_p, body_c)
		dang = jax.ops.segment_sum(dang_p, self.idx_p, qp.pos.shape[0])
		dang = dang + jax.ops.segment_sum(dang_c, self.idx_c, qp.pos.shape[0])

		qp = QP(
			qp.pos,
			qp.rot,
			qp.vel,
			qp.ang + dang * h,
		)
		
		return qp

	@jax.vmap
	def _apply (self, qp_p, qp_c, body_p, body_c):
		delta_q = math.qmult(qp_c.rot, math.qinv(qp_p.rot))
		alpha = jnp.arctan2(delta_q[1], delta_q[0]) + 0.5
		alpha_dot = jnp.sum(self.axis * (qp_c.ang - qp_p.ang))

		torque = - alpha * 70 - alpha_dot * 7
		torque = self.axis * torque
		print(torque)

		dang_p = jnp.matmul(body_p.inv_inertia, -torque)
		dang_c = jnp.matmul(body_c.inv_inertia, torque)

		return math.rotate(dang_p, qp_p.rot), math.rotate(dang_c, qp_c.rot)