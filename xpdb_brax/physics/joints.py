
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
class Joint (object):

	idx_p : int
	idx_c : int

	# body_p : FlaxBody
	# body_c : FlaxBody

	off_p: jnp.ndarray
	off_c: jnp.ndarray

	axis: jnp.ndarray

	compliance: jnp.ndarray

	def apply (self, qp, bodies):
		qp_p = take(qp, self.idx_p)
		qp_c = take(qp, self.idx_c)

		body_p = take(bodies, self.idx_p)
		body_c = take(bodies, self.idx_c)

		qp_p, qp_c = self._apply(qp_p, qp_c, body_p, body_c)
		# qp_p, qp_c = self._apply(qp_p, qp_c, body_p, body_c)

		qp = set(qp, self.idx_p, qp_p)
		qp = set(qp, self.idx_c, qp_c)
		
		return qp

	@functools.partial(jax.jit, static_argnums=(0,))
	def _apply (self, qp_p, qp_c):
		# raise RuntimeError(f"A subclass of Joint should implement the apply function.")
		pass

def solve_delta_pos (qp_p, qp_c, body_p, body_c, off_p, off_c):
	rp = math.rotate(off_p, qp_p.rot)
	rc = math.rotate(off_c, qp_c.rot)

	delta_x = qp_c.pos + rc - qp_p.pos - rp
	n, c, _ = math.safe_normalize(delta_x)
	
	rp_n = jnp.cross(off_p, math.inv_rotate(n, qp_p.rot))
	ip = (math.vector_dot(rp_n, math.matrix_vector_dot(body_p.inv_inertia, rp_n)))
	wp = 1. / body_p.mass + (math.vector_dot(rp_n, math.matrix_vector_dot(body_p.inv_inertia, rp_n)))
	wp = wp * body_p.is_active

	rc_n = jnp.cross(off_c, math.inv_rotate(n, qp_c.rot))
	ic = (math.vector_dot(rc_n, math.matrix_vector_dot(body_c.inv_inertia, rc_n)))
	wc = 1. / body_c.mass + (math.vector_dot(rc_n, math.matrix_vector_dot(body_c.inv_inertia, rc_n)))
	wc = wc * body_c.is_active

	delta_lamb = -c/(wp + wc)
	p = delta_lamb * n

	pp = math.inv_rotate(p, qp_p.rot)
	pc = math.inv_rotate(p, qp_c.rot)
	temp_p = math.matrix_vector_dot(body_p.inv_inertia, jnp.cross(off_p, pp))
	temp_c = math.matrix_vector_dot(body_c.inv_inertia, jnp.cross(off_c, pc))
	drot_p = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(temp_p, qp_p.rot))), qp_p.rot)
	drot_c = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(temp_c, qp_c.rot))), qp_c.rot)# * 0.3

	# drot_p = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(jnp.cross(off_p, pp), qp_p.rot))), qp_p.rot) * ip
	# drot_c = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(jnp.cross(off_c, pc), qp_c.rot))), qp_c.rot) * ic
	
	dqp = Q(p/body_p.mass * body_p.is_active, drot_p * body_p.is_active)
	dqc = Q(p/body_c.mass * body_c.is_active, drot_c * body_c.is_active)

	return qp_p + (-dqp), qp_c + dqc

def solve_delta_rot (qp_p, qp_c, body_p, body_c, delta_q):
	n, theta, is_zero = math.safe_normalize(delta_q)
		
	np = math.inv_rotate(n, qp_p.rot)
	wp = math.vector_dot(np, math.matrix_vector_dot(body_p.inv_inertia, np))
	wp = wp * body_p.is_active

	nc = math.inv_rotate(n, qp_c.rot)
	wc = math.vector_dot(nc, math.matrix_vector_dot(body_c.inv_inertia, nc))
	wc = wc * body_c.is_active

	delta_lamb = -theta/(wp + wc + 1e-5)
	safe_delta_lamb = jnp.where(is_zero, jnp.zeros_like(delta_lamb), delta_lamb)
	p = safe_delta_lamb * n

	pp = math.inv_rotate(p, qp_p.rot)
	pc = math.inv_rotate(p, qp_c.rot)
	temp_p = math.matrix_vector_dot(body_p.inv_inertia, pp)
	temp_c = math.matrix_vector_dot(body_c.inv_inertia, pc)
	drot_p = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(temp_p, qp_p.rot))), qp_p.rot) # to test
	drot_c = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(temp_c, qp_c.rot))), qp_c.rot)
	# drot_p = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), p * wp)), qp_p.rot)
	# drot_c = .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), p * wc)), qp_c.rot)
	
	dqp = Q(jnp.zeros((3,)), drot_p * body_p.is_active)
	dqc = Q(jnp.zeros((3,)), drot_c * body_c.is_active)

	return qp_p + (-dqp), qp_c + dqc

@struct.dataclass
class BallJoint (Joint):

	def _apply (self, qp_p, qp_c, body_p, body_c):

		qp_p, qp_c = solve_delta_pos (qp_p, qp_c, body_p, body_c, self.off_p, self.off_c)

		qp_p = QP(
			qp_p.pos,
			qp_p.rot / jnp.sqrt(jnp.sum(jnp.square(qp_p.rot), axis=0, keepdims=True)),
			qp_p.vel,
			qp_p.ang
		)
		return qp_p, qp_c


@struct.dataclass
class OrientationJoint (Joint):

	def _apply (self, qp_p, qp_c, body_p, body_c):

		delta_q = 2 * math.qmult(qp_c.rot, math.qinv(qp_p.rot))[1:]
		
		qp_p, qp_c = solve_delta_rot(qp_p, qp_c, body_p, body_c, delta_q)

		return qp_p, qp_c

@struct.dataclass
class RevoluteJoint (Joint):

	def _apply (self, qp_p, qp_c, body_p, body_c):
		
		ap = math.rotate(self.axis, qp_p.rot)
		ac = math.rotate(self.axis, qp_c.rot)
		delta_q = jnp.cross(ap, ac)

		qp_p, qp_c = solve_delta_rot(qp_p, qp_c, body_p, body_c, delta_q)
		

		qp_p = QP(
			qp_p.pos,
			qp_p.rot / jnp.sqrt(jnp.sum(jnp.square(qp_p.rot), axis=0, keepdims=True)),
			qp_p.vel,
			qp_p.ang
		)
		qp_c = QP(
			qp_c.pos,
			qp_c.rot / jnp.sqrt(jnp.sum(jnp.square(qp_c.rot), axis=0, keepdims=True)),
			qp_c.vel,
			qp_c.ang
		)

		qp_p, qp_c = solve_delta_pos (qp_p, qp_c, body_p, body_c, self.off_p, self.off_c)

		qp_p = QP(
			qp_p.pos,
			qp_p.rot / jnp.sqrt(jnp.sum(jnp.square(qp_p.rot), axis=0, keepdims=True)),
			qp_p.vel,
			qp_p.ang
		)
		qp_c = QP(
			qp_c.pos,
			qp_c.rot / jnp.sqrt(jnp.sum(jnp.square(qp_c.rot), axis=0, keepdims=True)),
			qp_c.vel,
			qp_c.ang
		)

		
		return qp_p, qp_c


@struct.dataclass
class MyJoint (Joint):

	def _apply (self, qp_p, qp_c, body_p, body_c):
		
		ap = math.rotate(self.axis, qp_p.rot)
		ac = math.rotate(self.axis, qp_c.rot)
		delta_q = jnp.cross(ap, ac)


		I_inv = jnp.zeros((12,12))
		for i in range(3):
			I_inv = jax.ops.index_update(I_inv, (i,i), 1./body_p.mass * body_p.is_active)
			I_inv = jax.ops.index_update(I_inv, (i+6,i+6), 1./body_c.mass * body_c.is_active)
			I_inv = jax.ops.index_update(I_inv, jax.ops.index[3:6,3:6], body_p.inv_inertia * body_p.is_active)
			I_inv = jax.ops.index_update(I_inv, jax.ops.index[9:,9:], body_c.inv_inertia * body_c.is_active)


		y = jnp.array([0, 1, 0])
		z = jnp.array([0, 0, 1])
		
		rp = math.rotate(self.off_p, qp_p.rot)
		rc = math.rotate(self.off_c, qp_c.rot)


		C = jnp.zeros((5,12))
		for i in range(3):
			C = jax.ops.index_update(C, (i,i), 1)
			C = jax.ops.index_update(C, (i,i+6), -1)
		
		C = jax.ops.index_update(C, jax.ops.index[0:3,3:6], -math.skew(rp))
		C = jax.ops.index_update(C, jax.ops.index[0:3,9:12], math.skew(rc))
		
		C = jax.ops.index_update(C, jax.ops.index[3,3:6], y)
		C = jax.ops.index_update(C, jax.ops.index[3+1,3:6], z)
		C = jax.ops.index_update(C, jax.ops.index[3,9:12], math.inv_rotate(math.rotate(y, qp_p.rot), qp_c.rot))
		C = jax.ops.index_update(C, jax.ops.index[3+1,9:12], math.inv_rotate(math.rotate(z, qp_p.rot), qp_c.rot))


		delta_x = qp_c.pos + rc - qp_p.pos - rp
		ap = math.rotate(self.axis, qp_p.rot)
		ac = math.rotate(self.axis, qp_c.rot)
		delta_q = -jnp.cross(self.axis, math.inv_rotate(math.rotate(self.axis, qp_c.rot), qp_p.rot))

		b = jnp.zeros((5,))
		b = jax.ops.index_update(b, jax.ops.index[0:3], delta_x)
		b = jax.ops.index_update(b, jax.ops.index[3:5], delta_q[1:3])
		

		vp = I_inv @ C.transpose() @ jnp.linalg.inv(C @ I_inv.transpose() @ C.transpose()) @ b
		# drot_p = jnp.clip(vp[3:6], -0.1, 0.1)
		drot_p = vp[3:6]
		qp_p = QP(
			qp_p.pos + vp[0:3],
			qp_p.rot + .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(drot_p, qp_p.rot))), qp_p.rot),
			qp_p.vel,
			qp_p.ang
		)
		# drot_c = jnp.clip(vp[9:12], -0.1, 0.1)
		drot_c = vp[9:12]
		qp_c = QP(
			qp_c.pos + vp [6:9],
			qp_c.rot + .5 * math.qmult(jnp.concatenate((jnp.zeros((1,)), math.rotate(drot_c, qp_c.rot))), qp_c.rot),
			qp_c.vel,
			qp_c.ang
		)

		qp_p = QP(
			qp_p.pos,
			qp_p.rot / jnp.sqrt(jnp.sum(jnp.square(qp_p.rot), axis=0, keepdims=True)),
			qp_p.vel,
			qp_p.ang
		)
		qp_c = QP(
			qp_c.pos,
			qp_c.rot / jnp.sqrt(jnp.sum(jnp.square(qp_c.rot), axis=0, keepdims=True)),
			qp_c.vel,
			qp_c.ang
		)

		
		return qp_p, qp_c