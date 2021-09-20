
from xpdb_brax.physics.base import QP
import jax
import jax.numpy as jnp
import functools

from xpdb_brax.physics.actuator import Actuator

from xpdb_brax.physics import math

# matrix_vector_dot = lambda a, b: jnp.tensordot(a, b, axes=([2],[1]))

# straight from page 76 of https://box2d.org/files/ErinCatto_NumericalMethods_GDC2015.pdf
def solve_gyroscopic (rot, I, ang, h):
	ang_loc = math.inv_rotate(ang, rot)
	f = h * jnp.cross(ang_loc, math.matrix_vector_dot(I, ang_loc))
	J = I + h * (jnp.matmul(math.skew(ang), I) - math.skew(math.matrix_vector_dot(I, ang)))
	new_ang_loc = ang_loc - jnp.linalg.solve(J, f)
	return math.rotate(new_ang_loc, rot)

def euler_step (init_qp, body, config, h):
	# h =	 0
	# def euler_step (init_qp, config, h):
	pos_prev, rot_prev, vel, ang = init_qp.pos, init_qp.rot, init_qp.vel, init_qp.ang

	vel = (vel + h * config.get_gravity()) * body.is_active
	pos = pos_prev + h * vel
	
	# semi-implicit euler
	# ang_loc = math.inv_rotate(ang, rot_prev)
	# ang = ang - math.rotate(h * math.matrix_vector_dot(body.inv_inertia, jnp.cross(ang_loc, math.matrix_vector_dot(body.inertia, ang_loc))), rot_prev)
	# implicit euler
	ang = solve_gyroscopic(rot_prev, body.inertia, ang, h)

	# TODO : pd control
	# ang = 

	ang = ang * body.is_active
	rot = rot_prev + h * .5 * math.qmult(jnp.concatenate([jnp.zeros((1,)), ang], axis=0), rot_prev)
	rot = rot / jnp.sqrt(jnp.sum(jnp.square(rot)))

	return QP(pos, rot, vel, ang)

def velocity_step (qp, prev_qp, h):
	
	vel = (qp.pos - prev_qp.pos) / h
	delta_q = math.qmult(qp.rot, math.qinv(prev_qp.rot))
	ang = 2 * delta_q[1:] / h
	ang = ang * jnp.where(delta_q[0:1] > 0, 1, -1)

	return QP(qp.pos, qp.rot, vel, ang)
	

class System:
	def __init__ (self, config):
		self.config = config

		bodies = [body.to_flax() for body in self.config.bodies]
		self.flax_bodies = jax.tree_multimap((lambda *args: jnp.stack(args)), *bodies)

		# if len(config.joints) > 0:
		# 	joints = [joint.to_flax() for joint in self.config.joints]
		# 	self.flax_joints = jax.tree_multimap((lambda *args: jnp.stack(args)), *joints)
		self.flax_joints = [joint.to_flax() for joint in self.config.joints]

		self.actuators = [Actuator(joint.idx_p, joint.idx_c, joint.off_p, joint.off_c, jnp.array(joint.axis)) for joint in self.config.joints if joint.actuated]
		if len(self.actuators) > 0:
			self.flax_actuators = jax.tree_multimap((lambda *args: jnp.stack(args)), *self.actuators)
		
		self.flax_collision_pairs = [collision_pair.to_flax() for collision_pair in self.config.collision_pairs]

		h = self.config.dt / self.config.substeps
		self.euler = jax.vmap(functools.partial(euler_step, config=self.config, h = h))
		self.velocity = jax.vmap(functools.partial(velocity_step, h = h))

	# @functools.partial(jax.jit, static_argnums=(0,))
	def step (self, qp: QP):
		
		qp, _ = jax.lax.scan(self.substep, qp, (), self.config.substeps)
		# qp, _ = self.substep(qp, ())

		return qp

	@functools.partial(jax.jit, static_argnums=(0,))
	def substep(self, prev_qp, _):
		# Euler update
		qp = prev_qp

		h = self.config.dt / self.config.substeps
		if len(self.actuators) > 0:
			qp = self.flax_actuators.apply(qp, self.flax_bodies, h)
		
		qp = self.euler(qp, self.flax_bodies)


		# Constrain resolution
		# if len(self.config.joints) > 0:
		# 	qp = self.flax_joints.apply(qp, self.flax_bodies)
		for joint in self.flax_joints:
			qp = joint.apply(qp, self.flax_bodies)
		
		for collision_pair in self.flax_collision_pairs:
			qp = collision_pair.apply(qp, prev_qp, self.flax_bodies)

		# Velocity update
		qp = self.velocity (qp, prev_qp)
		prev_qp = qp

		return qp, ()
