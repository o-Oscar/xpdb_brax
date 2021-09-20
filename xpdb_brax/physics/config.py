
from flax import struct
import jax.numpy as jnp
import numpy as np

class Config (object):
    def __init__ (self):
        self.dt = 0.01
        self.substeps = 100

        # self.gravity = [0, 0, -9.81]
        self.gravity = [0., 0., 0.]

        self.bodies = []
        self.joints = []
        self.collision_pairs = []

    def get_active_bodies (self):
        to_return = [0. if body.frozen else 1. for body in self.bodies]
        return jnp.expand_dims(jnp.array(to_return), axis=1)
    
    def get_gravity (self):
        return jnp.array(self.gravity)
    
    def get_inertia (self):
        to_return = [body.inertia for body in self.bodies]
        return jnp.array(to_return)

    def get_inertia_inv (self):
        to_return = [body.inv_inertia for body in self.bodies]
        return jnp.array(to_return)
    
    def get_n_bodies (self):
        return len(self.bodies)

@struct.dataclass
class FlaxBody:
    mass: jnp.ndarray
    inertia: jnp.ndarray
    inv_inertia: jnp.ndarray
    is_active: jnp.ndarray



# we assume the center of mass to be at 0,0
class Body:
    def __init__ (self, mass, inertia, frozen=False):
        self.mass = mass
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.frozen = frozen
    
    def get_active (self):
        return 0. if self.frozen else 1.
    
    def to_flax (self):
        return FlaxBody(self.mass, self.inertia, self.inv_inertia, self.get_active())

class Box (Body): # axis-aligned box
    def __init__ (self, mass, size, frozen=False): # size is the full size of the box

        self.size = size
        diag_inertia = 1/12 * mass * (np.sum(np.square(size)) - np.square(size))

        super().__init__(mass, np.diag(diag_inertia), frozen)
        
class Sphere (Body): # axis-aligned box
    def __init__ (self, mass, radius, frozen=False): # size is the full size of the box

        self.radius = radius
        inertia = 2/5 * mass * radius * radius

        super().__init__(mass, np.eye(3) * inertia, frozen)
        

class Plane (Body): # xy infinit plane
    def __init__(self):
        super().__init__(1, np.diag([1]*3), frozen=True)


from xpdb_brax.physics.joints import BallJoint, OrientationJoint, RevoluteJoint, MyJoint

import jax

class Joint:
    def __init__ (self, type, idx_p, idx_c, off_p, off_c, axis, compliance, actuated=False):
        self.type = type

        self.idx_p = idx_p
        self.idx_c = idx_c

        self.off_p = off_p
        self.off_c = off_c

        self.axis = axis
        self.compliance = compliance
    
        self.actuated = actuated

    def to_flax (self):
        if self.type == "ball":
            cls = BallJoint
        elif self.type == "orientation":
            cls = OrientationJoint
        elif self.type == "revolute":
            cls = RevoluteJoint
        elif self.type == "myjoint":
            cls = MyJoint
        else:
            raise NotImplementedError("The joint {} is not implemented".format(self.type))
        
        return cls(
            self.idx_p,
            self.idx_c,

            jnp.array(self.off_p),
            jnp.array(self.off_c),

            jnp.array(self.axis),
            self.compliance
        )