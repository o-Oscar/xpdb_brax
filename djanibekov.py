

import xpdb_brax
from xpdb_brax.physics.config import Config, Box, Plane, Joint
from xpdb_brax.viewer.server import MyViewer
from xpdb_brax.physics.base import QP
from xpdb_brax.physics.system import System

import numpy as np
import jax.numpy as jnp
import jax

import time

config = Config()

qp = []

config.bodies.append(Box(1, [.1, .3, 1]))
qp.append(QP(
    jnp.array([0., 0, 1]),
    jnp.array([1., 0, 0, 0]),
    jnp.array([0., 0., 0.]),
    jnp.array([0.1, 10., 0.]),
))

config.gravity = [0, 0, 0]
config.dt = 0.01
config.substeps = 100

qp = jax.tree_multimap((lambda *args: jnp.stack(args)), *qp)

system = System(config)
render = True

if render:
    xpdb_viewer = MyViewer()
    xpdb_viewer.init(config)


from xpdb_brax.physics import math

all_moments = []
e = []

for i in range(300 if not render else 300):
    qp = system.step(qp)
    pos, rot = qp.pos, qp.rot
    if render:
        xpdb_viewer.update(pos, rot)
        time.sleep(0.001)
    
    moment = math.rotate(math.matrix_vector_dot(system.flax_bodies.inertia[0], math.inv_rotate(qp.ang[0], qp.rot[0])), qp.rot[0])
    all_moments.append(np.asarray(moment))
    e.append(np.sum(moment * qp.ang[0]))

import matplotlib
matplotlib.use('Qt5Agg', force=True)
# print(matplotlib.rcsetup.interactive_bk)
import matplotlib.pyplot as plt

all_moments = np.asarray(all_moments)
plt.plot(all_moments)
plt.plot(e)
plt.show()

if render:
    input()

print("success !!")