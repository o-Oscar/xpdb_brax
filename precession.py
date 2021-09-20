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

config.bodies.append(Box(1, [.1, .1, .1], frozen=True))
qp.append(QP(
    jnp.array([0., 0, 1]),
    jnp.array([1., 0, 0, 0]),
    jnp.array([0., 0, 0]),
    jnp.array([0., 0, 0]),
))
config.bodies.append(Box(1, [.3, 3, 3]))
qp.append(QP(
    jnp.array([-0.5, 0, 1]),
    jnp.array([1., 0, 0, 0]),
    jnp.array([0, 0, 0]),
    jnp.array([-3., 0, 0]),
))

config.joints.append(Joint ("ball", 0, 1, [0., 0, 0], [0.5, 0, 0], [1, 0, 0], 0))
config.gravity = [0, 0, -9.81]
config.dt = 0.01
config.substeps = 1

qp = jax.tree_multimap((lambda *args: jnp.stack(args)), *qp)
# print(type(a))
# exit()

system = System(config)
render = True

if render:
    xpdb_viewer = MyViewer()
    xpdb_viewer.init(config)

for i in range(3 if not render else 1000):
    qp = system.step(qp)
    pos, rot = qp.pos, qp.rot
    if render:
        xpdb_viewer.update(pos, rot)
        time.sleep(0.001)

if render:
    input()

print("success !!")