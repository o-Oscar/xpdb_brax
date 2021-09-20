
from jax.config import config; config.update("jax_enable_x64", True)


import xpdb_brax
from xpdb_brax.physics.config import Config, Box, Plane, Sphere, Joint
from xpdb_brax.physics.collision_pairs import PlaneSphere
from xpdb_brax.viewer.server import MyViewer
from xpdb_brax.physics.base import QP
from xpdb_brax.physics.system import System

import numpy as np
import jax.numpy as jnp
import jax

import time

config = Config()

qp = []

config.bodies.append(Box(mass=1, size=[0.1, .1, .1], frozen=True))
qp.append(QP(
    jnp.array([0, 0, 2]),
    jnp.array([1, 0, 0, 0]),
    jnp.array([0, 0, 0]),
    jnp.array([0, 0, 0]),
))

config.bodies.append(Box(mass=1, size=[0.05, .1, .5], frozen=False))
qp.append(QP(
    jnp.array([0., 0.25, 2.]),
    # jnp.array([.9, 0, 0.4, 0]),
    jnp.array([.7, .7, 0, 0.]),
    jnp.array([0., 0, 0]),
    jnp.array([0., 0, 0]),
))

config.bodies.append(Box(mass=1, size=[0.05, .1, .5], frozen=False))
qp.append(QP(
    jnp.array([0., .75, 2]),
    # jnp.array([.9, 0, 0.4, 0]),
    jnp.array([.7, .7, 0, 0.]),
    jnp.array([0., 0, 0]),
    jnp.array([0., 0, 0]),
))

config.bodies.append(Box(mass=1, size=[0.05, .1, .5], frozen=False))
qp.append(QP(
    jnp.array([0., .25+.5*2, 2]),
    # jnp.array([.9, 0, 0.4, 0]),
    jnp.array([.7, .7, 0, 0.]),
    jnp.array([0., 0, 0]),
    jnp.array([0., 0, 0]),
))

# config.bodies.append(Box(mass=1, size=[0.05, .1, .5], frozen=False))
# qp.append(QP(
#     jnp.array([0., .25+.5*3, 2]),
#     # jnp.array([.9, 0, 0.4, 0]),
#     jnp.array([.7, .7, 0, 0.]),
#     jnp.array([0., 0, 0]),
#     jnp.array([0., 0, 0]),
# ))

config.gravity = [0, 0, -9.81]
config.dt = 1/30
config.substeps = 100

config.joints.append(Joint("myjoint", 0, 1, [0, 0, 0], [0, 0, 0.25], [1, 0, 0], 0))
config.joints.append(Joint("myjoint", 1, 2, [0, 0, -0.25], [0, 0, 0.25], [1, 0, 0], 0))
config.joints.append(Joint("myjoint", 2, 3, [0, 0, -0.25], [0, 0, 0.25], [1, 0, 0], 0))
# config.joints.append(Joint("myjoint", 3, 4, [0, 0, -0.25], [0, 0, 0.25], [1, 0, 0], 0))

qp = jax.tree_multimap((lambda *args: jnp.stack(args)), *qp)

system = System(config)
render = True

if render:
    xpdb_viewer = MyViewer()
    xpdb_viewer.init(config)


last_time = time.time()
for i in range(2 if not render else 300):
    qp = system.step(qp)
    if render:
        xpdb_viewer.update(qp.pos, qp.rot)
        while time.time() - last_time < 1/30:
            pass
        last_time = time.time()
    



import matplotlib
matplotlib.use('Qt5Agg', force=True)
# print(matplotlib.rcsetup.interactive_bk)
import matplotlib.pyplot as plt



if render:
    input()

print("success !!")