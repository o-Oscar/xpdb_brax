# `xpdb_brax`

This repository is a work I conducted to try and combine the strengh of both extended position based dynamics (xpdb) and a jit-compiled physics simulator (brax). 

## Using xpdb_brax locally

To install xpdb_brax from source, clone this repo, `cd` to it, and then:

```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```


## Physically accurate

Xpdb_brax is a rigid-body simulator that integrates dynamics using a Euler scheme.
It solves constrains iterativly using Gauss-Newton and sub-stepping. 

Its accurate integration of the physics allows it to simulate the djanibekov effect. Look at the way the box flips from time to time around its z-axis.

<img src="https://github.com/o-Oscar/xpdb_brax/blob/main/djanibekov.gif"/>

An other interesting effect it can simulate out of the box is the precession effect of a spinning wheel. 

<img src="https://github.com/o-Oscar/xpdb_brax/blob/main/precession.gif"/>

Xpdb_brax currently support ball joints and revolute joints but it can be extended to support any kind of constrain.

<img src="https://github.com/o-Oscar/xpdb_brax/blob/main/triple_pendulum.gif"/>

