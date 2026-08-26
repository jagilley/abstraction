"""The piece, as constants -- and nothing else in this module, deliberately.

`legato/gates.py` holds the same two values, but importing them from there registers legato's Modal
ENTRYPOINTS on the shared `mjc.shared` app and collides with this node's ("Duplicate local entrypoint
name: gates"). And `offbook/world.py` imports numpy at module scope, which a Modal LOCAL entrypoint
cannot do -- the local side of a `modal run` executes on a laptop with neither numpy nor mujoco (the
`arm_env.py` contract). So the constants live in a module that imports nothing.

The geometry is legato's verbatim -- the closed 4-leg loop found by FK search, `W0 = fk(q_center)`
closing it, the calibrated `l2` design point -- and gate G-F asserts the fork still reproduces the
donor traversal bit-for-bit rather than trusting that the numbers were copied correctly.

    q0 --A approach, H=14, 0.300 m--> W1 (0.0359, 0.5248)
       --B drill 0,  H=20, 0.428 m--> W2 (0.3820, 0.2739)
       --C drill 1,  H=20, 0.453 m, THE PATCH--> W3 (0.6029, 0.6696)
       --D drill 2,  H=20, 0.420 m--> W0 (0.1968, 0.7785)
"""

DEF_WPS = "0.0359,0.5248;0.3820,0.2739;0.6029,0.6696;0.1968,0.7785"
DEF_PATCH_SEG = 1
