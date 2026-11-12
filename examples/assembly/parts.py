"""The interim cylinder-and-hole parts of the assembly task (the final CAD will follow): dimensions, and the hole as a ring of box colliders (a hollow
body cannot be one convex collider, and a dynamic body cannot use a triangle mesh), checked against the analytic fit condition of assembly.fits. numpy only.

Part frame: origin at the centre of the mating face, +z out of it, the body behind it (z from -BODY_L to 0), the peg in front of it (z from 0 to PEG_L).
The parts stand upright on the table like the boxes (female opening up, male tip up) and are picked from the side.
"""
import math

import numpy as np

BODY_R, BODY_L = 0.0225, 0.070          # body: radius and length behind the mating face [m]
HOLE_R, HOLE_DEPTH = 0.010, 0.030       # socket: hole radius at the walls, depth [m]
PEG_L = 0.030                           # peg length beyond the shoulder = the hole depth
CLEARANCE = 0.002                       # radial gap peg to hole: ~80 % insertion odds at 1 mm / 1 deg alignment (assembly.py table)
PEG_R = HOLE_R - CLEARANCE
N_WALLS = 16                            # boxes around the hole: the hole is a 16-gon with the inradius HOLE_R (corners 0.19 mm further out)


def socket_walls(n=N_WALLS):
    """The walls of the hole as boxes in the part frame: [(centre xyz, size (radial, tangential, height), angle about z)]. Each box's inner face is tangent to
    the hole radius and as wide as the outer face of the ring needs, so neighbours overlap and the hole is a regular polygon."""
    t = BODY_R - HOLE_R
    w = 2 * BODY_R * math.tan(math.pi / n)
    rc = (HOLE_R + BODY_R) / 2
    return [((rc * math.cos(a), rc * math.sin(a), -HOLE_DEPTH / 2), (t, w, HOLE_DEPTH), a) for a in (2 * math.pi * k / n for k in range(n))]


def peg_surface(n_angle=36, n_z=16):
    """Points on the peg's side and tip in the male part frame (the shoulder face at z = 0, the tip at z = PEG_L)."""
    ang = np.linspace(0, 2 * math.pi, n_angle, endpoint=False)
    side = [(PEG_R * math.cos(a), PEG_R * math.sin(a), z) for z in np.linspace(0, PEG_L, n_z) for a in ang]
    tip = [(r * math.cos(a), r * math.sin(a), PEG_L) for r in np.linspace(0, PEG_R, 4) for a in ang]
    return np.array(side + tip)


def collides(T_fm, walls=None):
    """True when the peg of the male part at F_T_M touches a wall of the hole (only the part of the peg inside the mouth plane is tested)."""
    walls = walls or socket_walls()
    pts = peg_surface() @ T_fm[:3, :3].T + T_fm[:3, 3]
    pts = pts[pts[:, 2] < 0.0]
    for (cx, cy, cz), (t, w, h), a in walls:
        c, s = math.cos(a), math.sin(a)
        d = pts - (cx, cy, cz)
        x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        if np.any((np.abs(x) <= t / 2) & (np.abs(y) <= w / 2) & (np.abs(d[:, 2]) <= h / 2)):
            return True
    return False
