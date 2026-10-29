"""
Numba kernels for the first-order (factored) eikonal equation on tensor meshes.

Traveltimes live on the nodes of a 2D or 3D tensor mesh, treated as 3D with
a single node along z for 2D meshes. Node ``i`` has subscripts
``(ix, iy, iz)`` with ``i = ix + nx * (iy + ny * iz)``, the discretize order.

The traveltime is factored as ``T = D * tau``, where ``D`` is the distance to
the source, and the kernels solve for ``tau``. In a homogeneous medium
``tau`` equals the slowness everywhere, which removes the error caused by
the curvature of the wavefront near the source. With ``factored=False``,
``D = 1`` and ``tau = T``, which is the plain eikonal equation.

At a free node, the one-sided derivatives of ``T`` along axis ``d`` are

    p_minus = g_d tau_i + D (tau_i - tau_{i-1}) / h_minus = a_m tau_i - b_m
    p_plus  = g_d tau_i + D (tau_{i+1} - tau_i) / h_plus  = a_p tau_i - b_p

with ``g_d = dD/dx_d``. Godunov's upwind Hamiltonian uses, for each axis,

    max(max(p_minus, 0), -min(p_plus, 0))^2,

so each axis takes the "minus" branch, the "plus" branch or no contribution,
and the discrete equation is ``sum_d branch_d^2 = s_i^2`` (Luo & Qian, 2012).
The branches are stored per node as a base-3 code (0: none, 1: minus,
2: plus for each axis); a branch whose derivative is exactly zero contributes
nothing and is stored as none. With ``factored=False`` this is the classic
Godunov scheme. Fixed nodes (the corners of the cell holding the source) have
prescribed values.

Each node depends on the neighbours of its branches. In the factored scheme
two nodes on either side of the source's coordinate plane can depend on each
other, so the dependency graph has small cycles (a few nodes). ``solve``
groups the nodes into strongly connected components in dependency-first
order (Tarjan's algorithm). The linearized equations are then block
triangular: ``tangent`` and ``adjoint`` solve them component by component,
forwards and backwards, with a small dense solve for the rare components of
more than one node.

Every kernel loops over sources (or adjoint right-hand sides) in parallel.
"""

import numpy as np
from numba import njit, prange

_NONE, _MINUS, _PLUS = 0, 1, 2


@njit(cache=True, inline="always")
def _branch(code, d):
    return (code // 3**d) % 3


@njit(cache=True, inline="always")
def _distance(ix, iy, iz, nodes_x, nodes_y, nodes_z, src, factored, grad):
    """Distance to the source and its gradient (``D = 1``, zero gradient if not factored)."""
    if not factored:
        grad[:] = 0.0
        return 1.0
    cx = nodes_x[ix] - src[0]
    cy = nodes_y[iy] - src[1]
    cz = nodes_z[iz] - src[2]
    D = np.sqrt(cx * cx + cy * cy + cz * cz)
    if D == 0.0:
        grad[:] = 0.0
    else:
        grad[0] = cx / D
        grad[1] = cy / D
        grad[2] = cz / D
    return D


@njit(cache=True, inline="always")
def _axis(tau, i, idx, n, stride, nodes, D, g, coef, d):
    """One-sided derivative coefficients along one axis.

    ``coef[:, d]`` = a_m, b_m, a_p, b_p, h_m, h_p, has_m, has_p.
    """
    coef[6, d] = 0.0
    coef[7, d] = 0.0
    if idx > 0 and tau[i - stride] < np.inf:
        h = nodes[idx] - nodes[idx - 1]
        coef[0, d] = g + D / h
        coef[1, d] = D * tau[i - stride] / h
        coef[4, d] = h
        coef[6, d] = 1.0
    if idx < n - 1 and tau[i + stride] < np.inf:
        h = nodes[idx + 1] - nodes[idx]
        coef[2, d] = g - D / h
        coef[3, d] = -D * tau[i + stride] / h
        coef[5, d] = h
        coef[7, d] = 1.0


@njit(cache=True)
def _stencil(tau, i, nodes_x, nodes_y, nodes_z, src, factored, coef, grad):
    """Fill the coefficients of node ``i`` and return ``D_i``."""
    nx, ny = nodes_x.size, nodes_y.size
    return _stencil_at(
        tau,
        i,
        i % nx,
        (i // nx) % ny,
        i // (nx * ny),
        nodes_x,
        nodes_y,
        nodes_z,
        src,
        factored,
        coef,
        grad,
    )


@njit(cache=True, inline="always")
def _stencil_at(
    tau, i, ix, iy, iz, nodes_x, nodes_y, nodes_z, src, factored, coef, grad
):
    """``_stencil`` for a node whose subscripts are already known."""
    nx, ny, nz = nodes_x.size, nodes_y.size, nodes_z.size
    D = _distance(ix, iy, iz, nodes_x, nodes_y, nodes_z, src, factored, grad)
    _axis(tau, i, ix, nx, 1, nodes_x, D, grad[0], coef, 0)
    _axis(tau, i, iy, ny, nx, nodes_y, D, grad[1], coef, 1)
    _axis(tau, i, iz, nz, nx * ny, nodes_z, D, grad[2], coef, 2)
    return D


@njit(cache=True, inline="always")
def _consistent(coef, code, t):
    """Whether every axis takes the branch ``code`` asks for at ``tau_i = t``."""
    for d in range(3):
        p_minus = coef[0, d] * t - coef[1, d] if coef[6, d] else -np.inf
        p_plus = coef[2, d] * t - coef[3, d] if coef[7, d] else np.inf
        b = _branch(code, d)
        if b == _MINUS:
            if p_minus < 0.0 or p_minus < -p_plus:
                return False
        elif b == _PLUS:
            if p_plus > 0.0 or -p_plus < p_minus:
                return False
        elif p_minus > 0.0 or p_plus < 0.0:
            return False
    return True


@njit(cache=True, inline="always")
def _term(coef, d, t):
    """Godunov term of axis ``d`` at ``tau_i = t``, and the branch giving it."""
    value = 0.0
    b = _NONE
    if coef[6, d]:
        v = coef[0, d] * t - coef[1, d]
        if v > value:
            value = v
            b = _MINUS
    if coef[7, d]:
        v = coef[3, d] - coef[2, d] * t
        if v > value:
            value = v
            b = _PLUS
    return value, b


@njit(cache=True, inline="always")
def _monotone(coef):
    """Whether every one-sided term increases with ``tau_i`` (true for D > h)."""
    for d in range(3):
        if coef[6, d] and coef[0, d] <= 0.0:
            return False
        if coef[7, d] and coef[2, d] >= 0.0:
            return False
    return True


@njit(cache=True, inline="always")
def _first_branch(coef, d):
    """The branch of axis ``d`` that becomes positive first as ``t`` increases.

    Returns the ``t`` where it becomes positive, its slope ``alpha`` and offset
    ``beta`` (the term is ``alpha t - beta``), and the branch.
    """
    z = np.inf
    alpha = 1.0
    beta = 0.0
    b = _NONE
    if coef[6, d]:
        z = coef[1, d] / coef[0, d]
        alpha = coef[0, d]
        beta = coef[1, d]
        b = _MINUS
    if coef[7, d]:
        z_plus = coef[3, d] / coef[2, d]
        if z_plus < z:
            z = z_plus
            alpha = -coef[2, d]
            beta = -coef[3, d]
            b = _PLUS
    return z, alpha, beta, b


@njit(cache=True)
def _monotone_solve(coef, s, breakpoints):
    """Root of ``sum_d term_d(t)^2 = s^2`` when every term increases with ``t``.

    Fast path, as for the classic scheme: give each axis the branch that
    becomes positive first, add the axes in the order they become positive
    and stop when the next one would not be active. The answer is checked;
    if a branch switched before the root it falls back to ``_bracket_solve``.
    """
    z0, a0, b0, c0 = _first_branch(coef, 0)
    z1, a1, b1, c1 = _first_branch(coef, 1)
    z2, a2, b2, c2 = _first_branch(coef, 2)
    d0, d1, d2 = 0, 1, 2
    # sort the three axes by z
    if z1 < z0:
        z0, a0, b0, c0, d0, z1, a1, b1, c1, d1 = z1, a1, b1, c1, d1, z0, a0, b0, c0, d0
    if z2 < z1:
        z1, a1, b1, c1, d1, z2, a2, b2, c2, d2 = z2, a2, b2, c2, d2, z1, a1, b1, c1, d1
        if z1 < z0:
            z0, a0, b0, c0, d0, z1, a1, b1, c1, d1 = (
                z1,
                a1,
                b1,
                c1,
                d1,
                z0,
                a0,
                b0,
                c0,
                d0,
            )
    if z0 == np.inf:
        return np.inf, 0

    t = (b0 + s) / a0
    code = c0 * 3**d0
    if t > z1:
        A = a0 * a0 + a1 * a1
        B = a0 * b0 + a1 * b1
        C = b0 * b0 + b1 * b1 - s * s
        t = (B + np.sqrt(max(B * B - A * C, 0.0))) / A
        code += c1 * 3**d1
        if t > z2:
            A += a2 * a2
            B += a2 * b2
            C += b2 * b2
            t = (B + np.sqrt(max(B * B - A * C, 0.0))) / A
            code += c2 * 3**d2
    if _consistent(coef, code, t):
        return t, code
    return _bracket_solve(coef, s, breakpoints)


@njit(cache=True)
def _bracket_solve(coef, s, breakpoints):
    """Root of ``sum_d term_d(t)^2 = s^2`` when every term increases with ``t``.

    Each term is convex and piecewise linear, so the left side increases;
    the root is bracketed between sorted breakpoints, where the branches are
    fixed, and found with one quadratic.
    """
    nb = 0
    for d in range(3):
        if coef[6, d]:
            breakpoints[nb] = coef[1, d] / coef[0, d]
            nb += 1
        if coef[7, d]:
            breakpoints[nb] = coef[3, d] / coef[2, d]
            nb += 1
        # where the two branches cross (they are parallel if the slopes match)
        if coef[6, d] and coef[7, d] and coef[0, d] + coef[2, d] != 0.0:
            breakpoints[nb] = (coef[1, d] + coef[3, d]) / (coef[0, d] + coef[2, d])
            nb += 1
    if nb == 0:
        return np.inf, 0
    # insertion sort: at most 9 values
    for q in range(1, nb):
        value = breakpoints[q]
        r = q - 1
        while r >= 0 and breakpoints[r] > value:
            breakpoints[r + 1] = breakpoints[r]
            r -= 1
        breakpoints[r + 1] = value

    # the left side is zero below the first breakpoint
    k = nb
    for q in range(1, nb):
        total = 0.0
        for d in range(3):
            value, _ = _term(coef, d, breakpoints[q])
            total += value * value
        if total >= s * s:
            k = q
            break
    left = breakpoints[k - 1]
    t_mid = 0.5 * (left + breakpoints[k]) if k < nb else left + abs(left) + 1.0

    A = 0.0
    B = 0.0
    C = -s * s
    code = 0
    for d in range(3):
        _, b = _term(coef, d, t_mid)
        if b == _NONE:
            continue
        i = 0 if b == _MINUS else 2
        A += coef[i, d] * coef[i, d]
        B += coef[i, d] * coef[i + 1, d]
        C += coef[i + 1, d] * coef[i + 1, d]
        code += b * 3**d
    if A <= 0.0:
        return np.inf, 0
    return (B + np.sqrt(max(B * B - A * C, 0.0))) / A, code


@njit(cache=True)
def _local_solve(coef, s, breakpoints):
    """Solution for ``tau_i`` and its branch code (inf and 0 if there is none).

    Uses the bracketing solve when the terms are monotone, and otherwise the
    smallest consistent root over all branch combinations. ``breakpoints`` is
    a work array of length 9.
    """
    if _monotone(coef):
        return _monotone_solve(coef, s, breakpoints)
    best = np.inf
    best_code = 0
    for code in range(1, 27):
        A = 0.0
        B = 0.0
        C = -s * s
        usable = True
        for d in range(3):
            b = _branch(code, d)
            if b == _NONE:
                continue
            k = 0 if b == _MINUS else 2
            if not coef[6 + k // 2, d]:
                usable = False
                break
            A += coef[k, d] * coef[k, d]
            B += coef[k, d] * coef[k + 1, d]
            C += coef[k + 1, d] * coef[k + 1, d]
        if not usable or A <= 0.0:
            continue
        disc = B * B - A * C
        if disc < 0.0:
            continue
        root = np.sqrt(disc)
        for t in ((B - root) / A, (B + root) / A):
            if 0.0 < t < best and _consistent(coef, code, t):
                best = t
                best_code = code
    return best, best_code


@njit(cache=True, inline="always")
def _axis_lines(tau, i, idx, n, stride, nodes, D, g):
    """The one-sided lines of one axis as scalars (same values as ``_axis``).

    Returns has_m, a_m, b_m, has_p, a_p, b_p.
    """
    has_m = False
    a_m = 0.0
    b_m = 0.0
    has_p = False
    a_p = 0.0
    b_p = 0.0
    if idx > 0:
        t = tau[i - stride]
        if t < np.inf:
            r = D / (nodes[idx] - nodes[idx - 1])
            has_m = True
            a_m = g + r
            b_m = r * t
    if idx < n - 1:
        t = tau[i + stride]
        if t < np.inf:
            r = D / (nodes[idx + 1] - nodes[idx])
            has_p = True
            a_p = g - r
            b_p = -r * t
    return has_m, a_m, b_m, has_p, a_p, b_p


@njit(cache=True, inline="always")
def _first_line(has_m, a_m, b_m, has_p, a_p, b_p):
    """Scalar version of ``_first_branch``: returns z, alpha, beta, branch."""
    z = np.inf
    alpha = 1.0
    beta = 0.0
    b = _NONE
    if has_m:
        z = b_m / a_m
        alpha = a_m
        beta = b_m
        b = _MINUS
    if has_p:
        z_plus = b_p / a_p
        if z_plus < z:
            z = z_plus
            alpha = -a_p
            beta = -b_p
            b = _PLUS
    return z, alpha, beta, b


@njit(cache=True, inline="always")
def _takes_branch(b, t, has_m, a_m, b_m, has_p, a_p, b_p):
    """Scalar version of the per-axis check in ``_consistent``."""
    p_minus = a_m * t - b_m if has_m else -np.inf
    p_plus = a_p * t - b_p if has_p else np.inf
    if b == _MINUS:
        return p_minus >= 0.0 and p_minus >= -p_plus
    if b == _PLUS:
        return p_plus <= 0.0 and -p_plus >= p_minus
    return p_minus <= 0.0 and p_plus >= 0.0


@njit(cache=True, inline="always")
def _sweep_update(tau, i, ix, iy, iz, nodes_x, nodes_y, nodes_z, src, factored, s):
    """Local solve for the sweep, using scalars only.

    The same fast path as ``_monotone_solve``, without the work arrays the
    general solver needs; this is where the forward solve spends its time.
    Returns NaN when the general solver ``_local_solve`` is needed (near the
    source, or when a branch switches before the root).
    """
    nx, ny, nz = nodes_x.size, nodes_y.size, nodes_z.size
    D = 1.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    if factored:
        cx = nodes_x[ix] - src[0]
        cy = nodes_y[iy] - src[1]
        cz = nodes_z[iz] - src[2]
        D = np.sqrt(cx * cx + cy * cy + cz * cz)
        if D == 0.0:
            return np.nan
        gx = cx / D
        gy = cy / D
        gz = cz / D
    hmx, amx, bmx, hpx, apx, bpx = _axis_lines(tau, i, ix, nx, 1, nodes_x, D, gx)
    hmy, amy, bmy, hpy, apy, bpy = _axis_lines(tau, i, iy, ny, nx, nodes_y, D, gy)
    hmz, amz, bmz, hpz, apz, bpz = _axis_lines(tau, i, iz, nz, nx * ny, nodes_z, D, gz)
    # monotone: every one-sided term increases with tau_i
    if (
        (hmx and amx <= 0.0)
        or (hpx and apx >= 0.0)
        or (hmy and amy <= 0.0)
        or (hpy and apy >= 0.0)
        or (hmz and amz <= 0.0)
        or (hpz and apz >= 0.0)
    ):
        return np.nan

    z0, a0, b0, c0 = _first_line(hmx, amx, bmx, hpx, apx, bpx)
    z1, a1, b1, c1 = _first_line(hmy, amy, bmy, hpy, apy, bpy)
    z2, a2, b2, c2 = _first_line(hmz, amz, bmz, hpz, apz, bpz)
    d0, d1, d2 = 0, 1, 2
    if z1 < z0:
        z0, a0, b0, c0, d0, z1, a1, b1, c1, d1 = z1, a1, b1, c1, d1, z0, a0, b0, c0, d0
    if z2 < z1:
        z1, a1, b1, c1, d1, z2, a2, b2, c2, d2 = z2, a2, b2, c2, d2, z1, a1, b1, c1, d1
        if z1 < z0:
            z0, a0, b0, c0, d0, z1, a1, b1, c1, d1 = (
                z1,
                a1,
                b1,
                c1,
                d1,
                z0,
                a0,
                b0,
                c0,
                d0,
            )
    if z0 == np.inf:
        return np.inf

    t = (b0 + s) / a0
    n_active = 1
    if t > z1:
        A = a0 * a0 + a1 * a1
        B = a0 * b0 + a1 * b1
        C = b0 * b0 + b1 * b1 - s * s
        t = (B + np.sqrt(max(B * B - A * C, 0.0))) / A
        n_active = 2
        if t > z2:
            A += a2 * a2
            B += a2 * b2
            C += b2 * b2
            t = (B + np.sqrt(max(B * B - A * C, 0.0))) / A
            n_active = 3

    # the branch of each axis, then check that each axis really takes it at t
    branches = (_NONE, _NONE, _NONE)
    branches = _with_branch(branches, d0, c0)
    if n_active > 1:
        branches = _with_branch(branches, d1, c1)
    if n_active > 2:
        branches = _with_branch(branches, d2, c2)
    if (
        _takes_branch(branches[0], t, hmx, amx, bmx, hpx, apx, bpx)
        and _takes_branch(branches[1], t, hmy, amy, bmy, hpy, apy, bpy)
        and _takes_branch(branches[2], t, hmz, amz, bmz, hpz, apz, bpz)
    ):
        return t
    return np.nan


@njit(cache=True, inline="always")
def _with_branch(branches, d, b):
    """``branches`` with entry ``d`` set to ``b``."""
    if d == 0:
        return (b, branches[1], branches[2])
    if d == 1:
        return (branches[0], b, branches[2])
    return (branches[0], branches[1], b)


@njit(cache=True)
def _drop_zero_branches(coef, code, t):
    """Set the branches whose derivative is exactly zero to none."""
    out = 0
    for d in range(3):
        b = _branch(code, d)
        if b == _MINUS and coef[0, d] * t - coef[1, d] == 0.0:
            b = _NONE
        elif b == _PLUS and coef[2, d] * t - coef[3, d] == 0.0:
            b = _NONE
        out += b * 3**d
    return out


@njit(cache=True)
def _sweep(
    tau, fixed, s, nodes_x, nodes_y, nodes_z, dim, src, factored, tol, max_iterations
):
    """Fast sweeping in all 2**dim orderings until tau stops changing."""
    nx, ny, nz = nodes_x.size, nodes_y.size, nodes_z.size
    coef = np.zeros((8, 3))
    breakpoints = np.empty(9)
    grad = np.empty(3)
    for iteration in range(max_iterations):
        max_change = 0.0
        newly_reached = False
        for ordering in range(2**dim):
            for kz in range(nz):
                iz = kz if (ordering & 4) == 0 else nz - 1 - kz
                for ky in range(ny):
                    iy = ky if (ordering & 2) == 0 else ny - 1 - ky
                    for kx in range(nx):
                        ix = kx if (ordering & 1) == 0 else nx - 1 - kx
                        i = ix + nx * (iy + ny * iz)
                        if fixed[i]:
                            continue
                        t = _sweep_update(
                            tau,
                            i,
                            ix,
                            iy,
                            iz,
                            nodes_x,
                            nodes_y,
                            nodes_z,
                            src,
                            factored,
                            s[i],
                        )
                        if np.isnan(t):
                            _stencil_at(
                                tau,
                                i,
                                ix,
                                iy,
                                iz,
                                nodes_x,
                                nodes_y,
                                nodes_z,
                                src,
                                factored,
                                coef,
                                grad,
                            )
                            t, _ = _local_solve(coef, s[i], breakpoints)
                        if t < tau[i]:
                            if tau[i] == np.inf:
                                newly_reached = True
                            else:
                                max_change = max(max_change, tau[i] - t)
                            tau[i] = t
        if not newly_reached and max_change <= tol * np.max(tau):
            return iteration + 1
    return max_iterations


@njit(cache=True)
def _dependency(branch, fixed, v, e, strides):
    """The neighbour node ``v`` depends on through axis ``e``, or -1."""
    if fixed[v]:
        return -1
    b = _branch(branch[v], e)
    if b == _MINUS:
        return v - strides[e]
    if b == _PLUS:
        return v + strides[e]
    return -1


@njit(cache=True)
def _strong_components(branch, fixed, nx, ny, order, component_start):
    """Tarjan's algorithm on the dependency graph.

    Fills ``order`` with the nodes grouped by strongly connected component,
    each component after the components it depends on, and marks the first
    position of each component in ``component_start``. Returns the number of
    nodes in components of more than one node.
    """
    n = branch.size
    strides = (1, nx, nx * ny)
    index = np.full(n, -1, dtype=np.int64)
    low = np.empty(n, dtype=np.int64)
    on_stack = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    call_node = np.empty(n, dtype=np.int64)
    call_edge = np.empty(n, dtype=np.int64)
    counter = 0
    top = 0
    out = 0
    n_cyclic = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = counter
        low[root] = counter
        counter += 1
        stack[top] = root
        top += 1
        on_stack[root] = True
        depth = 1
        call_node[0] = root
        call_edge[0] = 0
        while depth > 0:
            v = call_node[depth - 1]
            e = call_edge[depth - 1]
            if e < 3:
                call_edge[depth - 1] += 1
                w = _dependency(branch, fixed, v, e, strides)
                if w < 0:
                    continue
                if index[w] == -1:
                    index[w] = counter
                    low[w] = counter
                    counter += 1
                    stack[top] = w
                    top += 1
                    on_stack[w] = True
                    call_node[depth] = w
                    call_edge[depth] = 0
                    depth += 1
                elif on_stack[w]:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                first = out
                while True:
                    top -= 1
                    w = stack[top]
                    on_stack[w] = False
                    order[out] = w
                    component_start[out] = 1 if out == first else 0
                    out += 1
                    if w == v:
                        break
                if out - first > 1:
                    n_cyclic += out - first
            depth -= 1
            if depth > 0:
                u = call_node[depth - 1]
                low[u] = min(low[u], low[v])
    return n_cyclic


@njit(parallel=True, cache=True)
def solve(
    s,
    nodes_x,
    nodes_y,
    nodes_z,
    dim,
    src_xyz,
    factored,
    fixed_ptr,
    fixed_idx,
    fixed_val,
    tol,
    max_iterations,
):
    """Solve for every source.

    Returns
    -------
    tau, T : (n_sources, n_nodes) numpy.ndarray
        Factor and traveltime, ``T = D * tau``.
    branch : (n_sources, n_nodes) numpy.ndarray of int8
        Branch code of each node.
    order : (n_sources, n_nodes) numpy.ndarray of int
        Nodes grouped by dependency component, dependencies first.
    component_start : (n_sources, n_nodes) numpy.ndarray of int8
        1 where a component starts in ``order``.
    n_iterations, n_cyclic : (n_sources,) numpy.ndarray of int
        Sweeping iterations, and nodes in components of more than one node.
    """
    n_sources = fixed_ptr.size - 1
    n = s.size
    nx, ny = nodes_x.size, nodes_y.size
    tau = np.empty((n_sources, n))
    T = np.empty((n_sources, n))
    branch = np.zeros((n_sources, n), dtype=np.int8)
    order = np.zeros((n_sources, n), dtype=np.int64)
    component_start = np.zeros((n_sources, n), dtype=np.int8)
    n_iterations = np.empty(n_sources, dtype=np.int64)
    n_cyclic = np.empty(n_sources, dtype=np.int64)
    for j in prange(n_sources):
        tau_j = tau[j]
        tau_j[:] = np.inf
        fixed = np.zeros(n, dtype=np.bool_)
        for k in range(fixed_ptr[j], fixed_ptr[j + 1]):
            tau_j[fixed_idx[k]] = fixed_val[k]
            fixed[fixed_idx[k]] = True
        n_iterations[j] = _sweep(
            tau_j,
            fixed,
            s,
            nodes_x,
            nodes_y,
            nodes_z,
            dim,
            src_xyz[j],
            factored,
            tol,
            max_iterations,
        )
        coef = np.zeros((8, 3))
        breakpoints = np.empty(9)
        grad = np.empty(3)
        for i in range(n):
            D = _stencil(
                tau_j, i, nodes_x, nodes_y, nodes_z, src_xyz[j], factored, coef, grad
            )
            T[j, i] = D * tau_j[i]
            if not fixed[i]:
                _, code = _local_solve(coef, s[i], breakpoints)
                branch[j, i] = _drop_zero_branches(coef, code, tau_j[i])
        n_cyclic[j] = _strong_components(
            branch[j], fixed, nx, ny, order[j], component_start[j]
        )
    return tau, T, branch, order, component_start, n_iterations, n_cyclic


@njit(cache=True)
def _linearization(
    tau, i, code, nodes_x, nodes_y, nodes_z, src, factored, coef, grad, c, neighbour
):
    """Weights of the linearized equation at a free node.

    ``C dtau_i - sum_d c_d dtau_{n_d} = s_i ds_i``. Fills ``c`` (zero for
    axes without a branch) and ``neighbour``, and returns ``C``.
    """
    nx, ny = nodes_x.size, nodes_y.size
    strides = (1, nx, nx * ny)
    D = _stencil(tau, i, nodes_x, nodes_y, nodes_z, src, factored, coef, grad)
    C = 0.0
    for d in range(3):
        c[d] = 0.0
        neighbour[d] = -1
        b = _branch(code, d)
        if b == _MINUS:
            r = coef[0, d] * tau[i] - coef[1, d]
            c[d] = r * D / coef[4, d]
            C += coef[0, d] * r
            neighbour[d] = i - strides[d]
        elif b == _PLUS:
            r = coef[2, d] * tau[i] - coef[3, d]
            c[d] = -r * D / coef[5, d]
            C += coef[2, d] * r
            neighbour[d] = i + strides[d]
    return C


@njit(cache=True)
def _component_end(component_start, k):
    end = k + 1
    while end < component_start.size and component_start[end] == 0:
        end += 1
    return end


@njit(cache=True)
def _position(order, start, end, node):
    for k in range(start, end):
        if order[k] == node:
            return k - start
    return -1


@njit(cache=True)
def _component_system(
    tau,
    order,
    start,
    end,
    branch,
    nodes_x,
    nodes_y,
    nodes_z,
    src,
    factored,
    coef,
    grad,
    c,
    neighbour,
    M,
    outside_c,
    outside_node,
):
    """Matrix of a multi-node component and its couplings to outside nodes."""
    m = end - start
    M[:m, :m] = 0.0
    for a in range(m):
        i = order[start + a]
        M[a, a] = _linearization(
            tau,
            i,
            branch[i],
            nodes_x,
            nodes_y,
            nodes_z,
            src,
            factored,
            coef,
            grad,
            c,
            neighbour,
        )
        for d in range(3):
            outside_c[a, d] = 0.0
            outside_node[a, d] = -1
            if c[d] == 0.0:
                continue
            b = _position(order, start, end, neighbour[d])
            if b >= 0:
                M[a, b] -= c[d]
            else:
                outside_c[a, d] = c[d]
                outside_node[a, d] = neighbour[d]


@njit(parallel=True, cache=True)
def tangent(
    tau,
    branch,
    order,
    component_start,
    s,
    ds,
    nodes_x,
    nodes_y,
    nodes_z,
    src_xyz,
    factored,
    fixed_ptr,
    fixed_idx,
    fixed_rhs,
):
    """Change in tau for a change in slowness, forwards along ``order``.

    At fixed nodes ``dtau`` is ``fixed_rhs``.
    """
    n_sources, n = tau.shape
    dtau = np.empty_like(tau)
    for j in prange(n_sources):
        tau_j = tau[j]
        dtau_j = dtau[j]
        fixed = np.zeros(n, dtype=np.bool_)
        for k in range(fixed_ptr[j], fixed_ptr[j + 1]):
            fixed[fixed_idx[k]] = True
            dtau_j[fixed_idx[k]] = fixed_rhs[k]
        coef = np.zeros((8, 3))
        grad = np.empty(3)
        c = np.empty(3)
        neighbour = np.empty(3, dtype=np.int64)
        k = 0
        while k < n:
            end = _component_end(component_start[j], k)
            m = end - k
            if m == 1:
                i = order[j, k]
                if not fixed[i]:
                    C = _linearization(
                        tau_j,
                        i,
                        branch[j, i],
                        nodes_x,
                        nodes_y,
                        nodes_z,
                        src_xyz[j],
                        factored,
                        coef,
                        grad,
                        c,
                        neighbour,
                    )
                    numerator = s[i] * ds[i]
                    for d in range(3):
                        if c[d] != 0.0:
                            numerator += c[d] * dtau_j[neighbour[d]]
                    dtau_j[i] = numerator / C
            else:
                M = np.empty((m, m))
                outside_c = np.empty((m, 3))
                outside_node = np.empty((m, 3), dtype=np.int64)
                _component_system(
                    tau_j,
                    order[j],
                    k,
                    end,
                    branch[j],
                    nodes_x,
                    nodes_y,
                    nodes_z,
                    src_xyz[j],
                    factored,
                    coef,
                    grad,
                    c,
                    neighbour,
                    M,
                    outside_c,
                    outside_node,
                )
                rhs = np.empty(m)
                for a in range(m):
                    i = order[j, k + a]
                    rhs[a] = s[i] * ds[i]
                    for d in range(3):
                        if outside_node[a, d] >= 0:
                            rhs[a] += outside_c[a, d] * dtau_j[outside_node[a, d]]
                x = np.linalg.solve(M, rhs)
                for a in range(m):
                    dtau_j[order[j, k + a]] = x[a]
            k = end
    return dtau


@njit(parallel=True, cache=True)
def adjoint(
    tau,
    branch,
    order,
    component_start,
    s,
    g,
    row_source,
    nodes_x,
    nodes_y,
    nodes_z,
    src_xyz,
    factored,
    fixed_ptr,
    fixed_idx,
):
    """Transpose of ``tangent`` applied to each row of ``g``, backwards along ``order``.

    Row ``r`` of ``g`` is a vector on the nodes of source ``row_source[r]``.

    Returns
    -------
    grad_nodes : (n_rows, n_nodes) numpy.ndarray
        Gradient with respect to the node slowness of the free nodes.
    fixed_adjoint : (n_rows, max_fixed) numpy.ndarray
        Adjoint variable at the fixed nodes of the row's source, in the order
        of ``fixed_idx``. Multiplying by the ``fixed_rhs`` coefficients gives
        the gradient with respect to the fixed values.
    """
    n_rows = g.shape[0]
    n = tau.shape[1]
    max_fixed = 0
    for j in range(fixed_ptr.size - 1):
        max_fixed = max(max_fixed, fixed_ptr[j + 1] - fixed_ptr[j])
    grad_nodes = np.zeros((n_rows, n))
    fixed_adjoint = np.zeros((n_rows, max_fixed))
    for r in prange(n_rows):
        j = row_source[r]
        tau_j = tau[j]
        acc = g[r].copy()
        fixed = np.zeros(n, dtype=np.bool_)
        for k in range(fixed_ptr[j], fixed_ptr[j + 1]):
            fixed[fixed_idx[k]] = True
        coef = np.zeros((8, 3))
        grad = np.empty(3)
        c = np.empty(3)
        neighbour = np.empty(3, dtype=np.int64)
        end = n
        while end > 0:
            start = end - 1
            while component_start[j, start] == 0:
                start -= 1
            m = end - start
            if m == 1:
                i = order[j, start]
                lam = acc[i]
                if not fixed[i] and lam != 0.0:
                    C = _linearization(
                        tau_j,
                        i,
                        branch[j, i],
                        nodes_x,
                        nodes_y,
                        nodes_z,
                        src_xyz[j],
                        factored,
                        coef,
                        grad,
                        c,
                        neighbour,
                    )
                    mu = lam / C
                    grad_nodes[r, i] = s[i] * mu
                    for d in range(3):
                        if c[d] != 0.0:
                            acc[neighbour[d]] += c[d] * mu
            else:
                rhs = np.empty(m)
                nonzero = False
                for a in range(m):
                    rhs[a] = acc[order[j, start + a]]
                    nonzero = nonzero or rhs[a] != 0.0
                if nonzero:
                    M = np.empty((m, m))
                    outside_c = np.empty((m, 3))
                    outside_node = np.empty((m, 3), dtype=np.int64)
                    _component_system(
                        tau_j,
                        order[j],
                        start,
                        end,
                        branch[j],
                        nodes_x,
                        nodes_y,
                        nodes_z,
                        src_xyz[j],
                        factored,
                        coef,
                        grad,
                        c,
                        neighbour,
                        M,
                        outside_c,
                        outside_node,
                    )
                    mu = np.linalg.solve(M.T.copy(), rhs)
                    for a in range(m):
                        i = order[j, start + a]
                        grad_nodes[r, i] = s[i] * mu[a]
                        for d in range(3):
                            if outside_node[a, d] >= 0:
                                acc[outside_node[a, d]] += outside_c[a, d] * mu[a]
            end = start
        for k in range(fixed_ptr[j], fixed_ptr[j + 1]):
            fixed_adjoint[r, k - fixed_ptr[j]] = acc[fixed_idx[k]]
    return grad_nodes, fixed_adjoint
