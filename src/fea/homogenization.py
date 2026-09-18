"""Periodic-boundary computational homogenization in FEniCSx (ROADMAP 3.3).

Strain-driven homogenization on the structured voxel tetrahedral mesh of
``src.geometry`` (the discretization choice of Andreassen & Andreasen, 2014):

    C_ijkl = (1/|Y|) int_Y C_ijmn(y) (e_mn^kl + eps_mn(u^kl)) dy

where ``u^kl`` is the periodic displacement fluctuation solved under the unit
macroscopic strain case ``e^kl``.  Periodic fluctuations are enforced EXACTLY
with master-slave multi-point constraints (``u_slave = u_master``) condensed
into the stiffness matrix; a single interior node is pinned to remove the rigid
translation null space.  Six load cases (3 normal + 3 engineering shear) recover
the full 6x6 Voigt tensor, which is symmetrized on return.

All material assigned to void voxels follows the Ersatz approach
(``void_scale * E_s``), keeping the system positive-definite.

Material configuration is module-level so that ``validate_vs_analytical`` and
the batch runner in ``generate_dataset.py`` stay in lockstep:

    >>> set_base_material(E=1.0, nu=0.3)
    >>> set_void_scale(1e-6)                # Ersatz void stiffness
"""
from __future__ import annotations

import time

import numpy as np
import ufl
import basix
import basix.ufl
import scipy.sparse as sp
from mpi4py import MPI
from scipy.sparse.linalg import splu

from dolfinx import fem, mesh as dmesh

from . import benchmarks
from .property_extraction import VOIGT_PAIRS

# --------------------------------------------------------------------------- #
# Module-level material configuration (inert without the FEniCSx stack)
# --------------------------------------------------------------------------- #

_BASE_E = 1.0
_BASE_NU = 0.3
_VOID_SCALE = 1e-6


def set_base_material(E: float, nu: float) -> None:
    """Set the solid-phase material shared by the whole pipeline."""
    global _BASE_E, _BASE_NU
    _BASE_E = float(E)
    _BASE_NU = float(nu)


def set_void_scale(scale: float) -> None:
    """Set the Ersatz void stiffness ratio (relative to the solid Young's modulus)."""
    global _VOID_SCALE
    _VOID_SCALE = float(scale)


def base_material() -> tuple[float, float]:
    return _BASE_E, _BASE_NU


def void_scale() -> float:
    return _VOID_SCALE


# Six unit macroscopic strain load cases as symmetric tensors --
# engineering shear convention (gamma23 = gamma13 = gamma12 = 1 -> tensor 0.5).
_UNIT_TENSORS = np.array([
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # e11
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],  # e22
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # e33
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]],  # g23
    [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # g13
    [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],  # g12
])


def _meshio_cells(mesh: "object") -> tuple[np.ndarray, np.ndarray]:
    """(points, tetra cells) arrays from a meshio structure mesh."""
    cells = mesh.cells_dict["tetra"]
    return mesh.points, cells


# --------------------------------------------------------------------------- #
# Per-resolution problem
# --------------------------------------------------------------------------- #

class _Problem:
    """Mesh, spaces, MPC tables and (re)compiled forms for one resolution.

    The structured lattice mesh is identical for every voxel cell of a given
    resolution, so it (and its factorization bookkeeping) is built once and
    cached globally.  Only the DG0 material fields change per call.
    """

    def __init__(self, meshio_mesh, res: int):
        self.res = res
        self.n = res + 1

        points, cells = _meshio_cells(meshio_mesh)
        if points.shape[0] != self.n**3 or cells.shape != (6 * res**3, 4):
            raise ValueError(
                f"expected the full {self.n}^3 lattice mesh at res={res}, "
                f"got points={points.shape}, cells={cells.shape}"
            )

        element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
        self.msh = dmesh.create_mesh(MPI.COMM_WORLD, cells, element, points)
        self.gx = self.msh.geometry.x
        self.V = fem.functionspace(self.msh, ("Lagrange", 1, (3,)))
        self.V0 = fem.functionspace(self.msh, ("DG", 0))

        assert self.V.dofmap.index_map_bs == 3, "node-major blocked dofs expected"

        # lattice node index -> dof block index (dof block == geometry point id)
        xs = np.arange(self.n, dtype=float) / res
        lattice = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
        row_lut = {tuple(np.round(c, 12)): i for i, c in enumerate(self.gx)}
        self.block = np.empty(self.n**3, dtype=np.int64)
        for i, c in enumerate(lattice):
            self.block[i] = row_lut[tuple(np.round(c, 12))]
        assert len(np.unique(self.block)) == self.n**3, "lattice nodes must be distinct"

        # material per cell: centroid -> voxel id -> (mu, lam)
        # voxels grids are numpy C-order arrays, so flat id = x*res^2 + y*res + z
        # (grid axis 0 = physical x carries the slowest stride).
        gm = self.msh.geometry.dofmaps[0]
        centers = self.gx[gm].mean(axis=1)
        vx = np.minimum((centers * res).astype(np.int64), res - 1)
        self.cell_voxel = vx[:, 0] * res**2 + vx[:, 1] * res + vx[:, 2]
        self.mu = fem.Function(self.V0)
        self.lam = fem.Function(self.V0)

        # PBC master/slave via union-find over the lattice, driven purely by
        # coordinates (immune to any node reordering by dolfinx).
        parent = np.arange(self.n**3)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for ax in range(3):
            other = np.delete(np.arange(3), ax)
            on_lo = np.flatnonzero(np.isclose(lattice[:, ax], 0.0))
            on_hi = np.flatnonzero(np.isclose(lattice[:, ax], 1.0))
            keys = np.round(lattice[:, other], 12)
            lookup = {tuple(k): i for k, i in zip(keys[on_lo], on_lo)}
            for hi in on_hi:
                lo = lookup[tuple(keys[hi])]
                union(int(lo), int(hi))

        self._rep = np.array([find(i) for i in range(self.n**3)])
        slave = np.flatnonzero(self._rep != np.arange(self.n**3))
        self.slave_master = np.stack([self._rep[s] for s in slave])   # lattice ids
        self.slave_node = slave

        # pin one interior node to remove rigid translations
        pin_lattice = 1 + self.n + self.n**2
        self.pin_block = self.block[pin_lattice]

        kept_lattice = np.flatnonzero(self._rep == np.arange(self.n**3))
        kept_blocks = self.block[kept_lattice]
        kept_blocks = np.setdiff1d(kept_blocks, [self.pin_block])
        self.kept = np.concatenate([3 * kept_blocks + c for c in range(3)])

        # dof ids of slaves and their masters (used by the solver to expand u)
        n_dof = 3 * self.n**3
        if len(slave):
            self._slave_dofs = np.concatenate(
                [3 * self.block[slave] + c for c in range(3)], axis=0)
            self._master_dofs = np.concatenate(
                [3 * self.block[self.slave_master] + c for c in range(3)], axis=0)
        else:
            self._slave_dofs = np.zeros(0, dtype=np.int64)
            self._master_dofs = np.zeros(0, dtype=np.int64)

        # condensation operator L: u_full = L u_kept (slaves folded into masters)
        rows = [np.arange(n_dof)]
        cols = [np.arange(n_dof)]
        data = [np.ones(n_dof)]
        if len(slave):
            rows.append(self._master_dofs)
            cols.append(self._slave_dofs)
            data.append(np.ones(len(self._slave_dofs)))
        self._L = sp.csc_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_dof, n_dof),
        )

        # solve machinery
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)

        def eps(w: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
            return ufl.sym(ufl.grad(w))

        self._a = fem.form(
            (2.0 * self.mu * ufl.inner(eps(u), eps(v))
             + self.lam * ufl.tr(eps(u)) * ufl.tr(eps(v))) * ufl.dx
        )
        self._ebar = fem.Constant(self.msh, np.zeros((3, 3)))
        self._Lf = fem.form(
            -(2.0 * self.mu * ufl.inner(self._ebar, eps(v))
              + self.lam * ufl.tr(self._ebar) * ufl.tr(eps(v))) * ufl.dx
        )
        self._uh = fem.Function(self.V)
        stress = (2.0 * self.mu * (self._ebar + eps(self._uh))
                  + self.lam * ufl.tr(self._ebar + eps(self._uh)) * ufl.Identity(3))
        self._stress_forms = [fem.form(stress[a, b] * ufl.dx) for (a, b) in VOIGT_PAIRS]

    # ------------------------------------------------------------------ #

    def _solve_once(self, voxels: np.ndarray) -> tuple[np.ndarray, float]:
        """Assemble, factorize and solve all 6 load cases; return (C6, wall_seconds)."""
        t0 = time.time()
        solid = np.asarray(voxels, dtype=bool).reshape(-1)
        E_s, nu_s = base_material()
        E_v = _VOID_SCALE * E_s
        mu_s = E_s / (2.0 * (1.0 + nu_s))
        lam_s = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))
        mu_v = E_v / (2.0 * (1.0 + nu_s))
        lam_v = E_v * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))

        cell_solid = solid[self.cell_voxel]
        self.mu.x.array[:] = np.where(cell_solid, mu_s, mu_v)
        self.lam.x.array[:] = np.where(cell_solid, lam_s, lam_v)

        A = fem.assemble_matrix(self._a).to_scipy().tocsr()
        A_cond = ((self._L @ (A @ self._L.T))[self.kept][:, self.kept]).tocsc()
        lu = splu(A_cond)

        C6 = np.zeros((6, 6))
        for k, ebar_t in enumerate(_UNIT_TENSORS):
            self._ebar.value = ebar_t
            b = np.asarray(fem.assemble_vector(self._Lf).array, dtype=float)
            x = lu.solve((self._L @ b)[self.kept])
            u_full = np.zeros(3 * self.n**3)
            u_full[self.kept] = x
            if len(self._slave_dofs):
                u_full[self._slave_dofs] = u_full[self._master_dofs]
            self._uh.x.array[:] = u_full

            for j, form in enumerate(self._stress_forms):
                C6[k, j] = fem.assemble_scalar(form)
        return C6, time.time() - t0


_PROBLEMS: dict[int, _Problem] = {}


def homogenize_periodic_cell(voxels: np.ndarray, mesh, pairings: dict | None, res: int) -> np.ndarray:
    """Effective 6x6 stiffness (engineering Voigt) of one voxel unit cell.

    Parameters mirror the batch-runner contract in ``generate_dataset.py``:
    ``mesh`` is the meshio structure mesh of the cell, ``pairings`` the exact
    PBC tables (used only as a consistency check -- the MPCs are built from the
    lattice coordinates directly).
    """
    problem = _PROBLEMS.get(res)
    if problem is None:
        problem = _Problem(mesh, res)
        _PROBLEMS[res] = problem
    if pairings is not None:
        for (lo, hi) in pairings.values():
            if len(lo) != len(hi) or len(lo) != (res + 1) ** 2:
                raise ValueError(f"PBC pairing mismatch at res={res}: {len(lo)} pairs")
    C6, _ = problem._solve_once(voxels)
    return 0.5 * (C6 + C6.T)


# --------------------------------------------------------------------------- #
# Analytical validation (required Phase 1 preflight, ROADMAP 3.3)
# --------------------------------------------------------------------------- #

def _entrywise_rel_err(C6: np.ndarray, expected: dict[tuple[int, int], float]) -> float:
    errs = []
    for (i, j), ref in expected.items():
        denom = max(1e-9, abs(ref))
        errs.append(abs(C6[i, j] - ref) / denom)
    return max(errs)


def _rel_err_over(entries: dict[str, tuple[float, float]]) -> float:
    """Max relative error over {label: (computed, target)} closure pairs."""
    errs = []
    for (value, ref) in entries.values():
        denom = max(1e-9, abs(ref))
        errs.append(abs(value - ref) / denom)
    return max(errs)


def validate_vs_analytical(tolerance: float = 0.05, res: int = 16) -> dict:
    """Run the three closed-form benchmarks and gate the result at ``tolerance``.

    Evaluated against the module-level base material / void scale.  Returns a
    summary dict (also logged) and raises ``RuntimeError`` on any breach --
    exactly the preflight behaviour ``generate_dataset.py`` requires.
    """
    from src.geometry import voxels_to_tetra
    from .property_extraction import voigt_isotropic

    E_s, nu_s = base_material()
    E_v, nu_v = _VOID_SCALE * E_s, nu_s

    voigt_cube = voigt_isotropic(E_s, nu_s)
    cube = benchmarks.solid_cube_grid(res)
    lam_grid, f_ach = benchmarks.laminate_grid(res, f_s=0.4)
    lam_ref = benchmarks.laminate_stiffness(f_ach, E_s, nu_s, E_v, nu_v)
    rods, rho_ach = benchmarks.rods_grid(res, rho=0.5)

    bench = {
        "solid_cube": (
            cube,
            {(i, j): voigt_cube[i, j] for i in range(6) for j in range(6)},
            None,
        ),
        "laminate_x": (
            lam_grid,
            {(i, j): lam_ref[i, j] for i in range(6) for j in range(6)},
            None,
        ),
        "aligned_rods_x": (
            rods,
            None,
            lambda C6: benchmarks.rods_closure(
                C6, rho_ach, E_s, nu_s, E_v, nu_v),
        ),
    }

    summary = {"tolerance": tolerance, "res": res, "benchmarks": {}}
    ok_all = True
    for name, (grid, expected, closure) in bench.items():
        mesh = voxels_to_tetra(grid)
        C6 = homogenize_periodic_cell(grid, mesh, None, res)
        if closure is not None:
            err = _rel_err_over(closure(C6))
        else:
            err = _entrywise_rel_err(C6, expected)
        ok = err <= tolerance
        ok_all &= ok
        summary["benchmarks"][name] = {"max_rel_err": float(err), "ok": bool(ok)}
        print(f"  [{name}] max relative error vs closed form: {err:.3e} "
              f"({'OK' if ok else 'FAIL'})")
    summary["valid"] = bool(ok_all)
    if not ok_all:
        raise RuntimeError(
            f"homogenization validation FAILED (tolerance {100 * tolerance:.1f}%); "
            "refusing to generate a labelled dataset from an unvalidated solver"
        )
    return summary


# --------------------------------------------------------------------------- #
# Public solver for the E2E / toy paths (kept close to generate_dataset)
# --------------------------------------------------------------------------- #

class Homogenizer:
    """Object wrapper around ``homogenize_periodic_cell`` with timing bookkeeping."""

    def __init__(self, res: int):
        self.res = res
        self._problem = _PROBLEMS.get(res)
        self.times: list[float] = []

    def __call__(self, voxels: np.ndarray, mesh, pairings, res: int) -> np.ndarray:
        if res != self.res:
            raise ValueError(f"Homogenizer locked to res={self.res}, got {res}")
        if self._problem is None:
            _PROBLEMS[res] = _Problem(mesh, res)
            self._problem = _PROBLEMS[res]
        C6, t = self._problem._solve_once(voxels)
        self.times.append(t)
        return 0.5 * (C6 + C6.T)