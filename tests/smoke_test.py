# Week-1 smoke test: hello-world FEniCSx homogenization on a single solid cube.
# ROADMAP.md Section 1 "Definition of done": clone -> create env -> this script must
# generate one voxel cube, homogenize it, and print its stiffness tensor.
#
# This is a TRIVIAL sanity check of the FEA stack only -- the full periodic-BC
# homogenization (PBC/MPC + 6 load cases + benchmark validation) is the Phase-1
# milestone in src/fea/ (Section 3.3). A fully solid, isotropic cube under a prescribed
# affine axial strain must return C1111 within tolerance of the closed form:
#   C1111 = λ + 2μ = E (1 − ν) / ((1 + ν)(1 − 2ν))
#
# Run:  `micromamba run -n metamaterials python tests/smoke_test.py`

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

E = 70e9      # Young's modulus [Pa]
NU = 0.33     # Poisson's ratio
TOL = 1e-4    # relative tolerance vs closed form (affine field is exact; this is generous)


def main() -> None:
    # 1) Generate one voxel cube (the Phase 1 representation): an all-solid
    #    binary occupancy grid (16^3 here for speed; the dataset uses 32^3).
    res = 16
    voxels = np.ones((res, res, res), dtype=np.uint8)
    print(f"voxel grid: shape={voxels.shape} dtype={voxels.dtype} "
          f"solid fraction={voxels.mean():.2f}")

    # 2) Build a trivial solid unit cube directly (no src/ imports on purpose):
    #    this checks the raw FEniCSx stack independently of the package. The
    #    real periodic voxel->mesh homogenization is in src/fea (Phase 1, 3.3).
    nx = 4
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, nx, nx, nx,
                                mesh.CellType.hexahedron)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))
    c1111_exact = lam + 2.0 * mu

    def sigma(w):
        eps = ufl.sym(ufl.grad(w))
        return 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(msh.geometry.dim)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = fem.form(ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx)
    zero = fem.Constant(msh, np.zeros(3, dtype=default_scalar_type))
    L = fem.form(ufl.dot(zero, v) * ufl.dx)

    # 3) Prescribed affine axial strain as Dirichlet data on the whole boundary.
    eps_xx = 1e-3

    def affine_u(x: np.ndarray) -> np.ndarray:
        vals = np.zeros((3, x.shape[1]), dtype=default_scalar_type)
        vals[0, :] = eps_xx * x[0, :]
        return vals

    u_bc = fem.Function(V)
    u_bc.interpolate(affine_u)
    boundary = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary)
    bc = fem.dirichletbc(u_bc, dofs)

    uh = fem.Function(V)
    LinearProblem(a, L, u=uh, bcs=[bc], petsc_options={"pc_type": "lu"},
                      petsc_options_prefix="smoke_").solve()

    # 4) Homogenize via the strain-energy formula: C1111 = 2*W / ε_xx^2 (Phase 1, Section 3.3).
    W = fem.assemble_scalar(
        fem.form(0.5 * ufl.inner(sigma(uh), ufl.sym(ufl.grad(uh))) * ufl.dx))
    c1111 = 2.0 * W / eps_xx**2
    rel_err = abs(c1111 - c1111_exact) / c1111_exact

    print(f"\nimposed E={E:.3e} Pa  nu={NU}")
    print(f"C1111 exact   : {c1111_exact:.6e} Pa")
    print(f"C1111 computed: {c1111:.6e} Pa")
    print(f"relative error: {rel_err:.2e}")
    assert rel_err <= TOL, f"C1111 off by {rel_err:.2e} > {TOL}"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()