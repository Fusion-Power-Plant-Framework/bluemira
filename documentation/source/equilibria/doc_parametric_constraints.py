from bluemira.equilibria.opt_constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.shapes import flux_surface_johner


class ClassicalSNConstraints(MagneticConstraintSet):
    def __init__(
        self, R_0, A, kappa_upper, kappa_lower, delta_upper, delta_lower, n_points=40
    ):
        shape = flux_surface_johner(
            R_0,
            0,
            R_0 / A,
            kappa_upper,
            kappa_lower,
            delta_upper,
            delta_lower,
            psi_u_neg=180,
            psi_u_pos=5,
            psi_l_neg=-120,
            psi_l_pos=30,
            n=n_points,
        )

        isoflux = IsofluxConstraint(
            x=shape.x, z=shape.z, ref_x=shape.x[0], ref_z=shape.z[0]
        )
        xpoint = FieldNullConstraint(
            x=R_0 - delta_lower * R_0 / A, z=-kappa_lower * R_0 / A
        )
        super().__init__([isoflux, xpoint])


my_sn = ClassicalSNConstraints(9, 3.1, 1.6, 1.8, 0.4, 0.4)
my_sn.plot()
