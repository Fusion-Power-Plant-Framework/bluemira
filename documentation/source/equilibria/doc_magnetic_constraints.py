import numpy as np

from bluemira.equilibria.opt_constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiConstraint,
)

magnetic_constraints = MagneticConstraintSet(
    [
        FieldNullConstraint(x=6, z=-6),
        IsofluxConstraint(
            x=np.array([4, 6, 8, 6]), z=np.array([0, -6, 0, -6]), ref_x=4, ref_z=0
        ),
        PsiConstraint(x=6, z=-6, target_value=10),
    ]
)
