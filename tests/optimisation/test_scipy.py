# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._scipy import ScipyOptimiser
from bluemira.optimisation._tools import process_scipy_result
from bluemira.optimisation.error import (
    OptimisationError,
)
from tests.optimisation.tools import rosenbrock


def no_op(x):
    # No op for use as a dummy function within tests
    pass


class TestScipyOptimiser:
    @pytest.mark.parametrize("alg", [1.1, [1, 2]])
    def test_valueerror_given_invalid_algorithm(self, alg):
        with pytest.raises(ValueError):  # noqa: PT011
            ScipyOptimiser(alg, 1, no_op)

    @pytest.mark.parametrize(
        ("string", "enum"),
        [
            ("SLSQP_SCIPY", Algorithm.SLSQP_SCIPY),
            ("COBYLA_SCIPY", Algorithm.COBYLA_SCIPY),
            ("COBYQA", Algorithm.COBYQA),
        ],
    )
    def test_algorithm_converted_from_str_to_enum_scipy(self, string, enum):
        opt = ScipyOptimiser(string, 5, no_op)
        assert opt.algorithm == enum

    def test_algorithm_parameters_set_if_they_exist_scipy(self):
        params = {"workers": 8}
        opt = ScipyOptimiser("SLSQP_SCIPY", 5, no_op, opt_parameters=params)
        assert opt.opt_parameters == params

    def test_minimising_objective_function_set_on_init_scipy(self):
        opt = ScipyOptimiser(
            algorithm="SLSQP_SCIPY",
            n_variables=2,
            f_objective=lambda _: 5,
            df_objective=lambda _: np.array([1]),
            opt_conditions={"max_eval": 200},
        )
        assert opt.f_objective(0) == 5

    def test_minimum_found_on_rosenbrock_no_bounds_scipy(self):
        opt = ScipyOptimiser(
            "SLSQP_SCIPY", 2, rosenbrock, opt_conditions={"ftol_rel": 1e-9}
        )
        result = opt.optimise()
        np.testing.assert_allclose(result.x, [1, 1], rtol=1e-5)

    def test_minimum_found_on_rosenbrock_within_bounds_scipy(self):
        opt = ScipyOptimiser(
            "SLSQP_SCIPY", 2, rosenbrock, opt_conditions={"ftol_rel": 1e-9}
        )
        opt.set_lower_bounds(np.array([1.1, 0.5]))
        opt.set_upper_bounds(np.array([2, 0.9]))
        result = opt.optimise()
        np.testing.assert_allclose(result.x, [1.1, 0.9])

    def test_minimum_found_on_hock_schittkowski_71(self):

        def hs71_objective(x):
            return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

        def hs71_ineq(x):
            return 25.0 - np.prod(x)

        def hs71_eq(x):
            return np.sum(np.array(x) ** 2) - 40.0

        opt = ScipyOptimiser(
            "COBYQA", 4, hs71_objective, opt_parameters={"feasibility_tol": 1e-7}
        )
        opt.set_lower_bounds(np.ones(4))
        opt.set_upper_bounds(np.ones(4) * 5)
        opt.add_ineq_constraint(hs71_ineq, tolerance=1e-6)
        opt.add_eq_constraint(hs71_eq, tolerance=1e-6)
        result = opt.optimise(x0=[1.0, 5.0, 5.0, 1.0])
        np.testing.assert_allclose(
            result.x, [1.0, 4.742999, 3.821149, 1.379408], rtol=1e-4
        )

    @pytest.mark.parametrize(
        "alg",
        [
            "L_BFGS_B",
            "NELDER_MEAD",
            "POWELL",
            "TNC",
        ],
    )
    def test_optimisationerror_adding_eq_constraint_on_unsupported_algorithm_scipy(
        self, alg
    ):
        opt = ScipyOptimiser(alg, 5, no_op)
        with pytest.raises(OptimisationError):
            opt.add_eq_constraint(no_op, np.zeros(2))

    def test_add_eq_constraint_slsqp_scipy(self):
        opt = ScipyOptimiser("SLSQP_SCIPY", 5, no_op)
        opt.add_eq_constraint(lambda _: -1, np.array(1), lambda _: 2)
        assert len(opt._eq_constraints) == 1
        result = opt._eq_constraints[0]["fun"](np.zeros(1))
        grad = opt._eq_constraints[0]["jac"](np.zeros(1))
        np.testing.assert_equal(result, -1)
        np.testing.assert_equal(grad, 2)

    @pytest.mark.parametrize("alg", ["COBYLA_SCIPY", "COBYQA"])
    def test_add_eq_constraint_cobyqa_scipy(self, alg):
        opt = ScipyOptimiser(alg, 5, no_op)
        opt.add_eq_constraint(lambda _: -1, np.array(1), lambda _: 2)
        assert len(opt._eq_constraints) == 1
        result = opt._eq_constraints[0].fun(np.zeros(1))
        grad = opt._eq_constraints[0].jac(np.zeros(1))
        np.testing.assert_equal(result, -1)
        np.testing.assert_equal(grad, 2)

    @pytest.mark.parametrize(
        "alg",
        [
            "L_BFGS_B",
            "NELDER_MEAD",
            "POWELL",
            "TNC",
        ],
    )
    def test_optimisationerror_adding_ineq_constraint_on_unsupported_algorithm_scipy(
        self, alg
    ):
        opt = ScipyOptimiser(alg, 5, no_op)
        with pytest.raises(OptimisationError):
            opt.add_ineq_constraint(no_op, np.zeros(2))

    def test_add_ineq_constraint_sets_ineq_constraint_scipy(self):
        opt = ScipyOptimiser("SLSQP_SCIPY", 5, no_op)
        opt.add_ineq_constraint(lambda _: -1, np.array(1), lambda _: 2)
        assert len(opt._ineq_constraints) == 1
        result = opt._ineq_constraints[0]["fun"](np.zeros(1))
        grad = opt._ineq_constraints[0]["jac"](np.zeros(1))
        np.testing.assert_equal(result, 1)  # ineq flipped for scipy slsqp only
        np.testing.assert_equal(grad, -2)

    @pytest.mark.parametrize("alg", ["COBYLA_SCIPY", "COBYQA"])
    def test_add_ineq_constraint_cobyqa_scipy(self, alg):
        opt = ScipyOptimiser(alg, 5, no_op)
        opt.add_ineq_constraint(lambda _: -1, np.array(1), lambda _: 2)
        assert len(opt._ineq_constraints) == 1
        result = opt._ineq_constraints[0].fun(np.zeros(1))
        grad = opt._ineq_constraints[0].jac(np.zeros(1))
        np.testing.assert_equal(result, -1)
        np.testing.assert_equal(grad, 2)

    def test_valueerror_setting_lower_bounds_with_wrong_dims_scipy(self):
        opt = ScipyOptimiser("SLSQP_SCIPY", 2, no_op)
        with pytest.raises(ValueError):  # noqa: PT011
            opt.set_lower_bounds(np.array([1, 2, 3]))

    def test_valueerror_setting_upper_bounds_with_wrong_dims_scipy(self):
        opt = ScipyOptimiser("SLSQP_SCIPY", 4, no_op)
        with pytest.raises(ValueError):  # noqa: PT011
            opt.set_upper_bounds(np.array([[1, 2], [3, 4]]))

    def test_set_lower_bounds_sets_bounds_scipy(self):
        bounds = np.array([0, 1, 2, 3])
        opt = ScipyOptimiser("SLSQP_SCIPY", 4, no_op)
        opt.set_lower_bounds(bounds)
        np.testing.assert_allclose(opt.lower_bounds, bounds)

    def test_set_upper_bounds_sets_bounds_scipy(self):
        bounds = np.array([0, 1, 2, 3])
        opt = ScipyOptimiser("SLSQP_SCIPY", 4, no_op)
        opt.set_upper_bounds(bounds)
        np.testing.assert_allclose(opt.upper_bounds, bounds)

    @pytest.mark.parametrize("alg", ["SLSQP", "COBYLA", "COBYQA"])
    def test_process_scipy_result_status_0(self, alg):
        res = OptimizeResult(x=np.array([1.0]), status=0, success=False, message="")
        np.testing.assert_allclose(process_scipy_result(res, alg), np.array([1.0]))

    @pytest.mark.parametrize("alg", ["SLSQP", "COBYLA", "COBYQA"])
    def test_raise_given_optimise_exception_scipy(self, alg, caplog):
        res = OptimizeResult(x=np.array([1.0]), status=-1, success=False, message="")
        with pytest.raises(OptimisationError):
            process_scipy_result(res, alg)
        assert len(caplog.records) == 1

    @pytest.mark.parametrize(
        ("alg", "status"),
        [
            ("SLSQP", 9),
            ("COBYLA", 2),
            ("COBYQA", 5),
        ],
    )
    def test_warning_given_inoptimal_solution_scipy(self, alg, status, caplog):
        res = OptimizeResult(x=np.array([1.0]), status=status, success=False, message="")
        np.testing.assert_allclose(process_scipy_result(res, alg), np.array([1.0]))
        assert len(caplog.records) == 1
        assert "returning inoptimal result" in caplog.messages[0]

    def test_warning_given_optimise_exception_unknown_alg_scipy(self):
        res = OptimizeResult(x=np.array([1.0]), status=0, success=False, message="")
        with pytest.raises(OptimisationError):
            process_scipy_result(res, "UNKNOWN")

    def test_warning_given_optimise_exception_no_status_scipy(self, caplog):
        res = OptimizeResult(x=np.array([1.0]), success=False, message="")
        with pytest.raises(OptimisationError):
            process_scipy_result(res, "SLSQP")
        assert len(caplog.records) == 1
        assert "Failed without status." in caplog.messages[0]
