# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
import numpy as np
import pytest

from BLUEPRINT.neutronics.plasma_source import PlasmaSource

plasma_params = {
    "elongation": 1.557,
    "ion_density_origin": 1.09e20,
    "ion_density_peaking_factor": 1,
    "ion_density_pedestal": 1.09e20,
    "ion_density_separatrix": 3e19,
    "ion_temperature_origin": 45.9,
    "ion_temperature_peaking_factor": 8.06,
    "ion_temperature_pedestal": 6.09,
    "ion_temperature_separatrix": 0.1,
    "major_radius": 906.0,
    "minor_radius": 292.258,
    "pedestal_radius": 0.8 * 292.258,
    "plasma_mode": "H",
    "shafranov_shift": 44.789,
    "triangularity": 0.270,
    "ion_temperature_beta": 6,
}


@pytest.fixture(scope="session")
def plasma_source():
    return PlasmaSource(**plasma_params)


class TestPlasmaSource:
    """
    A class to test the neutronics parameterised plasma source
    """

    def test_source_strength_magnetic_origin(self, plasma_source):
        shift_origin = (
            plasma_params["major_radius"] + plasma_params["shafranov_shift"]
        ) / 100.0
        source_strength = plasma_source.get_source_strength_xz(shift_origin, 0)

        assert np.isclose(source_strength, 9.67895904e18)

    def test_source_strength_origin(self, plasma_source):
        source_strength = plasma_source.get_source_strength_xz(0, 0)

        assert np.isnan(source_strength)

    def test_source_strength_boundary_z0(self, plasma_source):
        boundary = (
            plasma_params["major_radius"] + plasma_params["minor_radius"]
        ) / 100.0
        source_strength = plasma_source.get_source_strength_xz(boundary, 0)

        assert np.isclose(source_strength, 2236.3037837)

    def test_source_strength_point(self, plasma_source):
        source_strength = plasma_source.get_source_strength_xz(10.25, 3.4)

        assert np.isclose(source_strength, 1.56826565e15)

    def test_average_source_energy_magnetic_origin(self, plasma_source):
        shift_origin = (
            plasma_params["major_radius"] + plasma_params["shafranov_shift"]
        ) / 100.0
        average_source_energy = plasma_source.get_average_source_energy_xz(
            shift_origin, 0
        )

        assert np.isclose(average_source_energy, 14.58962449)

    def test_source_average_energy_origin(self, plasma_source):
        average_source_energy = plasma_source.get_average_source_energy_xz(0, 0)

        assert np.isnan(average_source_energy)

    def test_average_source_energy_boundary_z0(self, plasma_source):
        boundary = (
            plasma_params["major_radius"] + plasma_params["minor_radius"]
        ) / 100.0
        average_source_energy = plasma_source.get_average_source_energy_xz(boundary, 0)

        assert np.isclose(average_source_energy, 14.10378723)

    def test_average_source_energy_point(self, plasma_source):
        average_source_energy = plasma_source.get_average_source_energy_xz(10.25, 3.4)

        assert np.isclose(average_source_energy, 14.18716772)

    def test_plasma_mode_validation(self):
        test_plasma_params = plasma_params.copy()
        test_plasma_params["plasma_mode"] = "I"
        with pytest.raises(ValueError) as exception_info:
            test_plasma_source = PlasmaSource(**test_plasma_params)
        assert (
            str(exception_info.value)
            == "PlasmaSource.plasma_mode must be one of ['H', 'A', 'L'], provided value was 'I'."
        )


class TestParameterisation:
    """
    A class to test the units that go into the parameterisation of the plasma
    """

    def test_ion_density_magnetic_origin(self, plasma_source):
        ion_density = plasma_source.ion_density(0.0)

        assert np.isclose(ion_density, 1.09e20)

    def test_ion_density_inside_pedestal(self, plasma_source):
        ion_density = plasma_source.ion_density(0.2)

        assert np.isclose(ion_density, 1.09e20)

    def test_ion_density_outside_pedestal(self, plasma_source):
        ion_density = plasma_source.ion_density(2.4)

        assert np.isclose(ion_density, 1.00628584e20)

    def test_ion_density_boundary(self, plasma_source):
        boundary = plasma_params["minor_radius"] / 100.0
        ion_density = plasma_source.ion_density(boundary)

        assert np.isclose(ion_density, plasma_params["ion_density_separatrix"])

    def test_ion_temperature_magnetic_origin(self, plasma_source):
        ion_temperature = plasma_source.ion_temperature(0.0)

        assert np.isclose(ion_temperature, plasma_params["ion_temperature_origin"])

    def test_ion_temperature_inside_pedestal(self, plasma_source):
        ion_temperature = plasma_source.ion_temperature(0.2)

        assert np.isclose(ion_temperature, 45.89987429)

    def test_ion_temperature_outside_pedestal(self, plasma_source):
        ion_temperature = plasma_source.ion_temperature(2.4)

        assert np.isclose(ion_temperature, 5.45525594)

    def test_ion_temperature_boundary(self, plasma_source):
        boundary = plasma_params["minor_radius"] / 100.0
        ion_temperature = plasma_source.ion_temperature(boundary)

        assert np.isclose(ion_temperature, plasma_params["ion_temperature_separatrix"])

    def test_dt_cross_section(self, plasma_source):
        dt_cross_section = plasma_source.dt_cross_section(4.25e7)

        assert np.isclose(dt_cross_section, 0.0)

    def test_radial_source_strength_magnetic_origin(self, plasma_source):
        radial_source_strength = plasma_source.radial_source_strength(0.0)

        assert np.isclose(radial_source_strength, 9.67895904e18)

    def test_radial_source_strength_inside_pedestal(self, plasma_source):
        radial_source_strength = plasma_source.radial_source_strength(0.2)

        assert np.isclose(radial_source_strength, 9.67895201e18)

    def test_radial_source_strength_outside_pedestal(self, plasma_source):
        radial_source_strength = plasma_source.radial_source_strength(2.4)

        assert np.isclose(radial_source_strength, 1.82402025e17)

    def test_radial_source_strength_boundary(self, plasma_source):
        boundary = plasma_params["minor_radius"] / 100.0
        radial_source_strength = plasma_source.radial_source_strength(boundary)

        assert np.isclose(radial_source_strength, 2236.30378370)
