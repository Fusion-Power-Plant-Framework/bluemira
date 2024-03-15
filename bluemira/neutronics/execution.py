# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Different ways of running openmc"""
from pathlib import Path

import openmc

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.tools import _timing


class RunMode:
    """Generic run method"""

    xml_files_created = ()
    run_mode_str = ""

    def __init__(self): # noqa: D105
        if isinstance(self, RunMode):
            raise TypeError(
                "RunMode is a baseclass that is not meant to be initialized!"
            )

    def set_up(self, cross_section_xml):
        """Set up openmc"""
        try:
            from openmc.config import config  # noqa: PLC0415

            config["cross_sections"] = cross_section_xml

        except ModuleNotFoundError:
            # Not new enought openmc
            import os  # noqa: PLC0415

            os.environ["OPENMC_CROSS_SECTIONS"] = str(cross_section_xml)

        self.settings = openmc.Settings()
        self.settings.run_mode = self.run_mode_str
        self.settings.output = {"summary":False}

    @classmethod
    def clean_up_directory(cls, base_path):
        """Remove files generated during the run (mainly .xml files.)"""
        for file_name in cls.xml_files_created:
            Path(base_path, file_name).unlink(missing_ok=False)

    def run(self, *args, output=False, **kwargs) -> None:
        """Complete the run"""
        self.settings.export_to_xml()
        _timing(openmc.run, "Executed in", "Running OpenMC", debug_info_str=False)(
            *args, output=output, **kwargs
        )
        bluemira_debug(f"Cleaning up files ({', '.join(self.xml_files_created)}).")
        self.clean_up_directory(Path.cwd())


class Plotting(RunMode):
    """Plotting only"""
    xml_files_created = ("settings.xml", "geometry.xml", "materials.xml")


class PlasmaSourceSimulation(RunMode):
    """Run with our standard plasma source"""
    xml_files_created = ("settings.xml", "geometry.xml", "materials.xml", "tallies.xml")

    def set_up(self, cross_section_xml):
        super().set_up(cross_section_xml)
        self.settings.particles = ...
        self.settings.batches = ...
        self.settings.photon_transport = ...
        self.settings.electron_treatment = ...
        self.settings.source = ...


class RingSourceSimulation(RunMode):
    xml_files_created = ("settings.xml", "geometry.xml", "materials.xml", "tallies.xml")

    def set_up(self, cross_section_xml):
        super().set_up(cross_section_xml)
        self.settings.particles = ...
        self.settings.batches = ...
        self.settings.photon_transport = ...
        self.settings.electron_treatment = ...
        self.settings.source = ...

class CalculateVolume(RunMode):
    """Run Monte Carlo to get the volume"""
    xml_files_created = ("settings.xml", "geometry.xml", "volume_settings.xml")

    def set_up(self, cross_section_xml):
        super().set_up(cross_section_xml)
