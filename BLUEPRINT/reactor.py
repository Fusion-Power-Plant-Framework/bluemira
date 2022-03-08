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

"""
The Reactor God-object
"""
import datetime
import json
import os
import sys
from copy import deepcopy
from pathlib import Path, PosixPath
from time import time
from types import ModuleType
from typing import Type, Union

import numpy as np

# Configuration / Input imports
from bluemira.base.config import SingleNull

# Framework imports
from bluemira.base.file import BM_ROOT, FileManager, get_files_by_ext
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn, print_banner
from bluemira.base.parameter import ParameterFrame

# PROCESS imports
from bluemira.codes import systems_code_solver

# Equilibria imports
from bluemira.equilibria._deprecated_run import AbInitioEquilibriumProblem
from bluemira.equilibria.constants import (
    NB3SN_B_MAX,
    NB3SN_J_MAX,
    NBTI_B_MAX,
    NBTI_J_MAX,
)
from bluemira.equilibria.physics import normalise_beta
from bluemira.fuel_cycle.analysis import FuelCycleAnalysis
from bluemira.fuel_cycle.cycle import EUDEMOFuelCycleModel
from bluemira.fuel_cycle.lifecycle import LifeCycle

# Lifetime / fuel cycle imports
from bluemira.fuel_cycle.timeline_tools import (
    GompertzLearningStrategy,
    LogNormalAvailabilityStrategy,
)
from bluemira.geometry.error import GeometryError
from bluemira.utilities.tools import CommentJSONDecoder, json_writer
from BLUEPRINT.base import BLUE

# BLUPRINT.cad imports
from BLUEPRINT.cad import ReactorCAD
from BLUEPRINT.cad.cadtools import check_STL_folder
from BLUEPRINT.geometry.geomtools import qrotate

# Utility imports
from BLUEPRINT.geometry.loop import Loop, point_loop_cast
from BLUEPRINT.geometry.parameterisations import flatD, negativeD

# Neutronics imports
from BLUEPRINT.neutronics.simpleneutrons import BlanketCoverage
from BLUEPRINT.nova.firstwall import FirstWallProfile

# BLUEPRINT.nova imports
from BLUEPRINT.nova.stream import StreamFlow
from BLUEPRINT.nova.structure import CoilArchitect

# BLUEPRINT system imports
from BLUEPRINT.systems import (
    BalanceOfPlant,
    BreedingBlanket,
    Cryostat,
    Divertor,
    HCDSystem,
    Plasma,
    PoloidalFieldCoils,
    RadiationShield,
    ReactorCrossSection,
    ReactorSystem,
    STBreedingBlanket,
    ThermalShield,
    ToroidalFieldCoils,
    VacuumVessel,
)
from BLUEPRINT.systems.maintenance import RMMetrics
from BLUEPRINT.systems.optimisation_callbacks import (
    EQ_optimiser,
    FW_optimiser,
    TF_optimiser,
)
from BLUEPRINT.systems.physicstoolbox import (
    estimate_kappa95,
    lambda_q,
    n_DD_reactions,
    n_DT_reactions,
)
from BLUEPRINT.systems.plotting import ReactorPlotter
from BLUEPRINT.utilities.colortools import force_rgb


class Reactor(ReactorSystem):
    """
    Nuclear fusion reactor object (implicitly: a tokamak)

    This is the main reactor object used create reactor design points

    Parameters
    ----------
    config: dict
        The configuration dictionary of parameter variations from the config
        class
    build_config: dict
        The dictionary of build configuration (i.e. run vs. read, design objec-
        tives, etc.
    build_tweaks: dict
        The dictionary of numerical tweaks for the various calculations

    """

    # Input declarations
    config: dict
    build_config: dict
    build_tweaks: dict

    # ReactorSystem declarations
    STBB: Type[STBreedingBlanket]
    BB: Type[BreedingBlanket]
    BC: Type[BlanketCoverage]
    CR: Type[Cryostat]
    FW: Type[FirstWallProfile]
    PL: Type[Plasma]
    RS: Type[RadiationShield]
    TS: Type[ThermalShield]
    TF: Type[ToroidalFieldCoils]
    PF: Type[PoloidalFieldCoils]
    VV: Type[VacuumVessel]
    BOP: Type[BalanceOfPlant]
    DIV: Type[Divertor]
    HCD: Type[HCDSystem]
    TFV: Type[FuelCycleAnalysis]
    ATEC: Type[CoilArchitect]

    # Construction and calculation class declarations
    EQ: Type[AbInitioEquilibriumProblem]
    RB: Type[ReactorCrossSection]
    CAD: Type[ReactorCAD]
    n_CAD: Type[ReactorCAD]

    PlotConstructor = ReactorPlotter
    file_manager: Type[FileManager]

    # Input parameter declaration in config.py. Config values will overwrite
    # defaults in Configuration.
    default_params = SingleNull().to_records()

    def __init__(self, config, build_config, build_tweaks):

        # Initialise Reactor object with inputs
        self.config_keys = list(config.keys())
        self.build_config = build_config
        self.build_tweaks = build_tweaks

        self.params = ParameterFrame(self.default_params.to_records())
        self.prepare_params(config)

        self.nmodel = None

        # Set callbacks to use for optimisation of subsystems
        self.callbacks = {
            "TF": TF_optimiser,
            "EQ": EQ_optimiser,
            "FW": FW_optimiser,
        }

        # Final configurational defaults
        self.date = datetime.datetime.now().isoformat()
        self.plot_flag = self.build_config.get("plot_flag", False)
        self.palette = None
        self.specify_palette(BLUE)

        # Create the file manager for this reactor
        reactor_name = self.params.get("Name", "DEFAULT_REACTOR")

        reference_data_root = build_config.get(
            "reference_data_root", f"{BM_ROOT}/data/BLUEPRINT"
        )
        generated_data_root = build_config.get(
            "generated_data_root", f"{BM_ROOT}/data/BLUEPRINT"
        )
        self.file_manager = FileManager(
            reactor_name=reactor_name,
            reference_data_root=reference_data_root,
            generated_data_root=generated_data_root,
        )
        self.file_manager.build_dirs()

        self._generate_subsystem_classes(self.build_config)

    def __init_subclass__(cls):
        """
        Initialise sub-classes of Reactor, with a registry preventing the duplication
        of sub-classes in a namespace.
        """
        _registry = reactor_registry()
        if hasattr(_registry, cls.__name__):
            existing = getattr(_registry, cls.__name__)
            raise ValueError(
                f"reactor class {cls.__name__} already exists from {existing.__module__}"
            )

        super().__init_subclass__()
        setattr(_registry, cls.__name__, cls)

    def prepare_params(self, config):
        """
        Prepare the parameters so they are ready for building.
        """
        for key, val in config.items():
            if not isinstance(val, (tuple, dict)):
                config[key] = (val, "Input")
        self.params.update_kw_parameters(config)
        if self.build_config["process_mode"] == "mock":
            self.estimate_kappa_95()
            self.derive_inputs()

    def build(self):
        """
        Runs through the full reactor build sequence
        """
        print_banner()
        tic = time()

        # Run 0-1D systems code modules
        self.run_systems_code()

        self.build_0D_plasma()

        # Calculate or load preliminary plasma MHD equilibrium
        if self.build_config["plasma_mode"] == "run":
            self.create_equilibrium()
        elif self.build_config["plasma_mode"] == "read":
            self.load_equilibrium(
                self.build_config.get("plasma_filepath", None),
                self.build_config.get("reconstruct_jtor", False),
            )

        self.shape_firstwall()
        self.build_cross_section()
        self.define_HCD_strategy(method=self.build_config["HCD_method"])
        self.build_IVCs()
        self.build_vessels()
        self.build_TF_coils()

        self.build_PF_system()
        # TODO: re-shape first wall with equilibra snapshots
        self.build_ports()
        self.build_coil_cage()

        self.define_in_vessel_layout()
        self.estimate_IVC_powers()
        self.build_containments()
        self.power_balance(plot=self.plot_flag)
        # self.analyse_maintenance()
        self.life_cycle()
        self.add_parameter(
            "runtime", "Total BLUEPRINT runtime", time() - tic, "s", None, "BLUEPRINT"
        )

        self.specify_palette(BLUE)
        bluemira_print(f"Reactor designed in {time()-tic:.1f} seconds.")

    def run_systems_code(self):
        """
        Runs, reads, or mocks the systems code according to the build config dictionary.
        """
        PROCESS_solver = systems_code_solver(
            self.params,
            self.build_config,
            self.file_manager.generated_data_dirs["systems_code"],
            self.file_manager.reference_data_dirs["systems_code"],
        )

        PROCESS_solver.run()
        self.params.update_kw_parameters(PROCESS_solver.params.to_dict())

    def estimate_kappa_95(self):
        """
        Estimate maximum 95th elongation based on aspect ratio input. This is
        only valid for an EU-DEMO-like machine. This is an implicit vertical
        stability constraint on the elongation.
        """
        kappa_95 = estimate_kappa95(self.params.A, self.params.m_s_limit)
        self.params.kappa_95 = (kappa_95, "Initial Estimation")

    def derive_inputs(self):
        """
        Derive some Parameters based on the inputs.
        """
        self.params.kappa = 1.12 * self.params["kappa_95"]
        self.params.delta = 1.5 * self.params["delta_95"]

        self.params.get_param("kappa").source = "Derived Inputs"
        self.params.get_param("delta").source = "Derived Inputs"

    def __getstate__(self):
        """
        Pickling utility. Need to get rid of C objects prior to pickling.
        """
        d = super().__getstate__()
        # ReactorCAD can't be pickled so after unpickling build_cad must be
        # called
        d.pop("CAD", None)
        # ReactorCAD can't be pickled so after unpickling
        # build_neutronics_model must be called
        d.pop("nCAD", None)
        return d

    def build_0D_plasma(self):
        """
        Build a plasma object with a 0-D model
        """
        profiles = {}

        derived_params = {
            "f_DD_fus": self.params.P_fus_DD / self.params.P_fus,
            "beta_N": normalise_beta(
                self.params.beta,
                self.params.R_0 / self.params.A,
                self.params.B_0,
                self.params.I_p,
            ),
            "n_DT_reactions": n_DT_reactions(self.params.P_fus_DT),
            "n_DD_reactions": n_DD_reactions(self.params.P_fus_DD),
        }
        self.add_parameters(derived_params, "Derived")

        PlasmaClass = self.get_subsystem_class("PL")
        self.PL = PlasmaClass(self.params, profiles, self.build_config["plasma_mode"])

    def create_equilibrium(self, qpsi_calcmode=0):
        """
        Creates a reference MHD equilibrium for the Reactor.
        """
        # First make an initial TF coil shape along which to auto-position
        # some starting PF coil locations. We will design the TF later
        rin, rout = self.params["r_tf_in_centre"], self.params["r_tf_out_centre"]

        if self.params.delta_95 >= 0:
            x, z = flatD(rin, rout, 0)
        else:  # Negative triangularity
            x, z = negativeD(rin, rout, 0)
        tfboundary = Loop(x=x, z=z)
        tfboundary = tfboundary.offset(-0.5)

        profile = None

        bluemira_print("Generating reference plasma MHD equilibrium.")
        a = AbInitioEquilibriumProblem(
            self.params.R_0.value,
            self.params.B_0.value,
            self.params.A.value,
            self.params.I_p.value * 1e6,  # MA to A
            self.params.beta_p.value / 1.3,  # TODO: beta_N vs beta_p here?
            self.params.l_i.value,
            # TODO: 100/95 problem
            # TODO: This is a parameter patch... switch to strategy pattern
            self.params.kappa_95,
            1.2 * self.params.kappa_95,
            self.params.delta_95,
            1.2 * self.params.delta_95,
            -20,
            5,
            60,
            30,
            self.params.div_L2D_ib,
            self.params.div_L2D_ob,
            self.params.r_cs_in.value + self.params.tk_cs.value / 2,
            self.params.tk_cs.value / 2,
            tfboundary,
            self.params.n_PF.value,
            self.params.n_CS.value,
            c_ejima=self.params.C_Ejima.value,
            eqtype=self.params.plasma_type.value,
            rtype=self.params.reactor_type.value,
            profile=profile,
        )

        if self.params.PF_material.value == "NbTi":
            j_pf = NBTI_J_MAX
            b_pf = NBTI_B_MAX
        elif self.params.PF_material.value == "Nb3Sn":
            j_pf = NB3SN_J_MAX
            b_pf = NB3SN_B_MAX
        else:
            raise ValueError("Unrecognised material string")

        if self.params.CS_material.value == "NbTi":
            j_cs = NBTI_J_MAX
            b_pf = NBTI_B_MAX
        elif self.params.CS_material.value == "Nb3Sn":
            j_cs = NB3SN_J_MAX
            b_cs = NB3SN_B_MAX

        a.coilset.assign_coil_materials("PF", j_max=j_pf, b_max=b_pf)
        a.coilset.assign_coil_materials("CS", j_max=j_cs, b_max=b_cs)
        # NOTE: Is this doing optimisations? I don't think so, but not sure...
        a.solve(plot=self.plot_flag)
        print("")  # stdout flusher

        directory = self.file_manager.generated_data_dirs["equilibria"]
        a.eq.to_eqdsk(
            self.params["Name"] + "_eqref",
            directory=directory,
            qpsi_calcmode=qpsi_calcmode,
        )
        self.EQ = a
        self.eqref = deepcopy(a.eq)
        self.analyse_equilibrium(self.eqref)

    def load_equilibrium(self, filename=None, reconstruct_jtor=False, qpsi_calcmode=0):
        """
        Load an equilibrium from a file.
        """
        raise NotImplementedError("This method is being redesigned...")

    def analyse_equilibrium(self, eq):
        """
        Analyse an equilibrium and store important values in the Reactor parameters.
        """
        d = eq.analyse_plasma()
        lq = lambda_q(self.params.B_0, d["q_95"], self.params.P_sep, d["R_0"])
        dx_shaf = d["dx_shaf"]
        dz_shaf = d["dz_shaf"]
        shaf = np.hypot(dx_shaf, dz_shaf)

        # fmt: off
        params = [['I_p', 'Plasma current', d['Ip'] / 1e6, 'MA', None, 'equilibria'],
                  ['q_95', 'Plasma safety factor', d['q_95'], 'dimensionless', None, 'equilibria'],
                  ['Vp', 'Plasma volume', d['V'], 'm^3', None, 'equilibria'],
                  ['beta_p', 'Ratio of plasma pressure to poloidal magnetic pressure',
                  d['beta_p'], 'dimensionless', None, 'equilibria'],
                  ['li', 'Normalised plasma internal inductance', d['li'], 'dimensionless', None, 'equilibria'],
                  ['li3', 'Normalised plasma internal inductance (ITER def)', d['li(3)'], 'dimensionless', None, 'equilibria'],
                  ['Li', 'Plasma internal inductance', d['Li'], 'H', None, 'equilibria'],
                  ['Wp', 'Plasma energy', d['W'] / 1e6, 'MJ', None, 'equilibria'],
                  ['delta_95', '95th percentile plasma triangularity', d['delta_95'], 'dimensionless', None, 'equilibria'],
                  ['kappa_95', '95th percentile plasma elongation', d['kappa_95'], 'dimensionless', None, 'equilibria'],
                  ['delta', 'Plasma triangularity', d['delta'], 'dimensionless', None, 'equilibria'],
                  ['kappa', 'Plasma elongation', d['kappa'], 'dimensionless', None, 'equilibria'],
                  ['shaf_shift', 'Shafranov shift of plasma (geometric=>magnetic)', shaf, 'm', None, 'equilibria'],
                  ['lambda_q', 'Scrape-off layer power decay length', lq, 'm', None, 'Eich scaling']]
        # fmt: on
        self.add_parameters(params)
        # self.EM = EquilibriumManipulator(self.eqref)
        # self.EM.classify_legs(lq)
        self.PL.update_separatrix(eq.get_separatrix())
        self.PL.update_LCFS(eq.get_LCFS())
        self.PL.add_parameters(params)
        self.PF = PoloidalFieldCoils(self.params)
        self.PF.update_coilset(self.EQ.coilset)

    def shape_firstwall(self):
        """
        Shape a preliminary first wall around the plasma

        Notes
        -----
        Not a panelled FW..!
        """
        FirstWallProfileClass = self.get_subsystem_class("FW")
        self.FW = FirstWallProfileClass(
            self.params,
            {
                "name": self.params.Name,
                "parameterisation": self.build_config["FW_parameterisation"],
            },
        )
        bluemira_print(
            "Designing first wall with:\n"
            f"psi_n: {self.params.fw_psi_n}\n"
            f"dx: {self.params.tk_sol_ib}"
        )

        sym = self.params.plasma_type == "DN"

        if self.build_config["plasma_mode"] == "run":
            # Only used reference equilibrium created this run
            eq_name = self.eqref.filename
            self.FW.generate(
                [eq_name],
                dx=self.params.tk_sol_ib,
                psi_n=self.params.fw_psi_n,
                flux_fit=True,
                symetric=sym,
            )
            self.sf = StreamFlow(filename=self.eqref.filename)
        elif self.build_config["plasma_mode"] == "read":
            # Get all equilibria from saved files
            directory = self.file_manager.reference_data_dirs["equilibria"]
            eq_files = get_files_by_ext(directory, "json")
            eq_names = [os.path.join(directory, file) for file in eq_files]
            self.FW.generate(
                eq_names,
                dx=self.params.tk_sol_ib,
                psi_n=self.params.fw_psi_n,
                flux_fit=True,
            )
            self.sf = StreamFlow(filename=self.eqref.filename)

    def build_cross_section(self):
        """
        Build the 2-D reactor geometrical cross-section.
        """
        bluemira_print("Desiging reactor 2-D cross-section.")

        div_profile_class_name = self.build_config.get(
            "div_profile_class_name", "DivertorProfile"
        )
        to_rb = {
            "sf": self.sf,
            "VV_parameterisation": self.build_config["VV_parameterisation"],
            "div_profile_class_name": div_profile_class_name,
        }

        RadialBuildClass = self.get_subsystem_class("RB")
        self.RB = RadialBuildClass(self.params, to_rb)

        self.RB.build(self.FW)
        self.RB.get_sol(plot=False)

    def build_IVCs(self):
        """
        Build the in-vessel components (IVCs): i.e. divertor and breeding
        blanket components.
        """
        # Build the divertor
        to_div = {"geom_info": self.RB.geom["divertor"]}

        DivertorClass = self.get_subsystem_class("DIV")
        self.DIV = DivertorClass(self.params, to_div)

        # Cut off the separatrix legs
        sep = self.DIV.clip_separatrix(self.eqref.get_separatrix())
        self.PL.update_separatrix(sep)

        # Build the breeding blanket
        to_bb = {
            "inner_loop": self.RB.geom["inner_loop"],
            "blanket_outer": self.RB.geom["blanket_outer"],
            "blanket": self.RB.geom["blanket"],
            "blanket_inner": self.RB.geom["blanket_inner"],
        }

        BlanketClass = self.get_subsystem_class("BB")
        self.BB = BlanketClass(self.params, to_bb)

    def build_vessels(self):
        """
        Build the vacuum vessel and vacuum vessel thermal shield shells.
        """
        to_vv = {
            "vessel_shell": self.RB.geom["vessel_shell"],
        }
        VesselClass = self.get_subsystem_class("VV")
        self.VV = VesselClass(self.params, to_vv)

        to_ts = {"VV 2D outer": self.VV.geom["2D profile"].outer}
        ThermalShieldClass = self.get_subsystem_class("TS")
        self.TS = ThermalShieldClass(self.params, to_ts)

    def build_containments(self):
        """
        Build the cryostat and radiation shield systems.
        """
        bluemira_print("Designing cryostat, thermal shield, and radiation shield.")
        to_ts = {"TFprofile": self.TF.loops, "PFcoilset": self.PF}

        self.TS.build_cts(to_ts)
        to_cr = {
            "GS": self.ATEC.geom["feed 3D CAD"]["Gsupport"],
            "CTS": self.TS.geom["Cryostat TS"],
            "VVports": {
                "Upper port outer": self.VV.geom["Upper port"].outer,
                "Equatorial port outer": self.VV.geom["Equatorial port"].outer,
                "LP duct outer": self.VV.geom["Lower duct"].outer,
            },
        }
        CryostatClass = self.get_subsystem_class("CR")
        self.CR = CryostatClass(self.params, to_cr)

        to_rs = {"CRplates": self.CR.geom["plates"], "VVports": to_cr["VVports"]}
        RadiationShieldClass = self.get_subsystem_class("RS")
        self.RS = RadiationShieldClass(self.params, to_rs)
        # Adjust port extensions
        self.TS.adjust_ports()

        to_vv = {"CRplates": self.CR.geom["plates"], "tk": self.CR.params.g_cr_ts}
        self.VV.adjust_ports(to_vv)

    def build_ports(self):
        """
        Build port penetrations through the VV and VVTS
        """
        bluemira_print("Designing reactor ports.")
        lp_height = self.DIV.get_div_height(self.params.LPangle) + 0.2
        to_ts_build = {
            "Div_cog": self.DIV.get_div_cog(),
            "PFcoilset": self.PF,
            "TFprofile": self.TF.loops,
            "TFsection": self.TF.section,
            "lp_height": lp_height,
        }
        self.TS.build_ports(to_ts_build)

        to_vv_build = {
            "upper": self.TS.geom["Upper port"].inner,
            "eq": self.TS.geom["Equatorial port"].inner,
            "lower": self.TS.geom["Lower port"].inner,
            "lower_duct": self.TS.geom["Lower duct"].inner,
            "LP_path": self.TS.geom["LP path"],
        }
        self.VV.build_ports(to_vv_build)

    def build_TF_coils(
        self,
        ny=None,
        nr=None,
        nrippoints=None,
        objective=None,
        shape_type=None,
        wp_shape=None,
        conductivity=None,
    ):
        """
        Design and optimise the tokamak toroidal field coils.

        Parameters
        ----------
        ny: int
            WP discretisation in toroidal direction. Production runs should use
            at least ny=3
        nr: int
            WP discretisation in radial direction. Production runs should use
            at least nr=2
        nrippoints: int
            Number of points along the outer separatrix to check for ripple.
            Lower numbers for speed but careful please
        objective: str from ['L', 'E']
            The optimisation objective:
            - 'L': minimises the length of the winding pack profile. Fast.
            - 'E': minimises the stored energy of the TF coil set. Slow and
            will occasionally cause re-entrant profiles (bad for manufacture)
        shape_type: str from ['S', 'T', 'D', 'P']
            The TF coil shape parameterisation to use:
            - 'S': Spline coil shape (highly parameterised)
            - 'T': triple-arc coil shape
            - 'D': Princeton D coil shape
            - 'P': Picture frame coil shape
            - 'TP': Tapered Pictureframe coil shape
        """
        if ny is None:
            ny = self.build_tweaks["ny"]
        if nr is None:
            nr = self.build_tweaks["nr"]
        if nrippoints is None:
            nrippoints = self.build_tweaks["nrippoints"]
        if objective is None:
            objective = self.build_config["TF_objective"]
        if shape_type is None:
            shape_type = self.build_config["TF_type"]
        if wp_shape is None:
            wp_shape = self.build_config["wp_shape"]
        if conductivity is None:
            conductivity = self.build_config["conductivity"]

        to_tf = {
            "name": self.params.Name + "_TF",
            "plasma": self.PL.get_LCFS(),
            "koz_loop": self.TS.TF_koz,
            "shape_type": shape_type,
            "wp_shape": wp_shape,
            "conductivity": conductivity,
            "obj": objective,
            "ny": ny,
            "nr": nr,
            "npoints": 80,
            "nrip": nrippoints,
            "read_folder": self.file_manager.reference_data_dirs["geometry"],
            "write_folder": self.file_manager.generated_data_dirs["geometry"],
        }
        ToroidalFieldCoilsClass = self.get_subsystem_class("TF")
        self.TF = ToroidalFieldCoilsClass(self.params, to_tf)

        if self.build_config["tf_mode"] == "run":
            bluemira_print(
                f'Designing {self.build_config["TF_type"]}-type TF coils.\n'
                f"|   minimising: {objective} \n"
                f"|   subject to: {self.params.TF_ripple_limit} % ripple"
            )

            self.TF.build(self.callbacks.get("TF"))
        elif self.build_config["tf_mode"] == "read":
            bluemira_print(
                f'Loading {self.build_config["TF_type"]}-type TF coil shape' "."
            )
            try:
                self.TF.load_shape()
            except GeometryError:
                bluemira_warn(
                    "No hay una forma apropriada de TF para cargar.. hago " "una nueva!"
                )
                self.build_config["tf_mode"] = "run"
                self.build_TF_coils(ny, nr, nrippoints, objective, shape_type)
        self.add_parameters(self.TF.params.to_records())

    def build_PF_system(self, qpsi_calcmode=0):
        """
        Design and optimise the reactor poloidal field system.
        """
        callback = self.callbacks.get("EQ")
        if callback is not None:
            callback(
                self.EQ,
                self.TF,
                self.params,
                self.define_port_exclusions(),
                self.plot_flag,
            )

        for name, snap in self.EQ.snapshots.items():
            if name != "Breakdown":
                snap.eq.to_eqdsk(
                    self.params["Name"] + f"_{name}",
                    directory=self.file_manager.generated_data_dirs["equilibria"],
                    qpsi_calcmode=qpsi_calcmode,
                )

        PoloidalFieldCoilsClass = self.get_subsystem_class("PF")
        self.PF = PoloidalFieldCoilsClass(self.params)
        self.PF.update_coilset(self.EQ.coilset)

    def build_coil_cage(self):
        """
        Build the TF and PF coil cage, including support structures.
        """
        bluemira_print("Designing coil structures.")

        to_atec = {
            "tf": self.TF,
            "pf": self.PF,
            "exclusions": self.get_port_exclusions(),
            "gs_type": self.build_config["GS_type"],
        }
        CoilArchitectClass = self.get_subsystem_class("ATEC")
        self.ATEC = CoilArchitectClass(self.params, to_atec)

        self.ATEC.build()

    def define_port_exclusions(self):
        """
        Define exclusions zones for PF coils in the X-Z plane.

        Returns
        -------
        excl: list(Loop, Loop, Loop)
            The Loops of the lower, equatorial, and upper port exclusion zones
        """
        vv_tk = self.params.vv_dtk
        ts_cl = self.params.g_vv_ts
        ts_tk = self.params.tk_ts
        pf_ts_cl = self.params.g_ts_pf
        vv_cl = self.params.g_vv_ts
        clearance = 0.2 + vv_tk + ts_cl + ts_tk + vv_cl + pf_ts_cl
        tfout = Loop(x=self.TF.loops["out"]["x"], z=self.TF.loops["out"]["z"])
        v = self.VV.geom["2D profile"].inner
        d_vv = max(v.x) - min(v.x)

        # Crude minimum upper port size
        p = 2 * (d_vv / 3 + self.params.c_rm)  # 1.3*
        x_out = max(v.x) - clearance  # rOUGH
        x_in = x_out - p  # -clearance ## nEEW
        ztop = max(tfout.z) + 2
        up = Loop(x=[x_in, x_out, x_out, x_in, x_in], z=[0, 0, ztop, ztop, 0])
        # Equatorial port (Noah's Ark)
        z_up = 1
        z_down = -1
        x_out = max(tfout.x) + 2
        eq = Loop(
            x=[self.params.R_0, x_out, x_out, self.params.R_0, self.params.R_0],
            z=[z_down, z_down, z_up, z_up, z_down],
        )
        eq = eq.offset(clearance)
        # Lower port
        div_top, div_bottom = self.DIV.get_div_extrema(self.params.LPangle)
        tf_off = tfout.offset(2)
        bottom_out = point_loop_cast(div_bottom, tf_off, self.params.LPangle)
        top_out = point_loop_cast(div_top, tf_off, self.params.LPangle)

        lp = Loop(
            x=[div_bottom[0], bottom_out[0], top_out[0], div_top[0], div_bottom[0]],
            z=[div_bottom[1], bottom_out[1], top_out[1], div_top[1], div_bottom[1]],
        )
        lp = lp.offset(1.5 * clearance)
        return [lp, eq, up]

    def get_port_exclusions(self):
        """
        Calculate the actual port exclusion loops in the X-Z plane.
        Includes offset from TS.

        Returns
        -------
        exclusions: list(Loop, Loop, ..)
            The exclusion Loops
        """
        # Upper port
        ts_out = self.TS.geom["Upper port"].outer
        z_up = ts_out.z[0]
        x_in = min(ts_out.x) - self.params.g_ts_pf
        x_out = max(ts_out.x) + self.params.g_ts_pf
        up = Loop(x=[x_in, x_out, x_out, x_in, x_in], z=[0, 0, z_up, z_up, 0])

        # Equatorial port
        ts_out = self.TS.geom["Equatorial port"].outer
        x_out = ts_out.x[0]
        x_in = self.params.R_0
        z_up = max(ts_out.z) + self.params.g_ts_pf
        z_down = min(ts_out.z) - self.params.g_ts_pf
        eq = Loop(
            x=[x_in, x_out, x_out, x_in, x_in], z=[z_down, z_down, z_up, z_up, z_down]
        )

        # Lower port
        ts_out = self.TS.geom["Lower port exclusion"]
        lp = ts_out.offset(self.params.g_ts_pf)
        return [up, eq, lp]

    def define_xp_targets(self):
        """
        Re-hash of Nova `targets` structure
        """
        # TODO: merge rest of config/config_bp and handle it nicely for all
        # plasma_type
        defaults = {
            "L2D": None,
            "open": self.params.div_open,
            "graze": self.params.div_graze_angle,
            "dPlate": self.params.div_Ltarg,
            "dR": 0,
        }
        targets = {"inner": defaults, "outer": defaults}
        targets["inner"]["L2D"] = self.params.div_L2D_ib
        targets["outer"]["L2D"] = self.params.div_L2D_ob
        return targets

    def define_in_vessel_layout(self):
        """
        Define segmentation of the blanket and the divertors.
        """
        if self.params.plasma_type == "SN":
            bluemira_print("Segmenting in-vessel components.")
            up_inner = self.VV.geom["Upper port"].inner
            vv_2_d = self.VV.geom["2D profile"].inner
            bb_in = self.BB.geom["2D inner"]
            bb_out = self.BB.geom["2D outer"]
            # Outer cut
            cut = (
                max(up_inner["x"])
                + min(vv_2_d["x"])
                + self.params.g_vv_bb
                - 1.5 * self.params.c_rm
            ) / 2
            # Inner cut // dr_rm distance between outer OB and VV UP corner
            dr_rm = max(bb_out["x"]) - up_inner["x"][1]
            cut_in = cut + dr_rm
            cuto = bb_out.receive_projection([cut, 20], -90)
            cuti = bb_in.receive_projection([cut_in, 20], -90)
            omega = np.degrees(np.arctan((cuti[1] - cuto[1]) / (cuti[0] - cuto[0])))
            if omega < self.params.bb_min_angle:
                cuti = bb_in.receive_projection(cuto, -self.params.bb_min_angle)
                omega = self.params.bb_min_angle
            self.FW.panelise(bb_in, cuti)
            # First do a cut without panelled walls
            self.BB.segment_blanket(cuto, omega)
            self.BB.update_FW_profile(self.FW.geom["2D profile"])
            # Then again with panelled walls
            self.BB.segment_blanket(cuto, omega)

        else:  # Double null (blankets already segmented poloidally)
            self.FW.panelise(self.BB.geom["2D inner"], cut_location=None)
            self.BB.update_FW_profile(self.FW.geom["2D profile"])
            self.BB.split_radially()
            self.BB.build_radial_segments()

    def estimate_IVC_powers(self):
        """
        Carry out crude volumetric estimatations of the TBR and power
        deposition in the in-vessel components.
        """
        to_bc = {
            "max_TBR": self.BB.params.maxTBR,
            "ideal_shell": self.RB.geom["initial_bb"],
            "datadir": os.sep.join(
                [self.file_manager.reference_data_root, "neutronics"]
            ),
        }
        BlanketCoverageClass = self.get_subsystem_class("BC")
        self.BC = BlanketCoverageClass(self.params, to_bc)

        self.BC.calculate()
        self.add_parameter(
            "potential_TBR",
            "Potential TBR",
            self.BC.TBR,
            "dimensionless",
            None,
            "BLUEPRINT simpleneutrons",
        )

        # Add divertor non-breeding / non-HGH regions
        self.BC.add_divertor(self.RB.geom["divertor"])

        # Add auxiliary non-breeding / non-HGH regions
        for _ in range(self.HCD.NB.get_n()):
            nb = self.BC.add_plug(
                [7.5, -7.5], tor_width_f=1 / 3, pol_depth_f=1, typ="Aux"
            )
            self.HCD.NB.add_penetration(nb)
        for _ in range(self.HCD.EC.get_n()):
            ec = self.BC.add_plug([10, -10], tor_width_f=1 / 3, pol_depth_f=1, typ="Aux")
            self.HCD.EC.add_penetration(ec)

        # Re-estimate TBR now all penetrations installed
        self.BC.calculate()
        self.add_parameter(
            "TBR",
            "Estimated TBR",
            self.BC.TBR,
            "dimensionless",
            None,
            "BLUEPRINT simpleneutrons",
        )

    def calculate_TF_coil_peak_field(self):
        """
        Calculate the peak field in the TF coils, including the contributions
        of the TF coils and PF coils.

        Notes
        -----
        This calculation is done assuming a rectangular TF coil winding pack
        cross-section. The peak field is largely dominated by the toroidal field
        which is maximum at the inboard plasma-facing in the middle of the
        rectangular winding pack.
        """
        # Warn if the TF discretisation is "insufficient" for peak field calculation
        if self.TF.cage.nx * self.TF.cage.ny <= 4:
            bluemira_warn(
                "TF coil discretisation is low, peak TF field likely to be"
                "overestimated."
            )

        # Note that the cage 1st TF coil is centred on y=0!!
        # Get TF coil winding pack (at y = 0)
        wp_inner = self.TF.geom["TF WP"].inner.copy()
        wp_inner.rotate(theta=180 / self.params.n_TF, p1=[0, 0, 0], p2=[0, 0, 1])

        tf_fields = np.zeros((len(wp_inner), 3))

        # Calculate Bx, Bt, and Bz due to the TF coils at theta = pi/n_TF
        for i in range(len(tf_fields)):
            tf_fields[i] = self.TF.cage.get_field(wp_inner.xyz.T[i])

        # Gather all relevant equilibria
        eqs = [snap.eq for snap in self.EQ.snapshots.values()]

        peak_fields = np.zeros(len(eqs))

        for i, eq in enumerate(eqs):
            # Calculate Bx and Bz due to the PF coils at y = 0 and rotate
            # (these fields are axisymmetric about z)
            pf_fields = np.zeros((len(wp_inner), 3))
            pf_fields[:, 0] = eq.Bx(wp_inner.x, wp_inner.z)
            pf_fields[:, 2] = eq.Bz(wp_inner.x, wp_inner.z)
            pf_fields = qrotate(
                pf_fields, theta=np.pi / self.params.n_TF, p1=[0, 0, 0], p2=[0, 0, 1]
            )
            total_fields = tf_fields + pf_fields
            peak_fields[i] = np.max(np.sqrt(np.sum(total_fields**2, axis=1)))

        b_tf_peak = max(peak_fields)
        self.add_parameter(
            "B_tf_peak",
            "Peak field inside the TF coil winding pack",
            b_tf_peak,
            "T",
            None,
            "BLUEPRINT",
        )

    def analyse_maintenance(self):
        """
        Maintenance logistics model and technical feasibility index
        """
        self.BB.generate_RM_data(self.VV.geom["Upper port"].inner)
        to_rm = {
            "BBgeometry": self.BB.geom["rm"],
            "VV_A_enclosed": self.VV.geom["Upper port"].enclosed_area,
            "up_shift": self.VV.up_shift,
            "n_BB_seg": self.BB.n_BB_seg,
        }
        rm = RMMetrics(self.params, to_rm)
        # fmt: off
        params = [['bmd', 'Blanket mainteance duration', rm.FM, 'days', None, 'BLUEPRINT'],
                  ['dmd', 'Divertor maintenance duration', rm.DM, 'days', None, 'BLUEPRINT'],
                  ['RMTFI', 'RM Technical Feasibility Index', rm.RMTFI, 'dimensionless', None, 'BLUEPRINT']]
        # fmt: on
        self.add_parameters(params)

    def define_HCD_strategy(self, method="power"):
        """
        Define the H&CD strategy used in the design of the reactor.

        Parameters
        ----------
        method: str from ["power", "fraction", "free"]
            - [power] Sets the desired amount of heating power during the
                flattop, and then works out how much current is driven, and
                allocates the remainder to the CS and/or bootstrap.
                This logic is favoured by the EUROfusion PMU.
            - [fraction] Sets the desired flattop auxiliary current drive
                fraction, and works out the resulting plasma heating power.
            - [free] Works out how much current drive is required based on the
                CS flux swing and bootstrap performance.

        Notes
        -----
        Current drive 100% allocated to NBI during flat-top.
        L-H transition threshold power presently handled in hcd.py by 80% ECD
        and 20% NBI by default.
        """
        # TODO: Hook up Ohmic power to L-H transition during ramp-up and
        # TODO: alleviate HCD requirements on ECD and NBI..
        to_hcd = {"P_LH": self.PL.params.P_LH, "n_20": self.PL.params.rho}
        HeatingCurrentDriveClass = self.get_subsystem_class("HCD")
        self.HCD = HeatingCurrentDriveClass(self.params, to_hcd)

        if self.params["op_mode"] == "Pulsed":
            if method == "power":
                self.HCD.set_requirement("P_hcd_ss", self.params.P_hcd_ss)
                self.HCD.allocate("P_hcd_ss", f_NBI=1)
                self.HCD.build()
                hcd_current = self.HCD.NB.params.I
                self.params.f_cd_aux = hcd_current / self.params["I_p"]
                self.params.f_cd_ohm = 1 - self.params.f_bs - self.params.f_cd_aux
            elif method == "fraction":
                self.add_parameter(
                    "f_cd_aux",
                    "Auxiliary current drive fraction",
                    1 - self.params.f_ni - self.params.f_bs,
                    None,
                    None,
                )
                self.HCD.set_requirement("I_cd", self.params.I_p * self.params.f_cd_aux)
                self.HCD.allocate("I_cd", f_NBI=1)
                self.HCD.build()
            elif method == "free":
                self.params.f_cd_aux = 1 - self.params["f_bs"] - self.params.f_cd_ohm
                self.HCD.set_requirement("I_cd", self.params.I_p * self.params.f_cd_aux)
                self.HCD.allocate("I_cd", f_NBI=1)
        elif self.params["op_mode"] == "Steady-state":
            if "f_cd_ohm" in self.config_keys and self.params.f_cd_ohm != 0:
                bluemira_print(
                    "Steady-state operation cannot rely on an inductively"
                    "driven current. Setting f_cd_ohm=0."
                )
                self.params.f_cd_ohm = 0
            self.params.f_cd_aux = 1 - self.params.f_bs
            self.HCD.set_requirement("I_cd", self.params.I_p * self.params.f_cd_aux)
            self.HCD.allocate("I_cd", f_NBI=1)

    def build_TFV_system(self, n=10):
        """
        Build the tritium fuelling and vacuum (TFV) system. Calculates tritium
        start-up inventory and the so-called doubling time for the reactor.

        Parameters
        ----------
        method: str from ["run", "predict"]
            Runs given timeline(s) or predicts based off a ML model
        n: int
            The number of timelines to use
        plot: bool
            Whether or not to plot the results (dist plots)
        """
        bluemira_print(
            f"Running dynamic tritium fuel cycle model.\n" f"Monte Carlo (n={n})"
        )

        model = EUDEMOFuelCycleModel(self.params, {})
        self.TFV = FuelCycleAnalysis(model)
        life_cycle = self.life_cycle()
        time_dicts = [life_cycle.make_timeline().to_dict() for _ in range(n)]
        self.TFV.run_model(time_dicts)

        m = self.TFV.get_startup_inventory("max")
        t = self.TFV.get_doubling_time("max")
        params = [
            ["m_T_start", "Tritium start-up inventory", m, "kg", None, "BLUEPRINT"],
            ["t_d", "Tritium doubling time", t, "years", None, "BLUEPRINT"],
        ]
        self.add_parameters(params)

    def power_balance(self, plot=True):
        """
        Calculate the net electric output of the reactor, and its efficiency.

        Parameters
        ----------
        plot: bool
            If True, plot the Sankey diagram for the Reactor power balance
        """
        bluemira_print("Calculating reactor power balance.")
        to_bop = {
            "BB_P_in": self.BB.params.P_in,
            "BB_dP": self.BB.params.dP,
            "BB_T_in": self.BB.params.T_in,
            "BB_T_out": self.BB.params.T_out,
            "BBcoolant": self.BB.params.coolant,
            "multiplier": self.BB.params.mult,
            "f_decayheat": self.BB.params.f_dh,
            "nrgm": self.BB.params.bb_e_mult,
            "blkpfrac": self.BC.f_HGH,
            "divpfrac": self.BC.div_n_frac,
            "auxpfrac": self.BC.aux_n_frac,
            "vvpfrac": self.BC.params.vvpfrac,
            "P_hcd": self.HCD.get_heating_power(),
            "P_hcd_ec": self.HCD.EC.params.P_h_ss,
            "P_hcd_nb": self.HCD.NB.params.P_h_ss,
            "P_hcd_el": self.HCD.get_electric_power(),
        }
        BalanceOfPlantClass = self.get_subsystem_class("BOP")
        self.BOP = BalanceOfPlantClass(self.params, to_bop)

        p_el_net, eta = self.BOP.build()
        params = [
            ["P_el_net", "Net electric power", p_el_net, "MW", None, "BLUEPRINT"],
            ["eta_plant", "Plant efficiency", eta * 100, "%", None, "BLUEPRINT"],
        ]
        self.add_parameters(params)
        if plot:
            self.BOP.plot()

    def life_cycle(self, learning_strategy=None, availability_strategy=None):
        """
        Define a DEMO timeline and lifecycle, with random dwell durations
        to match target availability.

        Parameters
        ----------
        learning_strategy: Optional[LearningStrategy]
            Strategy to use to progress operational availability
        availability_strategy: Optional[AvailabilityStrategy]
            Strategy to distribute unplanned downtimes during operation
        """
        if learning_strategy is None:
            learning_strategy = GompertzLearningStrategy(
                learn_rate=1.0,
                min_op_availability=self.params.a_min,
                max_op_availability=self.params.a_max,
            )
        if availability_strategy is None:
            availability_strategy = LogNormalAvailabilityStrategy(sigma=2.0)

        lc_in = {
            "read_only": False,
            "datadir": self.file_manager.reference_data_root,
        }
        life_cycle = LifeCycle(
            self.params, learning_strategy, availability_strategy, lc_in
        )
        return life_cycle

    def build_CAD(self):
        """
        Create the CAD model for the reactor.
        """
        bluemira_print("Building reactor 3-D CAD geometry.")
        self.CAD = ReactorCAD(self)
        self.CAD.set_palette(BLUE)

    def show_CAD(self, pattern="threequarter"):
        """
        Show the Reactor CAD model.

        Parameters
        ----------
        pattern: str
            Global patterning to display
            ['full', 'half', 'threequarter', 'third', 'quarter', 'sector']
            or alternatively
            'ffhhttqqss..' for varied patterning
        """
        if self.CAD is None:
            # Check if CAD is already built (for typechecking and no-typing)
            self.build_CAD()
        self.CAD.display(pattern)

    def save_CAD_model(self, pattern="sector"):
        """
        Save the Reactor CAD model as a STEP assembly.
        """
        if self.CAD is None:
            self.build_CAD()
        self.CAD.pattern(pattern)
        bluemira_print("Exporting the reactor CAD to a STEP assembly file.")
        model_name = self.params.Name + "_CAD_MODEL"
        path = self.file_manager.get_path("CAD", model_name)
        self.CAD.save_as_STEP_assembly(path)

    def build_neutronics_model(self):
        """
        Construit un modèle neutronique complet du réacteur (3-D, 360°) et
        initialise une simulation neutronique.
        """
        bluemira_print("Building 3-D 360° neutronics model.")
        self.n_CAD = ReactorCAD(self, slice_flag=True, neutronics=True)

        stlfolder = self.file_manager.generated_data_dirs["neutronics"]
        self.n_CAD.save_as_STL(stlfolder)
        self.check_neutronics_CAD(stlfolder)

        pdict = {
            "R_0": self.params.R_0,
            "a": self.params.R_0 / self.params.A,
            "T": self.params.T_e,
            "delta": self.params.delta_95,
            "kappa": self.params.kappa_95,
            "dz": 0,
        }
        del pdict

        # TODO: replace deprecated Serpent II with OpenMC nmodel
        self.nmodel = NotImplemented

    @staticmethod
    def check_neutronics_CAD(stlfolder):
        """
        Check that the STL neutronics models are OK.
        """
        bluemira_print("Checking STL meshes for quality.")
        result = check_STL_folder(stlfolder)
        if all(result.values()):
            bluemira_print("All meshes OK!")
        else:
            txt = "As seguintes partes são ruins:\n"
            for k in result:
                txt += f"\t{k}\n"
            bluemira_warn(txt)

    def show_neutronics_CAD(self, **kwargs):
        """
        Shows the neutronics CAD model in 360°
        No patterning available, global_pattern set to single sector, to avoid
        patterning fully rotated parts
        """
        if self.n_CAD is None:
            self.n_CAD = ReactorCAD(self, slice_flag=True, neutronics=True)
            self.n_CAD.set_palette(BLUE)
        self.n_CAD.display(**kwargs)

    # TODO: Confirm deprection
    def run_neutronics_model(self):
        """
        Runs the global neutronics model for the Reactor
        """
        raise NotImplementedError
        # if self.nmodel is None:
        #     self.build_neutronics_model()
        # bluemira_print("Running 3-D 360° OpenMC neutronics model.")
        # self.nmodel.run()

    # =========================================================================
    # Plotting methods
    # =========================================================================

    @property
    def _plotter(self):
        """
        Provides a ReactorPlotter object as a property of the Reactor
        """
        return ReactorPlotter(self, palette=self.palette)

    def plot_radial_build(self, width=1.0):
        """
        Plots a 1-D vector of the radial build output from the systems code
        """
        self._plotter.plot_1D(width=width)

    def plot_xz(self, x=None, z=None, show_eq=False):
        """
        Plots the X-Z cross-section of the reactor through the middle of a
        sector. Colors will be ditacted by the reactor palette object.

        Parameters
        ----------
        x: (float, float)
            The range of x coordinates to plot
        z: (float, float)
            The range of z coordinates to plot
        show_eq: bool (default = False)
            Whether or not to overlay plot an equilibrium

        Note
        ----
        TF coils are plotted in the same plane and not rotated by
        n_TF/2
        """
        if z is None:
            z = [-22, 15]
        if x is None:
            x = [0, 22]
        self._plotter.plot_xz(x=x, z=z, show_eq=show_eq)

    def plot_xy(self, x=None, y=None):
        """
        Plots the midplane x-y cross-section of the reactor as seen from above
        the upper port.

        Parameters
        ----------
        x: (float, float)
            The range of x coordinates to plot
        y: (float, float)
            The range of y coordinates to plot
        """
        if y is None:
            y = [-8, 8]
        if x is None:
            x = [1, 20]
        self._plotter.plot_xy(x=x, y=y)

    def specify_palette(self, global_palette):
        """
        Specifies a global colour palette to use in plotting and CAD

        Parameters
        ----------
        global_palette: dict
            The dictionary of {part: color}
        """
        keys = ["RS", "CR", "TS", "VV", "BB", "ATEC", "DIV", "TF", "PF", "PL", "HCD"]
        diff = set(keys).difference(global_palette.keys())
        if len(diff) != 0:
            raise ValueError(f"Palette dict missing {diff} keys")
        self.palette = force_rgb(global_palette)

    def save(self, path=None):
        """
        Save the Reactor object as a pickle file.

        Parameters
        ----------
        path: Union[None, str]
            The folder in which to save the Reactor. Will default to the generated_data
            folder
        """
        if path is None:
            path = os.sep.join(
                [
                    self.file_manager.generated_data_dirs["root"],
                    self.params.Name + ".pkl",
                ]
            )

        original_module = self.__class__.__module__
        try:
            self.__class__.__module__ = _registry_name
            super().save(path)
        finally:
            self.__class__.__module__ = original_module
        return path

    @classmethod
    def load(cls, path, reference_data_root=None, generated_data_root=None):
        """
        Loads a Reactor object from a pickle file.

        Parameters
        ----------
        path: str
            Full path with filename
        reference_data_root : str, optional
            The reference data root to use, by default None.
            If None then the root is taken from the loaded reactor.
            Note this path must exist locally.
        generated_data_root : str, optional
            The generated data root to use, by default None

        Returns
        -------
        Reactor
            The loaded Reactor.
        """
        reactor = super().load(path)
        if reference_data_root is None:
            reference_data_root = reactor.file_manager.reference_data_root
        if generated_data_root is None:
            generated_data_root = reactor.file_manager.generated_data_root
        reactor.file_manager = FileManager(
            reactor_name=reactor.params.Name.value,
            reference_data_root=reference_data_root,
            generated_data_root=generated_data_root,
        )
        reactor.file_manager.build_dirs(create_reference_data_paths=True)
        return reactor

    def config_to_json(self, output_path: Union[str, Path], **kwargs):
        """
        Saves reactor default parameters, parameter diff, build config, and build tweaks
        to JSON in the format usable by the Configurable Reactor class.

        Parameters
        ----------
        output_path: str
            Directory path in which to store outputs.

        Notes
        -----
        The output files will be saved as:
            REACTORNAME_template.json
            REACTORNAME_config.json
            REACTORNAME_build_config.json
            REACTORNAME_build_tweaks.json
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)

        self.default_params.to_json(
            output_path=output_path.joinpath(self.params.Name + "_template.json"),
            verbose=True,
        )

        config_diff = self.default_params.diff_params(self.params)
        config_diff.to_json(
            output_path=output_path.joinpath(self.params.Name + "_config.json")
        )

        indent = kwargs.pop("indent", 2)

        json_writer(
            self.build_config,
            output_path.joinpath(self.params.Name + "_build_config.json"),
            indent=indent,
            **kwargs,
        )
        json_writer(
            self.build_tweaks,
            output_path.joinpath(self.params.Name + "_build_tweaks.json"),
            indent=indent,
            **kwargs,
        )


_registry_name = "BLUEPRINT.reactor._registry"
_registry = None


def reactor_registry():
    """
    Reactor registry function. Prevents duplicate sub-classes.
    """
    global _registry
    if _registry is None:
        _registry = ModuleType(_registry_name)
        sys.modules[_registry_name] = _registry
    return _registry


class ConfigurableReactor(Reactor):
    """
    Creates a Reactor Class Object from JSON files using a template config and a config
    file from which to overide specified parameters with specified values.
    """

    default_params = []

    def __init__(
        self,
        template_config: ParameterFrame,
        config: dict,
        build_config: dict,
        build_tweaks: dict,
    ):
        self.default_params = template_config
        super().__init__(config, build_config, build_tweaks)

    @staticmethod
    def load_config(name, path):
        """
        Load config form JSON file

        Parameters
        ----------
        name: str
           User facing name for file

        Returns
        -------
        dict
            JSON file as dictionary

        Raises
        ------
        FileNotFoundError

        """
        if isinstance(path, str):
            path = Path(path)

        if path.exists():
            with open(path, "r") as fh:
                return json.load(fh, cls=CommentJSONDecoder)
        else:
            raise FileNotFoundError(f"Could not find {name} at {path}")

    @classmethod
    def from_json(
        cls,
        template_config_path: Union[str, PosixPath],
        config_path: Union[str, PosixPath],
        build_config_path: Union[str, PosixPath],
        build_tweaks_path: Union[str, PosixPath],
    ):
        """
        Creates a Reactor Object from four JSON files.

        Parameters
        ----------
        template_config_path: Union[str, PosixPath]
            Full path with filename pointing to the template config file.
        config_path: Union[str, PosixPath]
            Full path with filename pointing to the overide config file.
        build_config_path: Union[str, PosixPath]
            Full path with filename pointing to the build config file.
        build_tweaks_path: Union[str, PosixPath]
            Full path with filename pointing to the build tweaks file.

        Returns
        -------
        Reactor
            The configured Reactor Object.
        """
        if isinstance(template_config_path, str):
            template_config_path = Path(template_config_path)

        if template_config_path.exists():
            template_config = ParameterFrame.from_json(template_config_path)
        else:
            raise FileNotFoundError(
                f"Could not find template configuration at {template_config_path}"
            )

        config = cls.load_config("configuration", config_path)
        build_config = cls.load_config("build configuration", build_config_path)
        build_tweaks = cls.load_config("build tweaks", build_tweaks_path)

        return cls(template_config, config, build_config, build_tweaks)

    @classmethod
    def from_directory(cls, config_dir: Union[str, PosixPath], reactor_name: str):
        """
        Creates a Reactor Object from four JSON files specified by name and directory.

        Parameters
        ----------
        config_dir: Union[str, PosixPath]
            Directory path in which the input files are contained.
        reactor_name : str
            The reactor name used as the filename prefix.

        Returns
        -------
        Reactor
            The configured Reactor Object.
        """
        if isinstance(config_dir, str):
            config_dir = Path(config_dir)

        if config_dir.is_dir():
            template_path = config_dir.joinpath(reactor_name + "_template.json")
            config_path = config_dir.joinpath(reactor_name + "_config.json")
            build_config_path = config_dir.joinpath(reactor_name + "_build_config.json")
            build_tweaks_path = config_dir.joinpath(reactor_name + "_build_tweaks.json")

            return cls.from_json(
                template_path, config_path, build_config_path, build_tweaks_path
            )
        else:
            raise FileNotFoundError(
                f"Specified config directory not a directory: {config_dir}"
            )
