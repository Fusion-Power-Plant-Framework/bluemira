materials
=========

Bluemira materials are defined in a way that is intended to be flexible while allowing
material properties to be used and accessed by the code. This is achieved by providing
a small number of material classes and allowing specific materials to be represented
by dynamic configuration. It leverages functionality provided by
`neutronics_material_maker <https://github.com/fusion-energy/neutronics_material_maker>`_,
while expanding to include material properties that are relevant for engineering as well
as neutronics.

Material Definitions
--------------------

A material in bluemira is defined by specifying the ``material_class``, the composition,
and the corresponding properties. This can be done either by creating the materials
directly via in code, creating the materials in code via dictionaries, or by loading the
materials from json files. The latter approach, to load from files, is the recommended
approach as it allows the material data to be sourced for a particular design at run
time.

The Material Cache
******************

Materials are passed around in bluemira from the material cache. A new cache can be
created as below, where we'll describe the contents of the materials.json and
mixtures.json in subsequent sections:

.. code-block:: python

    from bluemira.materials import MaterialCache
    from bluemira.base.file import get_bluemira_path

    path = get_bluemira_path('materials', subfolder='data')
    material_cache = MaterialCache()
    material_cache.load_from_file(path + "/materials.json")
    material_cache.load_from_file(path + "/mixtures.json")

Materials are then retrieved from the cache by name, allowing the relevant material to be
accessed dynamically depending on how a particular component or analysis is configured at
runtime:

.. code-block:: python

    bronze = material_cache.get_material("Bronze")

When defining a material via a file, the name by which the material is accessed is given
by the key in the definition. When a material is defined by a dictionary, then the name
must be passed into the load call in order to extract the relevant information from the
input dictionary.

.. code-block:: python

    mat_dict = {
        "H2O": {
            "material_class": "Liquid",
            "symbol": "H2O",
            "density": "PropsSI('D', 'T', temperature, 'P', pressure, 'Water')"
        }
    }
    material_cache.load_from_dict("H2O", mat_dict)

Material Classes
****************

The available material classes are defined as classes in
:py:mod:`bluemira.materials.material`. Each material class has its own set of available
properties and means for defining composition. The following describes the core classes
and their composition definitions. Note that most classes need to be provided with at
least a density definition in addition to the composition information - this can be
either a scalar value, or an equation that is dependent on temperature (and pressure in
the case of liquids), and will be described more when we discuss
:ref:`material-properties`.

- :py:class:`bluemira.materials.material.Void`: A dummy material class for defining
  void regions. This class has no composition and is taken to be zero density. The
  ``Void`` material is simply defined as below:

.. code-block:: json

    {
        "Void": {
            "material_class": "Void"
        }
    }

- :py:class:`bluemira.materials.material.MassFractionMaterial`: A material composed of
  fractions of individual elements. The elements are defined via a dictionary mapping the
  element's symbol to the fractional composition within that material.

.. code-block:: json

    {
        "Bronze": {
            "material_class": "MassFractionMaterial",
            "elements": {"Cu": 0.95, "Sn": 0.05},
            "density": 8877.5
        }
    }

- :py:class:`bluemira.materials.material.Liquid`: A material defined by it's chemical
  symbol, having a density that can be dependent on pressure.

.. code-block:: json

    {
        "H2O": {
            "material_class": "Liquid",
            "symbol": "H2O",
            "density": "PropsSI('D', 'T', temperature, 'P', pressure, 'Water')"
        }
    }

- :py:class:`bluemira.materials.material.UnitCellCompound` A material defined by a
  lattice structure with a composition given by the chemical symbol. The lattice must be
  defined with a volume of the unit cell in cm\ :sup:`3` and a number of atoms per unit
  cell. It can also have an optional packing fraction, defining the amount of the
  compound that is filled with void, and Li\ :sup:`6` enrichment fraction.

.. code-block:: json


    {
        "Li2SiO3": {
            "material_class": "UnitCellCompound",
            "symbol": "Li2SiO3",
            "volume_of_unit_cell_cm3": 0.23632e-21,
            "atoms_per_unit_cell": 4
        }
    }

- :py:class:`bluemira.materials.material.Plasma`: A material defined by its constituent
  isotopes. The relative composition per each isotope is given via a dictionary mapping
  the isotope symbol to the fractional composition.

.. code-block:: json

    {
        "D-T Plasma": {
            "material_class": "Plasma",
            "isotopes": {"H2": 0.5, "H3": 0.5}
        }
    }

.. _material-properties:

Material Properties
*******************

As similar materials can be found in a variety of conditions within a fusion reactor,
bluemira has the ability to define a variety of material properties that can be derived
across temperature distributions and, in the case of liquids, pressure distributions. For
the available material properties for the various material classes, please refer to
:py:mod:`bluemira.materials.material`.

A material with a scalar set of properties can be defined and loaded into our material
cache as below:

.. code-block:: python

    from bluemira.materials import MaterialCache

    mat_dict = {
        "Bronze": {
          "material_class": "MassFractionMaterial",
          "elements": {"Cu": 0.95, "Sn": 0.05},
          "density": 8877.5,
          "poissons_ratio": 0.33,
        }
    }

    material_cache = MaterialCache()
    material_cache.load_from_dict("Bronze", mat_dict)
    bronze = material_cache.get_material("Bronze")
    temperature = 300  # Kelvin
    print(f"Density of bronze at {temperature} K: {bronze.rho(temperature)}")
    print(f"Poisson's ratio of bronze at {temperature} K: {bronze.mu(temperature)}")

As you may note in the above, the material properties have been defined using a verbose
description of the property by then access using a shorthand form, which corresponds to
the symbol that will usually be used for that property in equations. The properties can
also be accessed using the long form, as they are defined, apart from density. This is a
known limitation on the interaction with neutronics processing. To handle this, it is
also possible to set the temperature of the material directly:

.. code-block:: python

    temperature= 500  # Kelvin
    bronze.temperature = temperature
    print(f"Density of bronze at {temperature} K: {bronze.density()}")
    print(f"Poisson's ratio of bronze at {temperature} K: {bronze.poissons_ratio(temperature)}")

This is not so useful for a material property that is temperature independent, so let's
define a material with some properties that vary with temperature in different ways (the
element composition here has been reduced down for simplicity).

.. code-block:: python

    mat_dict = {
        "SS316-LN": {
            "material_class": "MassFractionMaterial",
            "elements": {
                "Cr": 0.18,
                "Fe": 0.64,
                "Mn": 0.02,
                "Mo": 0.03,
                "Ni": 0.13,
            },
            "poissons_ratio": 0.33,
            "coefficient_thermal_expansion": {
                "value": "polynomial.Polynomial([15.13, 7.93e-3, -3.33e-6])(to_celsius(temperature))",
                "temp_min_celsius": 20,
                "temp_max_celsius": 1000,
                "reference": "ITER_D_222RLN v3.3 Equation 40"
            },
            "youngs_modulus": {
                "value": "0.001 * (201660 - 84.8 * to_celsius(temperature))",
                "temp_min_celsius": 20,
                "temp_max_celsius": 700,
                "reference": "ITER_D_222RLN v3.3 Equation 41"
            },
            "density": {
                "value": "interp(to_celsius(temperature), [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800], [7930, 7919, 7899, 7879, 7858, 7837, 7815, 7793, 7770, 7747, 7724, 7701, 7677, 7654, 7630, 7606, 7582])",
                "temp_min_celsius": 20,
                "temp_max_celsius": 800,
                "reference": "ITER_D_222RLN v3.3 Table A.S03.2.4-1"
            }
        }
    }

There are a few things to pick out here:

- It is possible to use a limited set of functions dynamically in these material
  property definitions. These are derived from `asteval <https://newville.github.io/asteval/>`_,
  with the extended numpy support enabled. This lets us perform interpolations and
  define polynomial functions.
- The temperature can be defined in this dynamic functions via ``temperature`` or
  or ``to_celsius(temperature)`` to allow the functional forms to used directly in K or to be
  converted from :sup:`o`\ C.
- We have defined temperature ranges (in celsius or kelvin) over which the functional
  forms are valid.
- We have specified a reference so the source of the functional form is kept.

.. code-block:: python

    material_cache.load_from_dict("SS316-LN", mat_dict)
    steel = material_cache.get_material("SS316-LN")

    temperature = 500  # Kelvin
    print(f"Density of steel at {temperature} K: {steel.rho(temperature)}")
    print(f"CTE of steel at {temperature} K: {steel.CTE(temperature)}")
    print(f"Young's modulus of steel at {temperature} K: {steel.E(temperature)}")

    temperature = 600  # Kelvin
    print(f"Density of steel at {temperature} K: {steel.rho(temperature)}")
    print(f"CTE of steel at {temperature} K: {steel.CTE(temperature)}")
    print(f"Young's modulus of steel at {temperature} K: {steel.E(temperature)}")

Liquids can be pressurised, so have a density property that is also dependent on pressure
(in Pa).

.. code-block:: python

    mat_dict = {
        "H2O": {
            "material_class": "Liquid",
            "symbol": "H2O",
            "density": "PropsSI('D', 'T', temperature, 'P', pressure, 'Water')"
        }
    }

    material_cache.load_from_dict("H2O", mat_dict)
    water = material_cache.get_material("H2O")
    print(f"Density of water at {400} K, {10e6} Pa: {water.rho(400, 10e6)}")

Mixtures
********

It is often convenient to simplify some components that are under analysis by assuming
that they are made from homogeneous mixtures of materials. This is supported in bluemira
by the :py:class:`bluemira.materials.mixtures.HomogenisedMixture` class. Mixtures must
always be defined after materials, hence why the material and mixtures json files are
loaded separately. Mixtures do not have the material properties associated with
individual materials, but the underlying materials can be accessed for use in
calculations by taking averages of the material properties, for example.

Mixtures are defined by providing the fractional composition of the constituent materials
keyed by the names of the materials.

.. code-block:: python

    mat_dict = {
        "Steel Water 60/40": {
            "material_class": "HomogenisedMixture",
            "materials": {
                "SS316-LN": 0.6,
                "H2O": 0.4
            },
            "temperature": 293.15
        }
    }
    material_cache.load_from_dict("Steel Water 60/40", mat_dict)
    steel_water = material_cache.get_material("Steel Water 60/40")
