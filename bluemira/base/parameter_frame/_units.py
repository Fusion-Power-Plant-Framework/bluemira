# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pint

from bluemira.base.constants import ANGLE, raw_uc, ureg

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pint import Quantity

    from bluemira.base.parameter_frame._parameter import ParamDictT


def _validate_units(param_data: ParamDictT, value_type: Iterable[type]):
    try:
        quantity = ureg.Quantity(param_data["value"], param_data["unit"])
    except ValueError:
        try:
            quantity = ureg.Quantity(f"{param_data['value']}*{param_data['unit']}")
        except pint.errors.PintError as pe:
            if param_data["value"] is None:
                quantity = ureg.Quantity(
                    1 if param_data["unit"] in {None, ""} else param_data["unit"]
                )
            else:
                raise ValueError("Unit conversion failed") from pe
        else:
            param_data["value"] = quantity.magnitude
        param_data["unit"] = quantity.units
    except KeyError as ke:
        raise ValueError("Parameters need a value and a unit") from ke
    except TypeError:
        if param_data["value"] is None:
            # dummy for None values
            quantity = ureg.Quantity(
                1 if param_data["unit"] in {None, ""} else param_data["unit"]
            )
        elif isinstance(param_data["value"], bool | str):
            param_data["unit"] = "dimensionless"
            return param_data
        else:
            raise

    return _ensure_SI_unit_system(quantity, param_data, value_type)


def _ensure_SI_unit_system(
    quantity: Quantity, param_data: ParamDictT, value_type: Iterable[type]
) -> ParamDictT:

    if quantity.units != ureg.dimensionless:
        unit_q = _remake_units(quantity)
    else:
        unit_q = quantity

    if unit_q.units != quantity.units and param_data["value"] is not None:
        # Convert to new units
        val = raw_uc(quantity.magnitude, quantity.units, unit_q.units)
        if isinstance(param_data["value"], int) and int in value_type:
            val = int(val)
        param_data["value"] = val

    param_data["unit"] = f"{unit_q.units:~P}"
    return param_data


def _remake_units(quantity: Quantity) -> pint.Quantity:
    """
    Reconstruct unit from its dimensionality.

    Parameters
    ----------
    quantity:
        The quantity to reconstruct

    Returns
    -------
    :
        The quantity in new units

    Notes
    -----
    The quantity of the conversion is not important here.
    We just want the custom SI version of the input
    The value conversion is done later.
    """
    unit_list, ind_list, fix_list = [], [], []
    for no, (unit, multiplier) in enumerate(quantity.units._units.items()):
        q = ureg.Quantity(unit)
        if q.units not in {ureg.fpy, ureg.dpa}:
            q = q.to_preferred()  # individually preferred (SI)
        if q.units in {ureg.fpy, ureg.dpa} or not q.dimensionality:
            # is not commutative or is angle (as they're dimensionless in pint)
            fix_list.append(no)
        else:
            ind_list.append(no)
        unit_list.append(ureg.Quantity(f"{q.magnitude}({q.units})^{multiplier}"))

    return _combine_commutative(unit_list, ind_list) * _convert_non_commutative(
        unit_list, fix_list
    )


def _convert_non_commutative(
    unit_list: list[Quantity], filter_index: list[int]
) -> Quantity:
    """Converts angle units and combines non commutative units"""  # noqa: DOC201

    # There is only one unit in _units of 'i' therefore extract the key/value
    def get_key(i: Quantity) -> str:
        return next(iter(i._units.keys()))

    def get_exp(i: Quantity) -> str:
        return next(iter(i._units.values()))

    filtered_list = [unit_list[i] for i in filter_index]
    for no, i in enumerate(filtered_list):
        if not i.dimensionality and get_key(i) != ureg.dpa:
            if get_key(i) in {ureg.steradian, ureg.square_degree}:
                raise NotImplementedError("Solid angle conversion not supported")

            filtered_list[no] = i.to(ureg.Unit(f"{ANGLE}**{get_exp(i)}"))

    return math.prod(filtered_list)


def _combine_commutative(unit_list: list[Quantity], filter_index: list[int]) -> Quantity:
    """
    Combine commutative units

    Notes
    -----
    Uses pints milp optimisation which can over combine eg W/m^2 -> kg/s^3.
    The optimisation minimises combinations eg the "knapsack problem".
    This function prioritise shorter original over new which could be improved in future
    """  # noqa: DOC201
    filtered_list = [unit_list[i] for i in filter_index]

    quantity = math.prod(filtered_list)
    if not isinstance(quantity, ureg.Quantity):
        return quantity

    # Prefer the users SI converted unit
    # if the number of constituent units is <= the number in the new unit
    pq = quantity.to_preferred()
    if len(quantity._units.keys()) <= len(pq._units.keys()):
        return quantity
    return pq
