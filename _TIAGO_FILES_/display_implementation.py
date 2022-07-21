# Suggestion to implement display and display units for parameters in
# bluemira

"""
# ...above slots

    __display_unit = None
   # ...somewhere in class
    @property
    def _display_unit(self):
        return self.__display_unit
    @_display_unit.setter
    def _display_unit(self, unit):
        self.__display_unit = _unitify(unit)
    def display(unit=None, constant=None):
        if unit is not None:
            return raw_uc(self.value, unit), _unitify(unit)
"""
