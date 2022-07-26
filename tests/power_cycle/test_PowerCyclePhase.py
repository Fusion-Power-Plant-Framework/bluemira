# Import
from pprint import pprint

from _TIAGO_FILES_.Tools import Tools as imported_tools

import bluemira.base.constants as constants
from bluemira.power_cycle.timeline import PowerCyclePhase as imported_class

# Header
imported_tools.print_header("Test PowerCyclePhase")

# Error messages
class_errors = imported_class._errors
pprint(class_errors)

# Test data
test_name = "flat-top"
test_label = "ftt"
test_dependency = "ss"
test_duration = constants.raw_uc(2, "hour", "second")

# Create instance of PowerCyclePhase
test_instance = imported_class(test_name, test_label, test_dependency, test_duration)
pprint(vars(test_instance))

# Test validation method
check_instance = imported_class._validate(test_instance)
"check_instance = imported_class._validate(test_name)"
pprint("No errors raised on validation!")
