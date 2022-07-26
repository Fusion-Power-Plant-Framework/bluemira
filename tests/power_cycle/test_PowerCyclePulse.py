# Import
from pprint import pprint

from _TIAGO_FILES_.Tools import Tools as imported_tools

import bluemira.base.constants as constants
from bluemira.power_cycle.timeline import PowerCyclePhase as imported_class_1
from bluemira.power_cycle.timeline import PowerCyclePulse as imported_class_2

# Header
imported_tools.print_header("Test PowerCyclePulse")

# Dwell-2-Flat
d2f = imported_class_1(
    "Dwell-2-Flat",
    "d2f",
    "tt",
    constants.raw_uc(2, "minute", "second"),
)

# Flat-Top
ftt = imported_class_1(
    "Flat-Top",
    "ftt",
    "ss",
    constants.raw_uc(2, "hour", "second"),
)

# Flat-2-Dwell
f2d = imported_class_1(
    "Flat-2-Dwell",
    "f2d",
    "tt",
    constants.raw_uc(2, "minute", "second"),
)

# Dwell-Time
dwl = imported_class_1(
    "Dwell-Time",
    "dwl",
    "ss",
    constants.raw_uc(10, "minute", "second"),
)


# Create instance of PowerCyclePulse
test_name = "Generic Pulse"
test_set = [d2f, ftt, f2d, dwl]
test_instance = imported_class_2(test_name, test_set)
pprint(vars(test_instance))

# Test validation method
check_instance = imported_class_2._validate(test_instance)
"check_instance = imported_class_2._validate(test_name)"
pprint("No errors raised on validation!")

# Test duration methods
test_duration = test_instance.duration()
pprint(test_duration)
test_total = test_instance.total_duration()
pprint(test_total)
