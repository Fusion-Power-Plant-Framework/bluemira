# Import general packages
from pprint import pprint

# Import BLUEMIRA packages
import bluemira.base.constants as constants

# Import Power Cycle packages
from bluemira.power_cycle.base import PowerCycleUtilities
from bluemira.power_cycle.timeline import PowerCyclePhase, PowerCyclePulse

# Header
PowerCycleUtilities.print_header("Test PowerCyclePulse")

# Dwell-2-Flat
d2f = PowerCyclePhase(
    "Dwell-2-Flat",
    "d2f",
    "tt",
    constants.raw_uc(2, "minute", "second"),
)

# Flat-Top
ftt = PowerCyclePhase(
    "Flat-Top",
    "ftt",
    "ss",
    constants.raw_uc(2, "hour", "second"),
)

# Flat-2-Dwell
f2d = PowerCyclePhase(
    "Flat-2-Dwell",
    "f2d",
    "tt",
    constants.raw_uc(2, "minute", "second"),
)

# Dwell-Time
dwl = PowerCyclePhase(
    "Dwell-Time",
    "dwl",
    "ss",
    constants.raw_uc(10, "minute", "second"),
)


# Create instance of PowerCyclePulse
test_name = "Generic Pulse"
test_set = [d2f, ftt, f2d, dwl]
test_instance = PowerCyclePulse(test_name, test_set)
pprint(vars(test_instance))

# Test validation method
check_instance = PowerCyclePulse._validate(test_instance)
"check_instance = PowerCyclePulse._validate(test_name)"
pprint("No errors raised on validation!")

# Test duration methods
test_duration = test_instance.duration()
pprint(test_duration)
test_total = test_instance.total_duration()
pprint(test_total)
