# Import general packages
from pprint import pprint

# Import BLUEMIRA packages
import bluemira.base.constants as constants

# Import Power Cycle packages
from bluemira.power_cycle.base import print_header
from bluemira.power_cycle.timeline import PowerCyclePhase

# Header
print_header("Test PowerCyclePhase")

# Error messages
class_errors = PowerCyclePhase._errors
pprint(class_errors)

# Test data
test_name = "flat-top"
test_label = "ftt"
test_dependency = "ss"
test_duration = constants.raw_uc(2, "hour", "second")

# Create instance of PowerCyclePhase
test_instance = PowerCyclePhase(
    test_name,
    test_label,
    test_dependency,
    test_duration,
)
pprint(vars(test_instance))

# Test validation method
check_instance = PowerCyclePhase._validate(test_instance)
"check_instance = PowerCyclePhase._validate(test_name)"
pprint("No errors raised on validation!")
