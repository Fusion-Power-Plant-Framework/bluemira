from sys import argv

from packaging import version

v = version.Version(argv[1])

print(f"{v.__str__()[1:] if v.__str__().startswith('v') else v.__str__()}")
