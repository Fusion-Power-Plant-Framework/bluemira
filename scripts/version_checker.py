"""
Validate an input version string using the packaging.version module, and
print the formatted version.
"""

from sys import argv

from packaging import version

v = version.Version(argv[1])

print(f"{str(v)[1:] if str(v).startswith('v') else v}")
