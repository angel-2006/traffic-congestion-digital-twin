import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("SUMO_HOME is not set properly")

import traci
import sumolib

print("SUCCESS: SUMO + TraCI + sumolib are working correctly!")
print("SUMO_HOME =", os.environ["SUMO_HOME"])