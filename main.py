
import os

from core.sim import Sim

"""This example is designed to demonstrate how to structure user-designed aspects of Chronos. The majority of the 
important code can be found in the vehicles directory under controllers/vehicles. In many cases, when determining the 
function of some code, it is important to reference both the core module and the controller one. Ie: core/states and 
controllers/vehicles/states. In most cases, the controller functions build on, or inherit from, the core functions by 
design. """

config_path = "vehicle_demo_hybrid/config_gui.ini"

sim = Sim()

sim.run(config_path)
