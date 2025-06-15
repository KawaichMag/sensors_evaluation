from Objects import Sensor, draw_plan
import math

drone_size = (670, 810) # In mm
sensors = [
    Sensor((-670/2, 810/4), 40*100, math.radians(75)),
    Sensor((670/2, 810/4), 40*100, math.radians(75)),
    Sensor((-670/2, -810/4), 40*100, math.radians(75)),
    Sensor((670/2, -810/4), 40*100, math.radians(75)),

    Sensor((670/4, 810/2), 40*100, math.radians(65)),
    Sensor((670/4, -810/2), 40*100, math.radians(65)),
    Sensor((-670/4, 810/2), 40*100, math.radians(65)),
    Sensor((-670/4, -810/2), 40*100, math.radians(65)),
]

for i in [0, 2]:
    sensors[i].rotate(math.radians(180))

for i in [4, 6]:
    sensors[i].rotate(math.radians(90))

for i in [5, 7]:
    sensors[i].rotate(math.radians(-90))

draw_plan(drone_size, sensors)