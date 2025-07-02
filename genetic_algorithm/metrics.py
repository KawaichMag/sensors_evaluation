import math

from objects.Objects import Sensor, ViewZone
from objects.Types import Genotype

def get_coverage(sensors: list[Sensor], view_zones: list[ViewZone]) -> float:
        coverage_area = 0

        for sensor in sensors:
            for view_zone in view_zones:
                if sensor.get_polygon().intersects(view_zone.get_polygon()):
                    coverage_area += sensor.get_polygon().intersection(view_zone.get_polygon()).area

        return coverage_area

def get_overlap(sensors: Genotype) -> float:
    # Get the overlap of the sensors
    intersection_areas = []

    for i, sensor in enumerate(sensors):
        for other_sensor in sensors[i+1:]:
            intersection_areas.append(
                -sensor.get_polygon().intersection(other_sensor.get_polygon()).area / sensor.get_polygon().union(other_sensor.get_polygon()).area)

    return sum(intersection_areas) / len(intersection_areas)

def get_angle_density(sensors: Genotype) -> float:
    # Get the angle density of the sensors
    angels = []
    positions = []

    middle_point = sensors[0].position

    for sensor in sensors:
        middle_point= (sensor.position[0] + middle_point[0]) / 2, (sensor.position[1] + middle_point[1]) / 2

    for i, sensor in enumerate(sensors):
        angel = sensor.rotation % (2 * math.pi)
        if angel < 0:
            angel += math.pi * 2
        mean_angel = (middle_point[0] - sensor.position[0], middle_point[1] - sensor.position[1])
        angel_vector = math.atan2(mean_angel[1], mean_angel[0])
        if angel_vector < 0:
            angel_vector = math.pi * 2 + angel_vector
        coef = -abs(abs(angel_vector - angel) - math.pi) / math.pi
        angels.append(coef)
                
    return sum(angels) / len(angels)

def fitness_function(sensors: Genotype, view_zones: list[ViewZone]) -> tuple[float, float]:
    return get_angle_density(sensors), get_overlap(sensors)