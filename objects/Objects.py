import math
import shapely

class Sensor:
    def __init__(self, position: tuple[float, float], distance: float, angle: float):
        self.position = position
        self.distance = distance
        self.angle = angle
        self.observation = None
        self.polygon = None
        self.rotation = 0.0
        
    def get_observation(self):
        if self.observation is None:
            main_point = self.position

            corner_point1 = (self.distance * math.cos(self.rotation + self.angle/2) + main_point[0], 
                            self.distance * math.sin(self.rotation + self.angle/2) + main_point[1])
            
            corner_point2 = (self.distance * math.cos(self.rotation - self.angle/2) + main_point[0], 
                            self.distance * math.sin(self.rotation - self.angle/2) + main_point[1])
            
            self.observation = [main_point, corner_point1, corner_point2]

        return self.observation
    
    def rotate(self, angle: float):
        self.rotation += angle
        self.observation = None
        self.polygon = None

    def set_rotation(self, rotation: float):
        self.rotation = rotation
        self.observation = None
        self.polygon = None

    def move(self, dx: float, dy: float, constrains: tuple[float, float] = (0, 0)):
        if constrains == (0, 0):
            self.position = (self.position[0] + dx, self.position[1] + dy)
            self.observation = None
            self.polygon = None
        else:
            sign_x = 1 if self.position[0] + dx > 0 else -1
            sign_y = 1 if self.position[1] + dy > 0 else -1

            new_x = sign_x * min(abs(self.position[0] + dx), constrains[0])
            new_y = sign_y * min(abs(self.position[1] + dy), constrains[1])

            self.position = (new_x, new_y)

            self.observation = None
            self.polygon = None

    def get_polygon(self):
        if self.polygon is None:
            if self.observation is None:
                self.get_observation()

            self.polygon = shapely.Polygon(self.observation)

        return self.polygon
    
class ViewZone:
    def __init__(self, path: list[tuple[float, float]]):
        self.path = path
        self.polygon = None

    def get_polygon(self):
        if self.polygon is None:
            self.polygon = shapely.Polygon(self.path)
            
        return self.polygon

    def move(self, dx: float, dy: float):
        self.path = [(x + dx, y + dy) for x, y in self.path]
        self.polygon = None

    def __str__(self):
        return str(self.path)