from math import sqrt, acos, pi


class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = coordinates
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError("The coordinates must not be empty!")

        except TypeError:
            raise TypeError("The coordinates must be an iterable!")


    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates


    def myplus(self, v):
        new_coordinates = [x+y for x, y in zip(self.coordinates, v.coordinates) ]
        return new_coordinates

    def myminus(self, v):
        new_coordinates = [x-y for x, y in zip(self.coordinates, v.coordinates)]
        return new_coordinates

    def mytimes(self, c):
        new_coordinates = [x*c for x in self.coordinates]
        return new_coordinates


    # 大小
    def magnitude(self):
        magnitude =0
        for i in  range(len(self.coordinates)):
            magnitude += self.coordinates[i] * self.coordinates[i]
        return sqrt(magnitude)

    def magnitude2(self):
        coordinate_squared  = [x**2 for x in self.coordinates]
        return sqrt(sum(coordinate_squared))

    def norm(self):
        magnitude1 = self.magnitude()
        new_coordinates = [x/magnitude1 for x in self.coordinates]
        return new_coordinates

    def myinner(self, v):
        new_coordinates = [x*y for x, y in zip(self.coordinates, v.coordinates)]
        return sum(new_coordinates)

    def angle_with(self, v, in_degrees=False):
        try:
            angle_in_radians = acos(self.myinner(v) / (self.magnitude2() * v.magnitude2()))

            if in_degrees:
                degrees_per_radian = 180 / pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR:
                raise Exception("Cannot compute an angle with a zero vector")
            else:
                raise e

    def checkParallel(self, v):
        return (self.is_zero() or
                v.is_zero() or
                self.angle_with(v) == 0 or
                self.angle_with(v) == pi)

    def is_zero(self, tolerance=1e-10):
        return self.magnitude2() < tolerance





my_vector =Vector([1,2,3])
print(my_vector)

my_vector1 = Vector([8.218,-9.341])
my_vector2 = Vector([-1.129,2.111])

print(my_vector1.myplus(my_vector2))
print(my_vector1.myminus(my_vector2))
print(my_vector1.mytimes(3))

print(my_vector1.magnitude())
print(my_vector1.magnitude2())
print(my_vector1.norm())
print(my_vector1.myinner(my_vector2))


angel3_arccos = acos(my_vector1.myinner(my_vector2) / (my_vector1.magnitude2() * my_vector2.magnitude2()))
print(angel3_arccos)

v3_vector = Vector([3.183,-7.627])
w3_vector = Vector([-2.668,5.319])

print(v3_vector.myinner(w3_vector) / (v3_vector.magnitude2() * w3_vector.magnitude2()))

angel3_arccos = acos(v3_vector.myinner(w3_vector) / (v3_vector.magnitude2() * w3_vector.magnitude2()))
print(angel3_arccos)
print(acos(1))

print(v3_vector.angle_with(w3_vector, True))

print(v3_vector.checkParallel( w3_vector))