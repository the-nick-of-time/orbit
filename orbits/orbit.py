from fractions import Fraction
from math import pi, cos, sin, atan, sqrt, tan

import numpy as np
from cproperty import cproperty
from pyunitx.angle import degrees
from pyunitx.constants import G
from pyunitx.length import meters
from pyunitx.mass import kilograms
from pyunitx.time import days, seconds


class EllipticalOrbit:
    def __init__(self, inclination: degrees, eccentricity, M: kilograms, raan: degrees, arg_periapsis: degrees,
                 period: days, epoch_position: degrees):
        self.inclination = inclination.to_radians()
        self.eccentricity = eccentricity
        self.orbited_mass = M
        self.raan = raan.to_radians()
        self.arg_periapsis = arg_periapsis.to_radians()
        self.period = period.to_seconds()
        self.epoch_position = epoch_position.to_radians()

    @cproperty
    def periapsis(self):
        return self.semimajor * (1 - self.eccentricity)

    @cproperty
    def apoapsis(self):
        return self.semimajor * (1 + self.eccentricity)

    @cproperty
    def semimajor(self):
        norm = self.period / (2 * pi)
        return (self.mu * norm ** 2) ** Fraction(1, 3)

    @cproperty
    def semiminor(self):
        return self.semimajor * sqrt(1 - self.eccentricity ** 2)

    @cproperty
    def mu(self):
        return G * self.orbited_mass

    @cproperty
    def angular_momentum_mag(self):
        return (self.mu * self.semimajor * (1 - self.eccentricity ** 2)) ** Fraction(1, 2)

    @cproperty
    def p(self):
        return self.periapsis * (1 + self.eccentricity)

    def evaluate(self, angle):
        # Returns (_r,_v) in body-pointed frame [r,theta]
        # Which is a rotation by angle around 3 from the central object's frame
        if isinstance(angle, degrees):
            thetastar = angle.to_radians()
        else:
            thetastar = angle
        distance = self.p / (1 + self.eccentricity * cos(thetastar))
        r = np.array([distance, meters(0), meters(0)])
        vr = (self.mu / self.angular_momentum_mag) * self.eccentricity * sin(thetastar)
        vtheta = (self.mu / self.angular_momentum_mag) * (1 + self.eccentricity * cos(thetastar))
        v = np.array([vr, vtheta, meters(0) / seconds(1)])
        return r.reshape((3, 1)), v.reshape((3, 1))

    def eval_unitless(self, angle):
        r, v = self.evaluate(angle)
        r = np.vectorize(lambda k: float(k.to_meters().value))(r)
        v = np.vectorize(lambda k: float(k.to_meters_per_second().value))(v)
        return r, v

    def find_angle(self, time_since_periapsis: days):
        # See Orbital Mechanics for Engineers eq. 3.17
        n = 2 * pi / self.period.to_seconds()
        Me = float(n * time_since_periapsis.to_seconds())
        ecc_anomaly = newtons_method(lambda E: E - self.eccentricity * sin(E) - Me,
                                     lambda E: 1 - self.eccentricity * cos(E),
                                     pi / 2)
        ecc = sqrt((1 + self.eccentricity) / (1 - self.eccentricity))
        arg = ecc * tan(ecc_anomaly / 2)
        return 2 * atan(arg)


class Body:
    def __init__(self, precession: degrees, tilt: degrees, epoch_spin: degrees, mass=None, radius=None, density=None,
                 circumference=None):
        self.precession = precession.to_radians()
        self.tilt = tilt.to_radians()
        self.epoch_spin = epoch_spin.to_radians()
        if radius:
            self.radius = radius
            self.circumference = 2 * pi * radius
        elif circumference:
            self.circumference = circumference
            self.radius = circumference / (2 * pi)
        else:
            raise ValueError("Must have exactly one of radius/circumference")
        if mass:
            self.mass = mass
            self.density = mass / self.volume
        elif density:
            self.density = density
            self.mass = self.volume * density
        else:
            raise ValueError("Must have exactly one of mass/density")

    @property
    def volume(self):
        return (4 / 3) * pi * self.radius ** 3

    def roche_limit(self, satellite: 'Body'):
        return self.radius * float(2 * self.density / satellite.density) ** Fraction(1, 3)

    def surface(self, lat: degrees, lon: degrees):
        direction = np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]).reshape((3, 1))
        # Currently can't go the other direction since unit multiplication can't handle its end
        return direction * self.radius


def newtons_method(f, fprime, x0):
    eps = 1e-6
    x = x0
    for i in range(100):
        delta = f(x) / fprime(x)
        x = x - delta
        if abs(delta) < eps:
            return x
    raise ValueError("Could not converge in 100 iterations, probably a problem")


if __name__ == '__main__':
    moon1 = EllipticalOrbit(
        inclination=degrees(72),
        period=days(13.37),
        raan=degrees(58),
        arg_periapsis=degrees(121),
        eccentricity=0.133,
        M=kilograms("3.33469e24")
    )
    print(moon1.semimajor, moon1.semiminor, moon1.periapsis, moon1.apoapsis)
    th = moon1.find_angle(days(0))
    print(th, moon1.evaluate(th))
    th = moon1.find_angle(days(4))
    print(th, moon1.evaluate(th))
    th = moon1.find_angle(days(13.37 / 2))
    print(th, moon1.evaluate(th))
