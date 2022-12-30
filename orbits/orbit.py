from fractions import Fraction
from math import pi, cos, sin, atan, sqrt, tan

import numpy as np
from pyunitx.angle import degrees
from pyunitx.constants import G
from pyunitx.length import meters
from pyunitx.mass import kilograms
from pyunitx.time import days, seconds
from frames import rotations


class EllipticalOrbit:
    def __init__(self, inclination: degrees, eccentricity, M: kilograms, raan: degrees, arg_periapsis: degrees,
                 period: days, epoch_position: degrees):
        self.inclination = inclination.to_radians()
        self.eccentricity = eccentricity
        self.orbited_mass = M.to_kilograms()
        self.raan = raan.to_radians()
        self.arg_periapsis = arg_periapsis.to_radians()
        self.period = period.to_seconds()
        self.epoch_position = epoch_position.to_radians()
        self._periapsis = None
        self._apoapsis = None
        self._semimajor = None
        self._semiminor = None
        self._mu = None
        self._angular_momentum_mag = None
        self._p = None

    @property
    def periapsis(self):
        if self._periapsis is None:
            self._periapsis = self.semimajor * (1 - self.eccentricity)
        return self._periapsis

    @property
    def apoapsis(self):
        if self._apoapsis is None:
            self._apoapsis = self.semimajor * (1 + self.eccentricity)
        return self._apoapsis

    @property
    def semimajor(self):
        if self._semimajor is None:
            norm = self.period / (2 * pi)
            self._semimajor = (self.mu * norm ** 2) ** Fraction(1, 3)
        return self._semimajor

    @property
    def semiminor(self):
        if self._semiminor is None:
            self._semiminor = self.semimajor * sqrt(1 - self.eccentricity ** 2)
        return self._semiminor

    @property
    def mu(self):
        if self._mu is None:
            self._mu = G * self.orbited_mass
        return self._mu

    @property
    def angular_momentum_mag(self):
        if self._angular_momentum_mag is None:
            self._angular_momentum_mag = (self.mu * self.semimajor * (1 - self.eccentricity ** 2)) ** Fraction(1, 2)
        return self._angular_momentum_mag

    @property
    def p(self):
        if self._p is None:
            self._p = self.periapsis * (1 + self.eccentricity)
        return self._p

    def evaluate(self, angle):
        # Returns (_r,_v) in body-pointed frame [r,theta]
        # Which is a rotation by angle around 3 from the central object's frame
        if isinstance(angle, degrees):
            thetastar = angle.to_radians()
        else:
            thetastar = angle
        p = float(self.p.to_meters().value)
        mu = float(self.mu.to_meters_cubed_per_second_squared().value)
        h = float(self.angular_momentum_mag.to_meters_squared_per_second().value)
        distance = p / (1 + self.eccentricity * cos(thetastar))
        r = np.array([distance, 0, 0])
        vr = (mu / h) * self.eccentricity * sin(thetastar)
        vtheta = (mu / h) * (1 + self.eccentricity * cos(thetastar))
        v = np.array([vr, vtheta, 0])
        Rbp = rotations([(3, thetastar)]).transpose()
        return Rbp @ r.reshape((3, 1)), Rbp @ v.reshape((3, 1))

    def find_angle(self, time_since_periapsis: days):
        # See Orbital Mechanics for Engineers eq. 3.17
        n = 2 * pi / self.period.to_seconds()
        if self.eccentricity == 0:
            return n * time_since_periapsis.to_seconds()
        Me = float(n * time_since_periapsis.to_seconds())
        ecc_anomaly = newtons_method(lambda E: E - self.eccentricity * sin(E) - Me,
                                     lambda E: 1 - self.eccentricity * cos(E),
                                     pi / 2)
        ecc = sqrt((1 + self.eccentricity) / (1 - self.eccentricity))
        arg = ecc * tan(ecc_anomaly / 2)
        return 2 * atan(arg)

    def find_time(self, theta: float):
        n = 2 * pi / self.period.to_seconds()
        if self.eccentricity == 0:
            return theta / n
        ecc = sqrt((1 - self.eccentricity) / (1 + self.eccentricity))
        E = 2 * atan(ecc * tan(theta / 2))
        dt = (E - self.eccentricity * sin(E)) / n
        return dt


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
        lat = lat.to_radians()
        lon = lon.to_radians()
        direction = np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]).reshape((3, 1))
        # Currently can't go the other direction since unit multiplication can't handle its end
        return strip_units(meters)(direction * self.radius)


def newtons_method(f, fprime, x0):
    eps = 1e-6
    x = x0
    for i in range(100):
        delta = f(x) / fprime(x)
        x = x - delta
        if abs(delta) < eps:
            return x
    raise ValueError("Could not converge in 100 iterations, probably a problem")


def strip_units(unit):
    def stripper(v):
        converter = getattr(v, "to_" + unit.__name__)
        return float(converter().value)

    return np.vectorize(stripper)


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
