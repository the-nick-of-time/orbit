from fractions import Fraction
from functools import cached_property
from math import pi, cos, sin, atan, sqrt, tan

import numpy as np
from pyunitx.angle import degrees
from pyunitx.constants import G
from pyunitx.length import meters
from pyunitx.mass import kilograms
from pyunitx.time import days

from frames import rotations
from solvers import newtons_method


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

    @cached_property
    def periapsis(self):
        return self.semimajor * (1 - self.eccentricity)

    @cached_property
    def apoapsis(self):
        return self.semimajor * (1 + self.eccentricity)

    @cached_property
    def semimajor(self):
        norm = self.period / (2 * pi)
        return (self.mu * norm ** 2) ** Fraction(1, 3)

    @cached_property
    def semiminor(self):
        return self.semimajor * sqrt(1 - self.eccentricity ** 2)

    @cached_property
    def mu(self):
        return G * self.orbited_mass

    @cached_property
    def angular_momentum_mag(self):
        return (self.mu * self.semimajor * (1 - self.eccentricity ** 2)) ** Fraction(1, 2)

    @cached_property
    def p(self):
        return self.periapsis * (1 + self.eccentricity)

    def evaluate(self, angle):
        # Returns (_r,_v) in central object frame
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
        Rbc = rotations([(3, thetastar)]).transpose()
        return Rbc @ r.reshape((3, 1)), Rbc @ v.reshape((3, 1))

    def find_angle(self, time_since_periapsis: days):
        # See Orbital Mechanics for Engineers eq. 3.17
        n = 2 * pi / self.period.to_seconds()
        if self.eccentricity == 0:
            return n * time_since_periapsis.to_seconds()
        Me = float(n * time_since_periapsis.to_seconds())
        ecc_anomaly = newtons_method(
            lambda E: E - self.eccentricity * sin(E) - Me,
            lambda E: 1 - self.eccentricity * cos(E),
            pi / 2
        )
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
    def __init__(self, precession: degrees, tilt: degrees, epoch_spin: degrees, spin_rate, mass=None, radius=None,
                 density=None, circumference=None):
        self.precession = precession.to_radians()
        self.tilt = tilt.to_radians()
        self.epoch_spin = epoch_spin.to_radians()
        self.spin_rate = spin_rate.to_hertz()  # really rad/s
        self._radius = radius
        self._circumference = circumference
        self._mass = mass
        self._density = density

    @cached_property
    def radius(self):
        if self._radius:
            return self._radius
        if self.circumference:
            return self.circumference / (2 * pi)
        if self.mass and self.density:
            volume = self.mass / self.density
            return (volume * (3 / 4) / pi) ** Fraction(1, 3)

    @cached_property
    def circumference(self):
        if self._circumference:
            return self._circumference
        if self.radius:
            return 2 * pi * self.radius

    @cached_property
    def mass(self):
        if self._mass:
            return self._mass
        if self.density:
            return self.volume * self.density

    @cached_property
    def density(self):
        if self._density:
            return self._density
        if self.mass:
            return self.mass / self.volume

    @cached_property
    def volume(self):
        return (4 / 3) * pi * self.radius ** 3

    @cached_property
    def gravity(self):
        return G * self.mass / self.radius ** 2

    def roche_limit(self, satellite: 'Body'):
        return self.radius * float(2 * self.density / satellite.density) ** Fraction(1, 3)

    def surface(self, lat: degrees, lon: degrees):
        lat = lat.to_radians()
        lon = lon.to_radians()
        direction = np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]).reshape((3, 1))
        # Currently can't go the other direction since unit multiplication can't handle its end
        return strip_units(meters)(direction * self.radius)


def strip_units(unit):
    def stripper(v):
        converter = getattr(v, "to_" + unit.__name__)
        return float(converter().value)

    return np.vectorize(stripper)
