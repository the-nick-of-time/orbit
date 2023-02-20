from fractions import Fraction
from functools import cached_property
from math import pi, cos, sin, atan, sqrt, tan

import numpy as np
from pyunitx.angle import degrees
from pyunitx.constants import G
from pyunitx.length import meters
from pyunitx.mass import kilograms
from pyunitx.time import days, seconds

from frames import rotations
from solvers import newtons_method


class EllipticalOrbit:
    def __init__(self, inclination: degrees, eccentricity, mass: kilograms, raan: degrees, arg_periapsis: degrees,
                 period: days, epoch_position: degrees):
        """Define an elliptical orbit.

        :param inclination: The angle between the north pole of the central body
            and the normal to the orbital plane. In [0, 90] degrees.
        :param eccentricity: Ellipse eccentricity. Dimensionless, in [0, 1).
            0 means circular orbit and 1 is escaped parabolic.
        :param mass: The mass of the central body.
        :param raan: The right ascension of the ascending node. Undefined when
            inclination is zero, since there are no nodes. Physically, this is
            the central-body longitude at which the orbiter crosses the
            equatorial plane going north. Unbounded, though obviously it's mod
            360 degrees.
        :param arg_periapsis: The argument of periapsis. Undefined when raan is.
            Physically, this is the angle between the ascending node and
            periapsis. Unbounded, though obviously it's mod 360 degrees.
        :param period: The orbital period of the body.
        :param epoch_position: The true anomaly of the body along its orbit at
            the time of the epoch.
        """
        self.inclination = inclination.to_radians()
        self.eccentricity = eccentricity
        self.orbited_mass = mass.to_kilograms()
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
        """Calculate the position and velocity at any point in the orbit.

        :param angle: The true anomaly along the orbit.
        :return: A tuple with two vectors, both expressed in the orbit's perifocal frame:
            1. The position, in meters.
            2. The velocity, in meters per second.
        """
        if isinstance(angle, degrees):
            thetastar = angle.to_radians()
        else:
            thetastar = angle
        p = float(self.p.to_meters().value)
        mu = float(self.mu.to_meters_cubed_per_second_squared().value)
        h = float(self.angular_momentum_mag.to_meters_squared_per_second().value)
        distance = p / (1 + self.eccentricity * cos(thetastar))
        # r, theta make a natural body-centric frame
        r = np.array([distance, 0, 0])
        vr = (mu / h) * self.eccentricity * sin(thetastar)
        vtheta = (mu / h) * (1 + self.eccentricity * cos(thetastar))
        v = np.array([vr, vtheta, 0])
        # To transform from body to perifocal frame, make the rotation matrix
        # from perifocal to body then invert it
        Rbc = rotations([(3, thetastar)]).transpose()
        return Rbc @ r.reshape((3, 1)), Rbc @ v.reshape((3, 1))

    def find_angle(self, time_since_periapsis: days) -> float:
        """Find the true anomaly at any time along the orbit.

        :param time_since_periapsis: How long has elapsed since periapsis. Can
            be negative so it's time before periapsis.
        :return: The true anomaly, in radians.
        """
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

    def find_time(self, theta: float) -> seconds:
        """Find the time that has elapsed since periapsis given a true anomaly.

        :param theta: The true anomaly, in radians.
        :return: The time since periapsis, in pyunitx seconds.
        """
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
        """Define the important parameters of a body in an orbit.

        Among mass, radius, density, and circumference, any two can be defined.

        :param precession: At the epoch, the angle between the tilt direction
            and its natural frame.
        :param tilt: The angle between the body's north and the north of the
            body it's orbiting. The direction of the tilt is defined by
            precession.
        :param epoch_spin: At the epoch, how much the body has spun about its
            own axis.
        :param spin_rate: How fast the body is spinning, in radians per second.
        :param mass: The mass of the body, in pyunitx kilograms.
        :param radius: The radius of the body (assumed to be a perfect sphere),
            in pyunitx meters.
        :param density: The density of the body, in pyunitx kilograms per cubic
            meter.
        :param circumference: The circumference of the body, in pyunitx meters.
        """
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
            return self._radius.to_meters()
        if self.circumference:
            return self.circumference / (2 * pi)
        if self.mass and self.density:
            volume = self.mass / self.density
            return (volume * (3 / 4) / pi) ** Fraction(1, 3)

    @cached_property
    def circumference(self):
        if self._circumference:
            return self._circumference.to_meters()
        if self.radius:
            return 2 * pi * self.radius

    @cached_property
    def mass(self):
        if self._mass:
            return self._mass.to_kilograms()
        if self.density:
            return self.volume * self.density

    @cached_property
    def density(self):
        if self._density:
            return self._density.to_kilograms_per_meter_cubed()
        if self.mass:
            return self.mass / self.volume

    @cached_property
    def volume(self):
        return (4 / 3) * pi * self.radius ** 3

    @cached_property
    def gravity(self):
        """The gravitational acceleration at the surface of the body."""
        return G * self.mass / self.radius ** 2

    def roche_limit(self, satellite: 'Body') -> meters:
        """The minimum orbital radius at which a moon breaks up into rings.

        :param satellite: The orbiter.
        :return: The Roche limit, in pyunitx meters.
        """
        return self.radius * float(2 * self.density / satellite.density) ** Fraction(1, 3)

    def surface(self, lat: degrees, lon: degrees) -> np.ndarray:
        """The position of an object on the surface of the body.

        :param lat: The latitude of the object.
        :param lon: The longitude of the object.
        :return: A 3x1 numpy vector of floats representing meters in the body's
            MEN frame.
        """
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
