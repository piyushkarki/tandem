import math
from typing import Tuple

# Constants used in the CircularHole simulation
S = 1.0
a = 0.2
mu0 = 1.0
nu = 0.25


class CircularHole:
    """Class representing a circular hole in a material."""

    def __init__(self) -> None:
        """Initialize a CircularHole instance."""
        pass

    def mu(self, x: float, y: float) -> float:
        """Return the shear modulus at the given (x, y) coordinates."""
        return mu0

    def lam(self, x: float, y: float) -> float:
        """Return the lambda parameter at the given (x, y) coordinates."""
        return 2 * mu0 * nu / (1 - 2 * nu)

    def force(self, x: float, y: float) -> Tuple[float, float]:
        """Return the force applied at the given (x, y) coordinates."""
        return 0.0, 0.0

    @staticmethod
    def polar(x: float, y: float) -> Tuple[float, float]:
        """Convert Cartesian coordinates (x, y) to polar coordinates (r, θ).

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            Tuple[float, float]: A tuple containing the radius and angle in polar coordinates.
        """
        r = math.sqrt(x**2 + y**2)
        t = math.atan2(y, x)
        return r, t

    @staticmethod
    def u_polar(r: float, t: float) -> Tuple[float, float]:
        """Calculate the polar displacement components (ur, ut).

        Args:
            r (float): The radius in polar coordinates.
            t (float): The angle in polar coordinates.

        Returns:
            Tuple[float, float]: A tuple containing the radial and tangential displacement components.
        """
        ur = (
            (1.0 / 2.0)
            * S
            * (-(a**4) - 4 * a**2 * r**2 * (nu - 1) + r**4)
            * math.sin(2 * t)
            / (mu0 * r**3)
        )
        ut = (
            (1.0 / 2.0)
            * S
            * (a**4 - 4 * a**2 * nu * r**2 + 2 * a**2 * r**2 + r**4)
            * math.cos(2 * t)
            / (mu0 * r**3)
        )
        return ur, ut

    def solution(self, x: float, y: float) -> Tuple[float, float]:
        """Calculate the displacement solution at the given (x, y) coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            Tuple[float, float]: A tuple containing the displacement components (ux, uy).
        """
        r, t = self.polar(x, y)
        ur, ut = self.u_polar(r, t)
        ux = ur * math.cos(t) - ut * math.sin(t)
        uy = ur * math.sin(t) + ut * math.cos(t)
        return ux, uy

    def solution_jacobian(
        self, x: float, y: float
    ) -> Tuple[float, float, float, float]:
        """Calculate the Jacobian of the displacement solution at the given (x, y) coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the components of the Jacobian matrix.
        """
        r, t = self.polar(x, y)
        c = math.cos(t)
        s = math.sin(t)

        ur, ut = self.u_polar(r, t)

        # Compute partial derivatives of ur and ut
        ur_r = (
            (1.0 / 2.0)
            * S
            * (3 * a**4 + 4 * a**2 * nu * r**2 - 4 * a**2 * r**2 + r**4)
            * math.sin(2 * t)
            / (mu0 * r**4)
        )
        ur_t = (
            -S
            * (a**4 + 4 * a**2 * r**2 * (nu - 1) - r**4)
            * math.cos(2 * t)
            / (mu0 * r**3)
        )
        ut_r = (
            (1.0 / 2.0)
            * S
            * (-3 * a**4 + 4 * a**2 * nu * r**2 - 2 * a**2 * r**2 + r**4)
            * math.cos(2 * t)
            / (mu0 * r**4)
        )
        ut_t = (
            S
            * (-(a**4) + 4 * a**2 * nu * r**2 - 2 * a**2 * r**2 - r**4)
            * math.sin(2 * t)
            / (mu0 * r**3)
        )

        # Derivatives of polar coordinates with respect to Cartesian coordinates
        r_x = c
        r_y = s
        t_x = -s / r if r != 0 else 0
        t_y = c / r if r != 0 else 0

        # Chain rule to calculate derivatives of ux and uy
        ur_x = ur_r * r_x + ur_t * t_x
        ur_y = ur_r * r_y + ur_t * t_y
        ut_x = ut_r * r_x + ut_t * t_x
        ut_y = ut_r * r_y + ut_t * t_y

        # Calculate the components of the Jacobian
        ux_x = c * ur_x - s * ut_x - t_x * (s * ur + c * ut)
        ux_y = c * ur_y - s * ut_y - t_y * (s * ur + c * ut)
        uy_x = s * ur_x + c * ut_x + t_x * (c * ur - s * ut)
        uy_y = s * ur_y + c * ut_y + t_y * (c * ur - s * ut)

        return ux_x, ux_y, uy_x, uy_y

    def boundary(self, x: float, y: float) -> Tuple[float, float]:
        """Return the displacement at the boundary for given (x, y) coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            Tuple[float, float]: A tuple containing the displacement components (ux, uy).
        """
        return self.solution(x, y)


# Create an instance of the CircularHole class
circular_hole = CircularHole()
