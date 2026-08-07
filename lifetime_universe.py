import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy.optimize import minimize_scalar, brentq


# Constants
C = 299792.458          # Speed of light, km/s
H0 = 70.0               # Hubble constant, km/s/Mpc
MPC_IN_KM = 3.0856775814913673e19
SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60


# Load Union2.1 supernova data
# Columns: redshift, distance modulus, distance modulus error
data = np.loadtxt(
    "SCPUnion2.1_mu_vs_z.txt",
    skiprows=5,
    usecols=(1, 2, 3)
)

z = data[:, 0]
mu_obs = data[:, 1]
mu_err = data[:, 2]

print(f"Number of supernovae: {len(z)}")


def mu_to_DL(mu):
    """Convert distance modulus to luminosity distance in Mpc."""
    return 10.0 ** ((mu - 25.0) / 5.0)


DL_obs = mu_to_DL(mu_obs)


def E(z, OmegaM):
    """Dimensionless Hubble parameter for a flat Lambda-CDM universe."""
    OmegaL = 1.0 - OmegaM
    return np.sqrt(
        OmegaM * (1.0 + z) ** 3 + OmegaL
    )


def luminosity_distance_scalar(z, OmegaM):
    """Calculate luminosity distance for one redshift value."""
    integral, _ = si.quad(
        lambda zp: 1.0 / E(zp, OmegaM),
        0.0,
        float(z),
        epsabs=1e-9,
        epsrel=1e-9
    )

    hubble_distance = C / H0
    comoving_distance = hubble_distance * integral

    return (1.0 + z) * comoving_distance


def luminosity_distance(z, OmegaM):
    """Calculate luminosity distance for a scalar or an array."""
    z_array = np.atleast_1d(z).astype(float)

    distances = np.array([
        luminosity_distance_scalar(z_i, OmegaM)
        for z_i in z_array
    ])

    if np.ndim(z) == 0:
        return distances[0]

    return distances


def theoretical_mu(z, OmegaM):
    """Calculate the theoretical distance modulus."""
    DL = luminosity_distance(z, OmegaM)
    return 5.0 * np.log10(DL) + 25.0


def chi_square(OmegaM):
    """Calculate chi-square for a given matter density."""
    mu_model = theoretical_mu(z, OmegaM)

    return np.sum(
        ((mu_obs - mu_model) / mu_err) ** 2
    )


# Find the best-fit value of Omega_M
result = minimize_scalar(
    chi_square,
    bounds=(0.01, 0.99),
    method="bounded",
    options={"xatol": 1e-6}
)

OmegaM_best = result.x
OmegaL_best = 1.0 - OmegaM_best
chi2_min = result.fun

degrees_of_freedom = len(z) - 1
reduced_chi2 = chi2_min / degrees_of_freedom


print()
print("Best-fit results")
print("----------------------------")
print(f"Omega_M       = {OmegaM_best:.6f}")
print(f"Omega_Lambda  = {OmegaL_best:.6f}")
print(f"Minimum chi^2 = {chi2_min:.3f}")
print(f"Degrees of freedom = {degrees_of_freedom}")
print(f"Reduced chi^2 = {reduced_chi2:.3f}")


# Estimate the 1-sigma uncertainty using Delta chi^2 = 1
target_chi2 = chi2_min + 1.0


def delta_chi_square(OmegaM):
    return chi_square(OmegaM) - target_chi2


try:
    OmegaM_lower = brentq(
        delta_chi_square,
        0.01,
        OmegaM_best
    )

    OmegaM_upper = brentq(
        delta_chi_square,
        OmegaM_best,
        0.99
    )

    error_minus = OmegaM_best - OmegaM_lower
    error_plus = OmegaM_upper - OmegaM_best

    print()
    print(
        f"Omega_M = {OmegaM_best:.4f} "
        f"-{error_minus:.4f} "
        f"+{error_plus:.4f}"
    )

except ValueError:
    OmegaM_lower = None
    OmegaM_upper = None
    print("\nCould not determine the 1-sigma interval.")


# Redshift range used for the model curves
z_plot = np.linspace(
    max(1e-4, np.min(z) * 0.5),
    np.max(z) * 1.03,
    300
)


# Hubble diagram
plt.figure(figsize=(10, 7))

plt.errorbar(
    z,
    mu_obs,
    yerr=mu_err,
    fmt=".",
    markersize=3,
    alpha=0.45,
    elinewidth=0.5,
    capsize=0,
    label="Union2.1 data"
)


# Compare several cosmological models
OmegaM_models = [0.01, 0.27, 0.99]

for OmegaM in OmegaM_models:
    mu_model = theoretical_mu(z_plot, OmegaM)

    plt.plot(
        z_plot,
        mu_model,
        linewidth=1.8,
        label=(
            rf"$\Omega_M={OmegaM:.2f},\ "
            rf"\Omega_\Lambda={1.0-OmegaM:.2f}$"
        )
    )


# Best-fit model
mu_best = theoretical_mu(z_plot, OmegaM_best)

plt.plot(
    z_plot,
    mu_best,
    "--",
    linewidth=2.5,
    label=rf"Best fit: $\Omega_M={OmegaM_best:.3f}$"
)

plt.xlabel(r"Redshift $z$", fontsize=15)
plt.ylabel(r"Distance modulus $\mu$", fontsize=15)
plt.title("Hubble Diagram for Union2.1 Supernovae", fontsize=15)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()


# Chi-square as a function of Omega_M
OmegaM_grid = np.linspace(0.05, 0.60, 100)

chi2_grid = np.array([
    chi_square(OmegaM)
    for OmegaM in OmegaM_grid
])

plt.figure(figsize=(9, 6))

plt.plot(
    OmegaM_grid,
    chi2_grid,
    linewidth=2
)

plt.axvline(
    OmegaM_best,
    linestyle="--",
    label=rf"Best fit: $\Omega_M={OmegaM_best:.3f}$"
)

plt.axhline(
    chi2_min + 1.0,
    linestyle=":",
    label=r"$\chi^2_{\mathrm{min}}+1$"
)

plt.xlabel(r"$\Omega_M$", fontsize=15)
plt.ylabel(r"$\chi^2$", fontsize=15)
plt.title(r"$\chi^2$ as a Function of $\Omega_M$", fontsize=15)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()


def age_of_universe(OmegaM):
    """Calculate the age of a flat Lambda-CDM universe in Gyr."""
    OmegaL = 1.0 - OmegaM

    # Convert H0 from km/s/Mpc to s^-1
    H0_si = H0 / MPC_IN_KM

    def integrand(z):
        Ez = np.sqrt(
            OmegaM * (1.0 + z) ** 3 + OmegaL
        )

        return 1.0 / ((1.0 + z) * Ez)

    integral, _ = si.quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10
    )

    age_seconds = integral / H0_si
    age_years = age_seconds / SECONDS_IN_YEAR

    return age_years / 1e9


# Ages for the three example models
print()
print("Age of the Universe")
print("----------------------------")

for OmegaM in OmegaM_models:
    OmegaL = 1.0 - OmegaM
    age = age_of_universe(OmegaM)

    print(
        f"Omega_M = {OmegaM:.2f}, "
        f"Omega_Lambda = {OmegaL:.2f}: "
        f"{age:.3f} Gyr"
    )


# Age for the best-fit model
best_age = age_of_universe(OmegaM_best)

print()
print("Best-fit model")
print("----------------------------")
print(f"Omega_M      = {OmegaM_best:.4f}")
print(f"Omega_Lambda = {OmegaL_best:.4f}")
print(f"Age          = {best_age:.3f} Gyr")


# Luminosity distance versus redshift
DL_best = luminosity_distance(z_plot, OmegaM_best)

plt.figure(figsize=(10, 7))

plt.plot(
    DL_obs,
    z,
    ".",
    markersize=3,
    alpha=0.5,
    label="Union2.1 data"
)

plt.plot(
    DL_best,
    z_plot,
    linewidth=2.5,
    label=rf"Best fit: $\Omega_M={OmegaM_best:.3f}$"
)

plt.xlabel(
    r"Luminosity distance $D_L$ [Mpc]",
    fontsize=15
)

plt.ylabel(
    r"Redshift $z$",
    fontsize=15
)

plt.title(
    "Luminosity Distance vs. Redshift",
    fontsize=15
)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
