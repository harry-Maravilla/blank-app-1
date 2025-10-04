#!/usr/bin/env python3
"""
Asteroid Impact Simulator (Python 3.13)

Features:
- Create asteroid objects (size, density, velocity, angle, strength)
- Compute mass, kinetic energy (Joules, megatons TNT)
- Estimate atmospheric breakup (airburst) altitude using dynamic pressure
- Provide simple crater diameter and ground blast footprint estimates
- Estimate casualties assuming uniform population density around impact
- Simple mitigation modules: delta-v deflection, fragmentation
- Monte Carlo uncertainty sampling
- Visualizations: energy vs size, impact footprint, casualty histogram

Note: These are simplified/educational models. Replace constants or implement
more advanced physics (e.g., detailed scaling laws by Holsapple / Collins / Melosh)
if you need higher-fidelity results.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Callable
import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Physical constants
# ---------------------
G = 9.80665  # m/s^2, gravity on Earth
RHO_AIR_SEA_LEVEL = 1.225  # kg/m^3
SCALE_HEIGHT = 8500.0  # m, atmospheric scale height
JOULES_PER_MEGATON_TNT = 4.184e15  # J


# ---------------------
# Utility helpers
# ---------------------
def km2m(x_km: float) -> float:
    return x_km * 1000.0


def m2km(x_m: float) -> float:
    return x_m / 1000.0


def safe_log10(x: float) -> float:
    return math.log10(x) if x > 0 else float("-inf")


# ---------------------
# Asteroid dataclass
# ---------------------
@dataclass
class Asteroid:
    diameter_m: float  # meters
    density_kg_m3: float = 3000.0  # typical rocky = 3000 kg/m^3; icy ~1000
    velocity_m_s: float = 20000.0  # m/s (20 km/s typical)
    entry_angle_deg: float = 45.0  # measured from horizontal (so 90=vertical)
    material_strength_pa: float = 1e6  # Pa - tensile strength; 1e6 for rubble pile low, 1e8+ for strong rocks
    porosity: float = 0.2  # 0..1, affects fragmentation (not fully used here)
    name: str = "Asteroid"

    # derived fields
    mass_kg: float = field(init=False)
    cross_section_area_m2: float = field(init=False)

    def __post_init__(self):
        radius = self.diameter_m / 2.0
        volume = (4.0 / 3.0) * math.pi * radius ** 3
        self.mass_kg = self.density_kg_m3 * volume
        self.cross_section_area_m2 = math.pi * radius ** 2

    def kinetic_energy_joules(self) -> float:
        """Kinetic energy = 0.5 m v^2"""
        return 0.5 * self.mass_kg * (self.velocity_m_s ** 2)

    def kinetic_energy_megatons(self) -> float:
        return self.kinetic_energy_joules() / JOULES_PER_MEGATON_TNT

    def __repr__(self):
        return (
            f"<Asteroid {self.name}: D={self.diameter_m:.1f} m, mass={self.mass_kg:.2e} kg, "
            f"v={self.velocity_m_s:.1f} m/s ({self.kinetic_energy_megatons():.2f} MT)>"
        )


# ---------------------
# Atmospheric breakup / airburst model
# ---------------------
def atmospheric_density_at_altitude(h_m: float) -> float:
    """Exponential atmosphere model (approx)."""
    return RHO_AIR_SEA_LEVEL * math.exp(-h_m / SCALE_HEIGHT)


def breakup_altitude_estimate(asteroid: Asteroid) -> Tuple[Optional[float], float]:
    """
    Estimate altitude (m) where dynamic pressure exceeds material_strength_pa.
    Dynamic pressure q = 0.5 * rho(h) * v^2 (ignoring deceleration during entry).
    If dynamic pressure threshold is never reached above ground, return None as ground impact.
    Returns (altitude_m or None, dynamic_pressure_at_sea_level).
    NOTE: This neglects deceleration, ablation and angle-dependence — simplistic.
    """
    v = asteroid.velocity_m_s
    strength = asteroid.material_strength_pa
    q_sea = 0.5 * RHO_AIR_SEA_LEVEL * v ** 2

    # Solve for h where 0.5 * rho0 * exp(-h/H) * v^2 = strength -> exp(-h/H) = (2*strength)/(rho0*v^2)
    rhs = (2.0 * strength) / (RHO_AIR_SEA_LEVEL * v ** 2)
    if rhs <= 0:
        return None, q_sea

    if rhs >= 1:
        # dynamic pressure at sea level is insufficient to break it -> ground impact
        return None, q_sea

    h = -SCALE_HEIGHT * math.log(rhs)
    # ensure h non-negative and not ridiculously high
    h = max(0.0, min(h, 1.5e5))
    return h, q_sea


# ---------------------
# Simple crater & blast scaling (approx)
# ---------------------
def crater_diameter_estimate(energy_megatons: float, target_density_kg_m3: float = 2500.0) -> float:
    """
    Very simplified crater diameter estimator (transient crater). Units: energy in megatons => diameter in meters.
    Uses an approximate power-law scaling: D_t (km) ~ a * E_mt^b, with conservative coefficients.
    Default coefficients chosen to give plausible numbers for small/medium impacts.
    NOTE: Replace with Holsapple/Collins scaling for high fidelity.
    """
    # coefficients (tunable)
    a_km = 0.12  # coefficient in km
    b = 0.30
    D_km = a_km * (energy_megatons ** b)
    return D_km * 1000.0


def blast_radius_overpressure_psi(energy_megatons: float, psi_threshold: float = 5.0) -> float:
    """
    Estimate radius (meters) for a given overpressure threshold (psi).
    This is a highly simplified scaling. Use R ~ C * E^(1/3) with constants tuned.
    Returns radius in meters.
    """
    # base constants (tunable)
    # For 1 MT, a ~ 1.0 to 2.0 km for ~5 psi depending on burst height; we choose conservative scale:
    C_km = 1.3  # kilometers per (MT)^(1/3) for ~5 psi threshold (roughly)
    # scale psi: higher psi -> smaller radius; simple inverse law assumed
    psi_scale = (5.0 / psi_threshold) ** 0.4
    R_km = C_km * (energy_megatons ** (1.0 / 3.0)) * psi_scale
    return R_km * 1000.0


# ---------------------
# Casualty / damage model
# ---------------------
def estimate_casualties(energy_megatons: float, population_density_per_km2: float, psi_lethal: float = 3.0) -> Dict[str, float]:
    """
    Estimate casualties given energy and population density (people/km^2).
    - Compute radius for psi_lethal overpressure
    - Assume uniform population density; casualties = area * density * lethality_fraction
    - lethality_fraction is an adjustable function of psi (higher psi, higher lethality)
    Returns dict with radii, population_inside, estimated_killed, estimated_injured.
    """
    R_m = blast_radius_overpressure_psi(energy_megatons, psi_threshold=psi_lethal)
    area_km2 = math.pi * (m2km(R_m) ** 2)
    pop_inside = area_km2 * population_density_per_km2

    # heuristics for lethality and injury fractions based on psi (very rough)
    if psi_lethal >= 10:
        lethality_frac = 0.9
        inj_frac = 0.1
    elif psi_lethal >= 5:
        lethality_frac = 0.5
        inj_frac = 0.3
    elif psi_lethal >= 3:
        lethality_frac = 0.2
        inj_frac = 0.5
    else:
        lethality_frac = 0.05
        inj_frac = 0.2

    killed = pop_inside * lethality_frac
    injured = pop_inside * inj_frac
    return {
        "psi_threshold": psi_lethal,
        "blast_radius_m": R_m,
        "area_km2": area_km2,
        "pop_inside": pop_inside,
        "estimated_killed": killed,
        "estimated_injured": injured,
    }


# ---------------------
# Mitigation models
# ---------------------
def apply_delta_v_deflection(asteroid: Asteroid, delta_v_m_s: float, time_to_impact_s: float) -> Tuple[float, float]:
    """
    Apply a delta-v perpendicular to the velocity (simplified) and estimate miss distance.
    Very simplified: assume small lateral delta-v applied long before impact yields lateral displacement:
        lateral_disp = delta_v * time_to_impact
    We then translate into miss distance at Earth's distance scale.
    Returns (new_velocity_m_s, lateral_miss_distance_m)
    NOTE: Real deflection requires orbital mechanics; this is a conceptual proxy.
    """
    lateral_disp = delta_v_m_s * time_to_impact_s
    # For perspective, a lateral displacement larger than Earth's radius (~6.37e6 m) will miss Earth.
    return asteroid.velocity_m_s, lateral_disp


def fragmentation_model(asteroid: Asteroid, n_fragments: int) -> List[Asteroid]:
    """
    Split the asteroid into n_fragments (equal-mass) and return a list of fragments.
    This reduces energy per fragment but can increase ground area affected.
    """
    if n_fragments <= 1:
        return [asteroid]
    fragments = []
    mass_each = asteroid.mass_kg / n_fragments
    # compute diameter for each fragment assuming same density
    volume_each = mass_each / asteroid.density_kg_m3
    radius_each = ((3.0 * volume_each) / (4.0 * math.pi)) ** (1.0 / 3.0)
    diameter_each = 2.0 * radius_each
    for i in range(n_fragments):
        frag = Asteroid(
            diameter_m=diameter_each,
            density_kg_m3=asteroid.density_kg_m3,
            velocity_m_s=asteroid.velocity_m_s,
            entry_angle_deg=asteroid.entry_angle_deg,
            material_strength_pa=asteroid.material_strength_pa,
            porosity=asteroid.porosity,
            name=f"{asteroid.name}_frag{i+1}",
        )
        fragments.append(frag)
    return fragments


# ---------------------
# Monte Carlo simulation
# ---------------------
def monte_carlo_simulation(
    asteroid_nominal: Asteroid,
    population_density_per_km2: float,
    n_samples: int = 500,
    uncertainty_factors: Optional[Dict[str, Tuple[float, float]]] = None,
    mitigation: Optional[Dict] = None,
) -> Dict:
    """
    Run Monte Carlo sampling over uncertainties.
    uncertainty_factors: dict mapping parameter name to (low_factor, high_factor) multiplicative range
        e.g. {"diameter_m": (0.8, 1.2), "velocity_m_s": (0.9, 1.1)}
    mitigation: dict could include {"delta_v_m_s": value, "time_to_impact_s": t, "fragment_count": n}
    Returns summary statistics and samples list.
    """
    if uncertainty_factors is None:
        uncertainty_factors = {
            "diameter_m": (0.9, 1.1),
            "velocity_m_s": (0.95, 1.05),
            "density_kg_m3": (0.8, 1.2),
            "material_strength_pa": (0.5, 2.0),
        }

    samples = []
    killed_list = []
    energy_list = []
    airburst_alt_list = []

    for i in range(n_samples):
        # sample multiplicative uncertainties uniformly in log-space
        d_factor = random.uniform(uncertainty_factors["diameter_m"][0], uncertainty_factors["diameter_m"][1])
        v_factor = random.uniform(uncertainty_factors["velocity_m_s"][0], uncertainty_factors["velocity_m_s"][1])
        rho_factor = random.uniform(uncertainty_factors["density_kg_m3"][0], uncertainty_factors["density_kg_m3"][1])
        strength_factor = random.uniform(uncertainty_factors["material_strength_pa"][0], uncertainty_factors["material_strength_pa"][1])

        ast = Asteroid(
            diameter_m=asteroid_nominal.diameter_m * d_factor,
            density_kg_m3=asteroid_nominal.density_kg_m3 * rho_factor,
            velocity_m_s=asteroid_nominal.velocity_m_s * v_factor,
            entry_angle_deg=asteroid_nominal.entry_angle_deg,
            material_strength_pa=asteroid_nominal.material_strength_pa * strength_factor,
            porosity=asteroid_nominal.porosity,
            name=f"{asteroid_nominal.name}_mc{i+1}",
        )

        # apply mitigation if present
        if mitigation:
            if "delta_v_m_s" in mitigation and "time_to_impact_s" in mitigation:
                _, lateral = apply_delta_v_deflection(ast, mitigation["delta_v_m_s"], mitigation["time_to_impact_s"])
                # if lateral > Earth's radius -> no impact (skip)
                if lateral > 6.37e6:
                    samples.append({"impact": False, "killed": 0.0, "energy_mt": 0.0, "airburst_h_m": None})
                    killed_list.append(0.0)
                    energy_list.append(0.0)
                    airburst_alt_list.append(None)
                    continue
            if "fragment_count" in mitigation and mitigation["fragment_count"] > 1:
                frags = fragmentation_model(ast, mitigation["fragment_count"])
            else:
                frags = [ast]
        else:
            frags = [ast]

        total_killed = 0.0
        total_energy_mt = 0.0
        # sum contribution from fragments
        for frag in frags:
            E_mt = frag.kinetic_energy_megatons()
            total_energy_mt += E_mt
            ab_h, _ = breakup_altitude_estimate(frag)
            # choose psi lethal based on whether airburst occurs: airburst tends to distribute energy; ground impact more destructive locally
            psi_lethal = 5.0 if ab_h is None else 3.0
            cas = estimate_casualties(E_mt, population_density_per_km2, psi_lethal)
            total_killed += cas["estimated_killed"]

        samples.append({"impact": True, "killed": total_killed, "energy_mt": total_energy_mt, "airburst_h_m": ab_h})
        killed_list.append(total_killed)
        energy_list.append(total_energy_mt)
        airburst_alt_list.append(ab_h)

    summary = {
        "n_samples": n_samples,
        "mean_killed": float(np.mean(killed_list)),
        "median_killed": float(np.median(killed_list)),
        "p95_killed": float(np.percentile(killed_list, 95)),
        "mean_energy_mt": float(np.mean(energy_list)),
        "median_energy_mt": float(np.median(energy_list)),
        "samples": samples,
    }
    return summary


# ---------------------
# Visualization utilities
# ---------------------
def plot_energy_vs_diameter(diameters_m: np.ndarray, energies_mt: np.ndarray, title: str = "Energy vs Diameter"):
    plt.figure(figsize=(8, 5))
    plt.loglog(diameters_m, energies_mt, marker="o", linewidth=1)
    plt.xlabel("Diameter (m)")
    plt.ylabel("Kinetic Energy (MT TNT)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_impact_footprint(energy_mt: float, population_density_per_km2: float, center=(0, 0)):
    """
    Plot a simple impact footprint showing concentric damage rings for different psi thresholds.
    """
    psi_levels = [1.0, 2.0, 3.0, 5.0, 10.0]
    radii_m = [blast_radius_overpressure_psi(energy_mt, psi) for psi in psi_levels]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(psi_levels)))
    for r, psi, c in zip(radii_m[::-1], psi_levels[::-1], colors[::-1]):
        circle = plt.Circle(center, r / 1000.0, color=c, alpha=0.35, label=f"{psi} psi: {m2km(r):.1f} km")
        ax.add_patch(circle)

    max_r_km = max(m2km(r) for r in radii_m) * 1.2
    ax.set_xlim(-max_r_km, max_r_km)
    ax.set_ylim(-max_r_km, max_r_km)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title(f"Impact footprint (Energy = {energy_mt:.2f} MT)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_casualty_histogram(killed_samples: List[float]):
    plt.figure(figsize=(7, 4))
    plt.hist(killed_samples, bins=40, edgecolor="black", alpha=0.7)
    plt.xlabel("Estimated Killed")
    plt.ylabel("Samples")
    plt.title("Monte Carlo Estimated Killed Distribution")
    plt.grid(True, ls="--", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ---------------------
# Example usage / CLI
# ---------------------
def example_run():
    print("ASTEROID IMPACT SIMULATOR - Example Run")
    # Example asteroid: 100 m diameter stony object
    ast = Asteroid(
        diameter_m=100.0,
        density_kg_m3=3000.0,
        velocity_m_s=17000.0,
        entry_angle_deg=45.0,
        material_strength_pa=5e6,
        name="Example_100m_stone",
    )
    print(ast)
    E_mt = ast.kinetic_energy_megatons()
    print(f"Kinetic energy: {E_mt:.2f} megatons TNT")

    # Atmospheric breakup
    ab_h, q_sea = breakup_altitude_estimate(ast)
    if ab_h is None:
        print("No airburst predicted; likely ground impact under this simple model.")
    else:
        print(f"Estimated breakup altitude (airburst) ≈ {ab_h/1000.0:.1f} km")

    # Crater estimate
    crater_m = crater_diameter_estimate(E_mt)
    print(f"Approx transient crater diameter: {crater_m:.0f} m (~{crater_m/1000.0:.2f} km)")

    # Casualties (assume population density, e.g., 1500 people/km^2 ~ dense city)
    pop_density_km2 = 1500.0
    casualties = estimate_casualties(E_mt, pop_density_km2, psi_lethal=3.0)
    print("Casualty estimate (very rough):")
    print(f" Blast radius (m) for 3 psi: {casualties['blast_radius_m']:.0f} m")
    print(f" Area (km^2): {casualties['area_km2']:.2f}, Population inside: {casualties['pop_inside']:.0f}")
    print(f" Estimated killed: {casualties['estimated_killed']:.0f}, injured: {casualties['estimated_injured']:.0f}")

    # Visualizations
    diameters = np.logspace(math.log10(10), math.log10(1000), 20)
    energies = np.array([Asteroid(d, density_kg_m3=3000.0, velocity_m_s=17000.0).kinetic_energy_megatons() for d in diameters])
    plot_energy_vs_diameter(diameters, energies, title="Energy vs Diameter (example)")

    plot_impact_footprint(E_mt, pop_density_km2)

    # Monte Carlo with mitigation example: delta-v small and fragmentation
    mitigation = {"delta_v_m_s": 0.01, "time_to_impact_s": 365 * 24 * 3600.0, "fragment_count": 5}
    summary = monte_carlo_simulation(ast, population_density_per_km2=pop_density_km2, n_samples=300, mitigation=mitigation)
    print("\nMonte Carlo Summary with mitigation:")
    print(f" mean killed: {summary['mean_killed']:.1f}, median: {summary['median_killed']:.1f}, 95p: {summary['p95_killed']:.1f}")
    # Plot histogram
    plot_casualty_histogram([s["killed"] for s in summary["samples"]])

    print("Example run complete. Note: All models are simplified and tunable.")


if __name__ == "__main__":
    example_run()
