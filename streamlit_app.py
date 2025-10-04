##!/usr/bin/env python3
# streamlit_app.py
"""
Asteroid Impact Simulator – Streamlit Version
Educational interactive model for asteroid impact scenarios.
"""

import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use headless backend for Streamlit
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------
# Physical constants
# ---------------------
G = 9.80665
RHO_AIR_SEA_LEVEL = 1.225
SCALE_HEIGHT = 8500.0
JOULES_PER_MEGATON_TNT = 4.184e15

# ---------------------
# Helper functions
# ---------------------
def kinetic_energy_megatons(mass_kg, velocity_m_s):
    return 0.5 * mass_kg * velocity_m_s**2 / JOULES_PER_MEGATON_TNT

def breakup_altitude(strength_pa, velocity_m_s):
    """Return breakup altitude in meters (None if ground impact)."""
    rhs = (2 * strength_pa) / (RHO_AIR_SEA_LEVEL * velocity_m_s**2)
    if rhs >= 1:
        return None
    h = -SCALE_HEIGHT * math.log(rhs)
    return max(0.0, min(h, 1.5e5))

def crater_diameter_m(energy_mt):
    a_km, b = 0.12, 0.30
    return (a_km * energy_mt**b) * 1000.0

def blast_radius_m(energy_mt, psi=3.0):
    C_km = 1.3
    psi_scale = (5.0 / psi) ** 0.4
    return C_km * (energy_mt ** (1 / 3)) * psi_scale * 1000.0

def estimate_casualties(energy_mt, pop_density, psi=3.0):
    r = blast_radius_m(energy_mt, psi)
    area_km2 = math.pi * (r / 1000) ** 2
    pop_inside = area_km2 * pop_density
    lethality_frac = 0.2 if psi <= 3 else 0.5
    injured_frac = 0.5 if psi <= 3 else 0.3
    return {
        "radius_m": r,
        "killed": pop_inside * lethality_frac,
        "injured": pop_inside * injured_frac,
    }

def plot_footprint(energy_mt):
    psi_levels = [1, 2, 3, 5, 10]
    radii = [blast_radius_m(energy_mt, p) / 1000 for p in psi_levels]
    fig, ax = plt.subplots(figsize=(5,5))
    for r, psi in zip(radii[::-1], psi_levels[::-1]):
        ax.add_patch(plt.Circle((0,0), r, color=plt.cm.viridis(psi/10), alpha=0.4, label=f"{psi} psi"))
    maxr = max(radii) * 1.2
    ax.set_xlim(-maxr, maxr)
    ax.set_ylim(-maxr, maxr)
    ax.set_aspect("equal")
    ax.legend(title="Overpressure")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title("Blast Footprint")
    return fig

# ---------------------
# Streamlit UI
# ---------------------
st.title("Asteroid Impact Simulator")

st.sidebar.header("Asteroid Parameters")
diameter_m = st.sidebar.slider("Diameter (m)", 10.0, 2000.0, 100.0, 10.0)
density = st.sidebar.slider("Density (kg/m³)", 1000, 8000, 3000, 100)
velocity = st.sidebar.slider("Velocity (km/s)", 5.0, 40.0, 17.0, 0.5) * 1000
angle = st.sidebar.slider("Entry Angle (°)", 10, 90, 45, 1)
strength = st.sidebar.number_input("Material Strength (Pa)", 1e5, 1e8, 5e6, step=1e6, format="%.0e")
population_density = st.sidebar.number_input("Population Density (people/km²)", 0.0, 20000.0, 1500.0, 100.0)

st.sidebar.header("Mitigation (Optional)")
delta_v = st.sidebar.number_input("Δv applied (m/s)", 0.0, 10.0, 0.0, 0.01)
time_to_impact_days = st.sidebar.number_input("Time before impact (days)", 0.0, 3650.0, 0.0, 10.0)
fragments = st.sidebar.slider("Fragmentation count", 1, 10, 1, 1)

# Derived properties
radius_m = diameter_m / 2
volume_m3 = (4/3) * math.pi * radius_m**3
mass_kg = density * volume_m3
energy_mt = kinetic_energy_megatons(mass_kg, velocity)
altitude = breakup_altitude(strength, velocity)
crater_m = crater_diameter_m(energy_mt)
cas = estimate_casualties(energy_mt, population_density, 3)

# Mitigation effect
miss_distance_km = delta_v * (time_to_impact_days * 86400) / 1000
miss_earth = miss_distance_km > 6371

# Display results
st.subheader("Results")
col1, col2 = st.columns(2)
with col1:
    st.metric("Kinetic Energy", f"{energy_mt:.2f} MT TNT")
    if altitude:
        st.metric("Airburst Altitude", f"{altitude/1000:.1f} km")
    else:
        st.metric("Airburst Altitude", "Ground impact")
    st.metric("Crater Diameter", f"{crater_m/1000:.2f} km")

with col2:
    st.metric("Blast Radius (3 psi)", f"{cas['radius_m']/1000:.1f} km")
    st.metric("Estimated Killed", f"{cas['killed']:,.0f}")
    st.metric("Estimated Injured", f"{cas['injured']:,.0f}")

if miss_earth:
    st.success(f"Deflection success! Miss distance ≈ {miss_distance_km/6371:.2f} Earth radii.")
else:
    if delta_v > 0:
        st.info(f"Lateral displacement ≈ {miss_distance_km:.0f} km — insufficient to miss Earth.")

# Visualization
st.subheader("Visualizations")
fig = plot_footprint(energy_mt)
st.pyplot(fig)

# Optional energy scaling plot
st.write("### Energy vs. Diameter (for comparison)")
sizes = np.logspace(1, 3.5, 30)
energies = [kinetic_energy_megatons(density*(4/3)*math.pi*(d/2)**3, velocity) for d in sizes]
fig2, ax = plt.subplots()
ax.loglog(sizes, energies)
ax.set_xlabel("Diameter (m)")
ax.set_ylabel("Energy (MT TNT)")
ax.set_title("Energy Scaling with Size")
ax.grid(True, which="both", ls="--", alpha=0.5)
st.pyplot(fig2)

st.caption("⚠️ Educational model – uses simplified physics for demonstration.")
