
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# PFR Core (fixed units)
# ----------------------------
R = 8.314462618  # J/mol/K

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_max(x, lo):
    return lo if x < lo else x

def arrhenius(k0, Ea, T):
    return k0 * np.exp(-Ea / (R * T))

def ideal_gas_ctot(P_pa, T_k):
    return P_pa / (R * T_k)  # mol/m3

def mw_mix_kg_per_mol(F, MW_g):
    Ft = max(np.sum(F), 1e-30)
    y = F / Ft
    mw_g = np.dot(y, MW_g)
    return mw_g / 1000.0  # kg/mol

def ergun_dPdz(u, rho, mu, eps, dp_m):
    term1 = 150.0 * ((1 - eps) ** 2) / (eps ** 3) * (mu * u) / (dp_m ** 2)
    term2 = 1.75 * (1 - eps) / (eps ** 3) * (rho * u * u) / dp_m
    return -(term1 + term2)  # Pa/m (negative)

def simulate_pfr(inputs):
    # unpack
    phase = inputs["phase"]

    FA0 = max(inputs["FA0"], 0.0)
    FB0 = max(inputs["FB0"], 0.0)
    FC0 = max(inputs["FC0"], 0.0)
    FD0 = max(inputs["FD0"], 0.0)

    T0 = inputs["T0_C"] + 273.15
    P0 = inputs["P0_bar"] * 1e5  # Pa

    L = safe_max(inputs["L"], 1e-6)
    Di = safe_max(inputs["Di"], 1e-6)
    nSteps = int(inputs["nSteps"])

    packedBed = inputs["packedBed"]
    eps = clamp(inputs["porosity"], 0.05, 0.9)
    dp = safe_max(inputs["dp_mm"], 0.01) / 1000.0  # m
    activity = max(inputs["activity"], 0.0)

    heatTransfer = inputs["heatTransfer"]
    U = inputs["U"]
    Tcool = inputs["Tcool_C"] + 273.15

    radialModel = inputs["radialModel"]
    hi = inputs["hi"]

    pressureDrop = inputs["pressureDrop"]
    mu = inputs["mu"]

    k0 = inputs["k0"]
    Ea = inputs["Ea"]
    ordA = inputs["ordA"]
    ordB = inputs["ordB"]

    Cp_molar = safe_max(inputs["Cp_molar"], 1.0)
    dHrxn = inputs["dHrxn"]  # J/mol (exo negative)

    MW = np.array([inputs["MW_A"], inputs["MW_B"], inputs["MW_C"], inputs["MW_D"]], dtype=float)

    rho_liquid = inputs["rho_liquid"]
    rho_gas_override = inputs["rho_gas_override"]

    # geometry
    Ac = np.pi * Di * Di / 4.0
    perim = np.pi * Di
    dz = L / nSteps

    # states
    Tcore = T0
    Twall = T0
    P = P0
    F = np.array([FA0, FB0, FC0, FD0], dtype=float)
    FA_in = max(FA0, 1e-30)

    # outputs
    z = np.zeros(nSteps + 1)
    Tcore_C = np.zeros(nSteps + 1)
    Twall_C = np.zeros(nSteps + 1)
    P_bar = np.zeros(nSteps + 1)
    X_A = np.zeros(nSteps + 1)
    CA = np.zeros(nSteps + 1)
    CB = np.zeros(nSteps + 1)
    rr = np.zeros(nSteps + 1)

    Tmax = -1e9
    zhot = 0.0

    for i in range(nSteps + 1):
        zi = i * dz
        z[i] = zi
        Tcore_C[i] = Tcore - 273.15
        Twall_C[i] = Twall - 273.15
        P_bar[i] = P / 1e5

        FA, FB, FC, FD = F
        Ft = max(np.sum(F), 1e-30)

        X_A[i] = (FA_in - FA) / FA_in

        if Tcore_C[i] > Tmax:
            Tmax = Tcore_C[i]
            zhot = zi

        # stop reaction if depleted
        if FA <= 1e-20 or FB <= 1e-20:
            CA[i] = 0.0
            CB[i] = 0.0
            rr[i] = 0.0
            continue

        # concentrations + velocity
        if phase == "gas":
            Ctot = ideal_gas_ctot(P, Tcore)
            yA = FA / Ft
            yB = FB / Ft
            CAi = yA * Ctot
            CBi = yB * Ctot
            vdot = Ft / Ctot
            u = vdot / Ac
        else:
            rhoL = safe_max(rho_liquid, 1.0)
            MWmix = mw_mix_kg_per_mol(F, MW)
            mdot = Ft * MWmix
            vdot = mdot / rhoL
            u = vdot / Ac
            CAi = FA / vdot
            CBi = FB / vdot

        CA[i] = CAi
        CB[i] = CBi

        # kinetics
        k = arrhenius(k0, Ea, Tcore)
        r = activity * k * (CAi ** ordA) * (CBi ** ordB)  # mol/m3/s
        if packedBed:
            r *= (1 - eps)
        rr[i] = r

        # species balance (Euler)
        dFA_dz = -r * Ac
        dFB_dz = -r * Ac
        dFC_dz = +r * Ac

        # energy
        Qrxn_per_m = (-dHrxn) * r * Ac  # W/m

        if radialModel:
            Qcw = -hi * perim * (Tcore - Twall)
            Qwc = (-U * perim * (Twall - Tcool)) if heatTransfer else 0.0

            dTcore_dz = (Qrxn_per_m + Qcw) / (Ft * Cp_molar)
            dTwall_dz = (-Qcw + Qwc) / (Ft * Cp_molar)
        else:
            Qht = (-U * perim * (Tcore - Tcool)) if heatTransfer else 0.0
            dTcore_dz = (Qrxn_per_m + Qht) / (Ft * Cp_molar)
            dTwall_dz = 0.0

        # pressure drop
        dP_dz = 0.0
        if pressureDrop and packedBed:
            if phase == "gas":
                if rho_gas_override and rho_gas_override > 0:
                    rho = rho_gas_override
                else:
                    MWmix = mw_mix_kg_per_mol(F, MW)
                    rho = (P * MWmix) / (R * Tcore)
            else:
                rho = safe_max(rho_liquid, 1.0)

            dP_dz = ergun_dPdz(u, rho, mu, eps, dp)

        # update
        F[0] = max(F[0] + dFA_dz * dz, 0.0)
        F[1] = max(F[1] + dFB_dz * dz, 0.0)
        F[2] = max(F[2] + dFC_dz * dz, 0.0)

        Tcore = safe_max(Tcore + dTcore_dz * dz, 1.0)
        Twall = safe_max(Twall + dTwall_dz * dz, 1.0)

        P = safe_max(P + dP_dz * dz, 1e3)  # min 0.01 bar

    kpis = {
        "Tmax_C": float(np.max(Tcore_C)),
        "z_hot_m": float(zhot),
        "Xout": float(X_A[-1]),
        "Pout_bar": float(P_bar[-1]),
        "rmax": float(np.max(rr)),
    }
    profiles = {
        "z": z,
        "Tcore_C": Tcore_C,
        "Twall_C": Twall_C,
        "P_bar": P_bar,
        "X_A": X_A,
        "CA": CA,
        "CB": CB,
        "r": rr,
    }
    return profiles, kpis

def tube_heatmap(z, tcore, twall, nr=120):
    r = np.linspace(0, 1, nr)  # r/R
    field = np.zeros((nr, len(z)))
    for i, ri in enumerate(r):
        field[i, :] = tcore + (twall - tcore) * ri
    return r, field

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Cumene Hotspot (Single Tube)", layout="wide")

st.title("Cumene Alkylation — Single Tube Hotspot Explorer")
st.caption("Tek dosya, tek tık: input → PFR simülasyonu → hotspot + heatmap + grafikler.")

with st.sidebar:
    st.header("Inputs")

    phase = st.selectbox("Phase", ["gas", "liquid"], index=0)

    st.subheader("Inlet")
    FA0 = st.number_input("FA0 (Benzene) mol/s", value=10.0, step=0.1)
    FB0 = st.number_input("FB0 (Propylene) mol/s", value=1.0, step=0.1)
    FC0 = st.number_input("FC0 (Cumene) mol/s", value=0.0, step=0.1)
    FD0 = st.number_input("FD0 (placeholder) mol/s", value=0.0, step=0.1)
    T0_C = st.number_input("T0 (°C)", value=150.0, step=1.0)
    P0_bar = st.number_input("P0 (bar)", value=20.0, step=1.0)

    st.subheader("Geometry")
    L = st.number_input("L (m)", value=10.0, step=0.5)
    Di = st.number_input("Di (m)", value=0.05, step=0.005)

    st.subheader("Bed / ΔP")
    packedBed = st.checkbox("Packed bed", value=True)
    pressureDrop = st.checkbox("Pressure drop (Ergun)", value=True)
    porosity = st.number_input("Porosity ε", value=0.4, step=0.01)
    dp_mm = st.number_input("Particle dp (mm)", value=2.0, step=0.1)
    mu = st.number_input("Viscosity μ (Pa·s)", value=0.001, step=0.0001, format="%.6f")
    activity = st.number_input("Activity a", value=1.0, step=0.1)

    rho_liquid = st.number_input("ρ_liquid (kg/m3) [only liquid]", value=800.0, step=10.0)
    rho_gas_override = st.number_input("ρ_gas override (kg/m3) [0=auto]", value=0.0, step=0.1)

    st.subheader("Heat transfer")
    heatTransfer = st.checkbox("Heat transfer enabled", value=True)
    radialModel = st.checkbox("Core/Wall (1.5D)", value=True)
    U = st.number_input("U (W/m2-K)", value=300.0, step=10.0)
    Tcool_C = st.number_input("Tcool (°C)", value=130.0, step=1.0)
    hi = st.number_input("hi core↔wall (W/m2-K)", value=1500.0, step=50.0)

    st.subheader("Kinetics (power law)")
    k0 = st.number_input("k0", value=1e-3, step=1e-3, format="%.6g")
    Ea = st.number_input("Ea (J/mol)", value=80000.0, step=1000.0)
    ordA = st.number_input("order A", value=1.0, step=0.1)
    ordB = st.number_input("order B", value=1.0, step=0.1)

    st.subheader("Thermo")
    Cp_molar = st.number_input("Cp_molar (J/mol-K)", value=160.0, step=5.0)
    dHrxn = st.number_input("ΔHrxn (J/mol) (exo negative)", value=-90000.0, step=5000.0)

    st.subheader("MW (g/mol)")
    MW_A = st.number_input("MW_A (Benzene)", value=78.11, step=0.1)
    MW_B = st.number_input("MW_B (Propylene)", value=42.08, step=0.1)
    MW_C = st.number_input("MW_C (Cumene)", value=120.19, step=0.1)
    MW_D = st.number_input("MW_D (DIPB placeholder)", value=162.27, step=0.1)

    st.subheader("Numerics")
    nSteps = st.slider("nSteps", 100, 2000, 600, 50)

    run = st.button("Run Simulation", type="primary")

if run:
    inputs = dict(
        phase=phase,
        FA0=FA0, FB0=FB0, FC0=FC0, FD0=FD0,
        T0_C=T0_C, P0_bar=P0_bar,
        L=L, Di=Di,
        packedBed=packedBed, porosity=porosity, dp_mm=dp_mm, activity=activity,
        heatTransfer=heatTransfer, U=U, Tcool_C=Tcool_C, radialModel=radialModel, hi=hi,
        pressureDrop=pressureDrop, mu=mu, rho_liquid=rho_liquid, rho_gas_override=rho_gas_override,
        k0=k0, Ea=Ea, ordA=ordA, ordB=ordB,
        Cp_molar=Cp_molar, dHrxn=dHrxn,
        MW_A=MW_A, MW_B=MW_B, MW_C=MW_C, MW_D=MW_D,
        nSteps=nSteps
    )

    profiles, kpis = simulate_pfr(inputs)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tmax (°C)", f"{kpis['Tmax_C']:.2f}")
    col2.metric("z* (m)", f"{kpis['z_hot_m']:.2f}")
    col3.metric("Xout (%)", f"{100*kpis['Xout']:.2f}")
    col4.metric("Pout (bar)", f"{kpis['Pout_bar']:.2f}")
    col5.metric("max r", f"{kpis['rmax']:.3e}")

    z = profiles["z"]
    tcore = profiles["Tcore_C"]
    twall = profiles["Twall_C"]
    pbar = profiles["P_bar"]
    xA = profiles["X_A"]
    rr = profiles["r"]

    st.subheader("Tube Heatmap (COMSOL vari) + Colorbar")
    rgrid, field = tube_heatmap(z, tcore, twall)

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(
        field,
        aspect="auto",
        origin="lower",
        extent=[z.min(), z.max(), rgrid.min(), rgrid.max()]
    )
    plt.colorbar(im, ax=ax, label="T (°C)")
    ax.axvline(kpis["z_hot_m"], linestyle="--", linewidth=2)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("r/R (-)")
    ax.set_title("Temperature map across radius (linear interpolation core→wall)")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Profiles")
    c1, c2 = st.columns(2)

    with c1:
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(z, tcore, label="Tcore (°C)")
        ax.plot(z, twall, label="Twall (°C)")
        ax.axvline(kpis["z_hot_m"], linestyle="--")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("T (°C)")
        ax.legend()
        ax.set_title("Temperature profile")
        st.pyplot(fig, clear_figure=True)

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(z, xA, label="X_A")
        ax.axvline(kpis["z_hot_m"], linestyle="--")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Conversion")
        ax.legend()
        ax.set_title("Conversion profile")
        st.pyplot(fig, clear_figure=True)

    with c2:
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(z, pbar, label="P (bar)")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Pressure (bar)")
        ax.legend()
        ax.set_title("Pressure profile")
        st.pyplot(fig, clear_figure=True)

        fig = plt.figure()
        ax = plt.gca()
        ax.plot(z, rr, label="r (mol/m3/s)")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Rate")
        ax.legend()
        ax.set_title("Rate diagnostic (if ~0, k0 scale mismatch)")
        st.pyplot(fig, clear_figure=True)

else:
    st.info("Sol menüden değerleri ayarla, sonra **Run Simulation**’a bas.")
