import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    import catppuccin
    import matplotlib as mpl
    from dataclasses import dataclass
    from pathlib import Path

    if mo.app_meta().theme == "dark":
        mpl.style.use(catppuccin.PALETTE.mocha.identifier)
    else:
        mpl.style.use(catppuccin.PALETTE.latte.identifier)
    return dataclass, mo, np, plt, solve_ivp


@app.cell
def _(mo):
    mo.md(
        r"""
    # Hot-Cold Dynamical System

    Newton's law of cooling can be described by:

    $$
        \frac{dT}{dt} = -k(T - T_{m})
    $$

    Here, we have:

    - $T$ is the temperature of the water in the pan.
    - $T_{m}$ is the temperature of the surrounding medium.

    In this notebook, I am interested in the following system:

    Hot water at a temperature $T_H\degree C$ is flowing into a bowl at a rate of $R$ $Ls^{-1}$.
    After some time $t_1$, the inlet temperature switches to a colder $T_C\degree C$. We then allow for this mixture to run for some amount of time $t_{final}$.

    We are interested in modelling this system in order to determine the value of $t_{final}$ if we have a goal for the mixed temperature $T_{final}$. That is, we wish to answer the question of solving $f(t_{final}) = T_{final}$, for some given $T_{final}$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Defining the System
    ### Assumptions

    We will make sever assumptions about this system:

    1. Pan accumulates all inflow: Every drop you pour stays in the pan; there is no overflow.  Thus its volume grows unbounded as you pour.
    2. Perfect mixing: At every instant the water in the pan is completely uniform in temperature.
    3. Negligible heat losses: We ignore any heat transfer to the pan walls or to the air—only mixing matters.
    4. Constant specific heat and density: We take water’s density and specific heat both equal to 1 in our units, so "volume" doubles as "mass" and energy = volume $\times$ temperature.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### System State

    In our system, $V(t)$ will be the volume of water in the pan at time $t$, and $T(t)$ will be the temperature of the water in the pan at time $t$. We can then define the state $\mathbf{x}(t)$ of the system as:

    $$
        \mathbf{x}(t) = \begin{bmatrix}
            V(t) \\
            T(t)
        \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Inflow Temperature

    The input temperature can be defined as a piecewise function:

    $$
        T_{in}(t) = \begin{cases}
            T_H & \text{if } t \le t_1 \\
            T_C & \text{if } t \gt t_1
        \end{cases}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Volume

    Volume is given as a constant inflow rate $R$. Thus, we have:

    $$
        \frac{dV}{dt} = R
    $$

    Given the initial condition $V(0) = V_0$, we can integrate this to get:

    $$
        V(t) = V_0 + Rt
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Thermal Energy

    The total thermal energy in the pan is given by:

    $$
        E(t) = V(t)T(t)
    $$

    As we add water, the energy changes according to:

    $$
        \frac{dE}{dt} = R T_{in}(t)
    $$

    When we differentiate $E(t)$, we can use the product rule:

    $$
        \frac{dE}{dt} = V(t) \frac{dT}{dt} + T(t) \frac{dV}{dt}
    $$

    And so, by equating the results, substituting known values, and simplifying, we find

    $$
        \frac{dT}{dt} = \frac{R}{V(t)} (T_{in}(t) - T(t))
    $$

    or more simply:

    $$
        \frac{dT}{dt} = R \frac{T_{in}(t) - T(t)}{V_0 + Rt}
    $$
    """
    )
    return


@app.cell
def _(C, H, R, V0, t1):
    def T_in(t):
        """
        Return the inlet temperature based on the time.

        After time t1, the water is instantly changed to the colder temp C."""
        return H if t < t1 else C


    def ode(t, y):
        V, T = y
        dVdt = R
        dTdt = R * (T_in(t) - T) / (V0 + R * t)
        return [dVdt, dTdt]
    return T_in, ode


@app.cell
def _(dataclass):
    @dataclass
    class OdeMethods:
        RK45: str = "RK45"
        Radau: str = "Radau"
        BDF: str = "BDF"


    @dataclass
    class Config:
        R: float = 0.2  # Flow rate (volume units per time)
        H: float = 90.0  # Hot water temperature
        C: float = 20.0  # Cold water temperature
        t1: float = 5.0  # Time at which inlet temperature switches from H to C
        V0: float = 1.0  # Initial volume of water in the pan
        T0: float = 20.0  # Initial temperature of the water
        t_final: float = 10.0  # Total simulation time
        ode_method: str = OdeMethods.Radau  # ODE solver method to use
        k: float = 0.1  # Cooling rate constant
        T_m: float = 25.0  # Ambient temperature of the surrounding medium


    config = Config()
    return OdeMethods, config


@app.cell
def _(OdeMethods, config, mo):
    sider_R = mo.ui.slider(
        start=0.001,
        stop=1.0,
        step=0.001,
        value=config.R,
        label="Flow Rate ($R$)",
        show_value=True,
    )
    sider_H = mo.ui.slider(
        start=10.0,
        stop=100.0,
        step=1.0,
        value=config.H,
        label="Hot Water Temperature ($T_H$)",
        show_value=True,
    )
    sider_C = mo.ui.slider(
        start=0.0,
        stop=100.0,
        step=1.0,
        value=config.C,
        label="Cold Water Temperature ($T_C$)",
        show_value=True,
    )
    sider_t1 = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.1,
        value=config.t1,
        label="Time Switch ($t_1$)",
        show_value=True,
    )
    sider_V0 = mo.ui.slider(
        start=0.01,
        stop=10.0,
        step=0.1,
        value=config.V0,
        label="Initial Volume ($V_0$)",
        show_value=True,
    )
    sider_T0 = mo.ui.slider(
        start=10.0,
        stop=100.0,
        step=1.0,
        value=config.T0,
        label="Initial Temperature ($T_0$)",
        show_value=True,
    )
    sider_t_final = mo.ui.slider(
        start=10.0,
        stop=20.0,
        step=1.0,
        value=config.t_final,
        label="Total Simulation Time ($t_{final}$)",
        show_value=True,
    )
    dropdown_ode_method = mo.ui.dropdown(
        options=[
            OdeMethods.RK45,
            OdeMethods.Radau,
            OdeMethods.BDF,
        ],
        value=config.ode_method,
        label="ODE Method",
    )

    controls = mo.vstack(
        [
            sider_R,
            sider_H,
            sider_C,
            sider_t1,
            sider_V0,
            sider_T0,
            sider_t_final,
            dropdown_ode_method,
        ]
    )

    controls
    return (
        controls,
        dropdown_ode_method,
        sider_C,
        sider_H,
        sider_R,
        sider_T0,
        sider_V0,
        sider_t1,
        sider_t_final,
    )


@app.cell
def _(
    dropdown_ode_method,
    sider_C,
    sider_H,
    sider_R,
    sider_T0,
    sider_V0,
    sider_t1,
    sider_t_final,
):
    R = sider_R.value
    H = sider_H.value
    C = sider_C.value
    t1 = sider_t1.value
    V0 = sider_V0.value
    T0 = sider_T0.value
    t_final = sider_t_final.value
    ode_method = dropdown_ode_method.value
    return C, H, R, T0, V0, ode_method, t1, t_final


@app.cell
def _(T0, V0, np, ode, ode_method, solve_ivp, t_final):
    # Time points where solution is computed
    t_eval = np.linspace(0, t_final, 1000)

    # Solve the ODE system
    sol = solve_ivp(ode, (0, t_final), [V0, T0], t_eval=t_eval, method=ode_method)
    return sol, t_eval


@app.cell
def _(plt, sol, t1):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot Temperature vs Time

    _ax1.plot(sol.t, sol.y[1])
    _ax1.axvline(t1, linestyle="--")
    _ax1.set_xlabel("Time")
    _ax1.set_ylabel("Temperature")
    _ax1.set_title("Temperature vs Time")
    _ax1.grid()

    _ax2.plot(sol.t, sol.y[0])
    _ax2.axvline(t1, linestyle="--")
    _ax2.set_xlabel("Time")
    _ax2.set_ylabel("Volume")
    _ax2.set_title("Volume vs Time")
    _ax2.grid()

    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Solving the ODE

    We know that at time $t$, the inflow temperature is given by:

    $$
        T_{in}(t) = \begin{cases}
            T_H & \text{if } t \le t_1 \\
            T_C & \text{if } t \gt t_1
        \end{cases}
    $$

    Furthermore, we described the temperature of the water in the pan as:

    $$
        \frac{dT}{dt} = R \frac{T_{in}(t) - T(t)}{V_0 + Rt}
    $$

    Which can be rewritten as:

    $$
        (V_0 + Rt) \frac{dT}{dt} + R T(t) = R T_{in}(t)
    $$

    Thus, we find:

    $$
        \frac{d}{dt}\left( (V_0 + Rt) T \right) = R T_{in}(t)
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Case 1: $\quad 0 \le t \le t_1$

    Integrating both sides gives:

    $$
        \begin{align*}
        \int_0^{t} \frac{d}{dt}\left( (V_0 + Rt) T \right) dt &= \int_0^{t} R T_{in}(t) dt\\
        (V_0 + Rt) T(t) - V_0 T_0 &= R T_H\int_0^{t} dt\\
        (V_0 + R t_1) T(t) - V_0 T_0 &= R T_H t
        \end{align*}
    $$

    Thus,

    $$
        T(t) = \frac{R T_H t + V_0 T_0}{V_0 + R t}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Case 2: $\quad t \gt t_1$

    Integrating both sides gives:

    $$
        \begin{align*}
        \int_{t_1}^{t} \frac{d}{dt}\left( (V_0 + Rt) T \right) dt &= \int_{t_1}^{t} R T_{in}(t) dt\\
        (V_0 + Rt) T(t) - (V_0 + Rt_1) T(t_1) &= R T_C\int_{t_1}^{t} dt\\
        (V_0 + Rt) T(t) - (V_0 + Rt_1) T(t_1) &= R T_C (t - t_1)
        \end{align*}
    $$

    Giving:

    $$
        T(t) = \frac{R T_C (t - t_1) + (V_0 + Rt_1) T(t_1)}{V_0 + Rt}
    $$

    From case 1, when $t=t_1$, we have:

    $$
        T(t_1) = \frac{R T_H t_1 + V_0 T_0}{V_0 + R t_1}
    $$

    Thus, finally:

    $$
        T(t) = \frac{R T_C (t - t_1)}{V_0 + Rt} + \frac{V_0 + Rt_1}{V_0 + Rt} \frac{R T_H t_1 + V_0 T_0}{V_0 + R t_1}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Final Result

    By combing the results, we see that we have found:

    $$
        T(t) = \begin{cases}
            \displaystyle \frac{R T_H t + V_0 T_0}{V_0 + R t} & \text{if } 0 \le t \le t_1 \\[1.5em]
            \displaystyle \frac{R T_C (t - t_1)}{V_0 + Rt} + \frac{V_0 + Rt_1}{V_0 + Rt} \frac{R T_H t_1 + V_0 T_0}{V_0 + R t_1} & \text{if } t \gt t_1
        \end{cases}
    $$
    """
    )
    return


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(C, H, R, T0, V0, plt, sol, t1):
    def soln_T(t):
        denom = V0 + R * t
        if t <= t1:
            return (R * H * t + V0 * T0) / denom
        else:
            T1 = (R * H * t1 + V0 * T0) / (V0 + R * t1)
            return (R * C * (t - t1) + (V0 + R * t1) * T1) / denom


    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.plot(sol.t, sol.y[1], label="Numerical Solution")
    _ax.plot(
        sol.t,
        [soln_T(t) for t in sol.t],
        label="Analytical Solution",
        linestyle="--",
    )
    _ax.axvline(t1, linestyle="--", color="red", label="Switch Time ($t_1$)")
    _ax.set_xlabel("Time")
    _ax.set_ylabel("Temperature")
    _ax.set_title("Temperature vs Time (Numerical vs Analytical)")
    _ax.legend()
    _ax.grid()

    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Extending the Model to Consider Heat Loss
    <small><i>Updated: 2025/06/29</i></small>

    Currently, this model fails to consider how the volume of water looses heat to the surrounding environment. We can extend the model by adding a cooling term to the temperature equation, as described earlier on Newton's law of cooling:

    $$
        \frac{dT}{dt} = R \frac{T_{in}(t) - T(t)}{V_0 + Rt} - k(T(t) - T_m)
    $$

    Here, $k$ is a constant that describes the rate of heat loss to the environment, and $T_m$ is the temperature of the surrounding medium. Now, we can carefully manipulate the equation in order to find a solution.

    $$
        \begin{align*}
            \frac{dT}{dt} &= R \frac{T_{in}(t) - T(t)}{V_0 + Rt} - k(T(t) - T_m)\\
            &= R \frac{T_{in}(t) - T(t)}{V(t)} - k(T(t) - T_m)\\
            V(t) \frac{dT}{dt} &= R T_{in}(t) - R T(t) - k V(t)(T(t) - T_m)\\
            &= R T_{in}(t) - R T(t) - k V(t) T(t) + k V(t) T_m
        \end{align*}\\
    $$

    $$
        \begin{align*}
            V(t) \frac{dT}{dt} + (R + k V(t)) T(t) &= R T_{in}(t) + k V(t) T_m\\
            \frac{dT}{dt} + \frac{R + k V(t)}{V(t)} T(t) &= \frac{R T_{in}(t) + k V(t) T_m}{V(t)}\\
            \frac{dT}{dt} + \left(\frac{R}{V_0 + Rt} + k\right) T(t) &= \frac{R T_{in}(t)}{V_0 + Rt} + k T_m
        \end{align*}
    $$

    Solving for the integrating factor, $\mu(t)$:

    $$
        \begin{align*}
            \mu &= \exp{\left(\int\frac{R}{V_0 + Rt} + k\, dt\right)}\\
                &= \exp{(\ln(V_0 + Rt) + kt)}\\
                &= (V_0 + Rt)e^{kt}
        \end{align*}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    And therefore, we can write our ODE as:

    $$
    \begin{align*}
        \frac{d}{dt}\left[(V_0 + Rt)e^{kt} T(t) \right] &= R T_{in}(t) e^{kt} + k T_m (V_0 + Rt)e^{kt}
    \end{align*}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Case 1: $\quad 0 \le t \le t_1$

    As before, we can integrate both sides. Noting that for $t\in[0,t_1]$, $T_{in}(t)=T_H$, we find through integration by parts that:

    $$
        \begin{align*}
            \int_0^{t} \frac{d}{dt}\left[(V_0 + Rt)e^{kt} T(t) \right] dt &= \int_0^{t} R T_{in}(t) e^{kt} + k T_m (V_0 + Rt)e^{kt} dt\\
            &= \int_0^{t} \left[R T_H + k T_m (V_0 + Rt)\right] e^{kt} dt\\
            &= \left.\left[ \frac{RT_H}{k} e^{kt} + k T_m \int (V_0 + Rt) e^{kt} dt \right]\right|_0^t\\
            &= \left.\left[ \frac{RT_H}{k} e^{kt} + k T_m \left( \frac{V_0+Rt}{k}e^{kt} - \frac{R}{k^2} e^{kt} \right) \right]\right|_0^t\\
            &= \left.\left[ \frac{RT_H}{k} e^{kt} + T_m \left( V(t) - \frac{R}{k} \right) e^{kt} \right]\right|_0^t\\
            &= \left.\left[ \left(T_m V(t) + \frac{R}{k} (T_H - T_m) \right)e^{kt} \right]\right|_0^t\\
            \therefore V(t)e^{kt}T(t) - V_0T_0 &= \left.\left[ \left(T_m V(t) + \frac{R}{k} (T_H - T_m) \right)e^{kt} \right]\right|_0^t\\
            &= \left(T_m V(t) + \frac{R}{k} (T_H - T_m) \right)e^{kt} - T_m V_0 - \frac{R}{k} (T_H - T_m)\\
            V(t)e^{kt}T(t) &= \left(T_m V(t) + \frac{R}{k} (T_H - T_m) \right)e^{kt} - V_0 (T_m - T_0) - \frac{R}{k} (T_H - T_m)\\
            \therefore T(t) &= T_m + \frac{R}{kV(t)}(T_H-T_m) - \left( V_0 (T_m - T_0) + \frac{R}{k} (T_H - T_m) \right) \frac{e^{-kt}}{V(t)}\\
            &= T_m + \frac{R}{kV(t)}(T_H-T_m) \left( 1 - e^{-kt} \right) -  \frac{V_0}{V(t)} (T_m - T_0) e^{-kt}
        \end{align*}
    $$

    So, therefore, for $0 \le t \le t_1$:

    $$
        T(t) = T_m + \frac{R}{kV(t)}(T_H-T_m) \left( 1 - e^{-kt} \right) -  \frac{V_0}{V(t)} (T_m - T_0) e^{-kt}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Case 2: $\quad t \ge t_1$

    Simiularly, we can integrate as above, noting that for $t \in [t_1, \infty)$, $T_{in}(t)=T_C$:

    $$
        \begin{align*}
            \int_{t_1}^{t} \frac{d}{dt}\left[(V_0 + Rt)e^{kt} T(t) \right] dt &= \left.\left[ \left(T_m V(t) + \frac{R}{k} (T_C - T_m) \right)e^{kt} \right]\right|^t_{t_1}\\
            &= T_m V(t)e^{kt} + \frac{R}{k} (T_C - T_m) e^{kt} - T_m V(t_1)e^{kt_1} - \frac{R}{k} (T_C - T_m) e^{kt_1}\\
            &= T_m \left(V(t)e^{kt} - V(t_1)e^{kt_1}\right) + \frac{R}{k} (T_C - T_m) \left( e^{kt} - e^{kt_1} \right)\\
            \therefore V(t) e^{kt} T(t) - V(t_1)e^{kt_1}T_1 &= T_m \left(V(t)e^{kt} - V(t_1)e^{kt_1}\right) + \frac{R}{k} (T_C - T_m) \left( e^{kt} - e^{kt_1} \right)\\
            \therefore T(t) &= T_m \left(1 - \frac{V(t_1)}{V(t)} e^{-k(t-t_1)}\right) + \frac{R}{kV(t)} (T_C - T_m) \left(1 - e^{-k(t - t_1)} \right) + \frac{V(t_1)}{V(t)} T_1 e^{-k(t-t_1)}\\
            &= T_m + \frac{R}{kV(t)} (T_C - T_m) \left(1 - e^{-k(t - t_1)} \right) + \frac{V(t_1)}{V(t)} e^{-k(t-t_1)} (T_1 - T_m)
        \end{align*}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Using the result from case 1, we can solve for $T_1 = T(t_1)$:

    $$
    \begin{align*}
        T(t) &= T_m + \frac{R}{kV(t)}(T_H-T_m) \left( 1 - e^{-kt} \right) -  \frac{V_0}{V(t)} (T_m - T_0) e^{-kt}\\
        \therefore T(t_1) &= T_m + \frac{R}{kV(t_1)}(T_H-T_m) \left( 1 - e^{-kt_1} \right) -  \frac{V_0}{V(t_1)} (T_m - T_0) e^{-kt_1}
    \end{align*}
    $$

    Now, substituting gives:

    $$
    \begin{align*}
        T(t) &= T_m + \frac{R}{kV(t)} (T_C - T_m) \left(1 - e^{-k(t - t_1)} \right) + \frac{V(t_1)}{V(t)} e^{-k(t-t_1)} \left( \frac{R}{kV(t_1)}(T_H-T_m) \left( 1 - e^{-kt_1} \right) - \frac{V_0}{V(t_1)} (T_m - T_0) e^{-kt_1} \right)\\
        &= T_m + \frac{R}{kV(t)} (T_C - T_m) \left(1 - e^{-k(t - t_1)} \right) + \frac{R}{kV(t)}(T_H-T_m) \left(e^{-k(t-t_1)}  - e^{-kt} \right) -  \frac{V_0}{V(t)} (T_m - T_0) e^{-kt}
    \end{align*}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Final Result

    By combing the results, we see that we have found:

    $$
        T(t) = \begin{cases}
            \displaystyle T_m + \frac{R}{kV(t)} (T_C - T_m) \left(1 - e^{-k(t - t_1)} \right) + \frac{V(t_1)}{V(t)} e^{-k(t-t_1)} (T_1 - T_m) & \text{if } 0 \le t \le t_1 \\[1.5em]
            \displaystyle T_m + \frac{R}{kV(t)} (T_C - T_m) \left(1 - e^{-k(t - t_1)} \right) + \frac{R}{kV(t)}(T_H-T_m) \left(e^{-k(t-t_1)}  - e^{-kt} \right) -  \frac{V_0}{V(t)} (T_m - T_0) e^{-kt} & \text{if } t \gt t_1
        \end{cases}
    $$
    """
    )
    return


@app.cell
def _(R, T_in, T_m, k):
    def ode_cooling(t, y):
        V, T = y
        dVdt = R
        dTdt = R * (T_in(t) - T) / V - k * (T - T_m)
        return [dVdt, dTdt]
    return (ode_cooling,)


@app.cell
def _(
    config,
    dropdown_ode_method,
    mo,
    sider_C,
    sider_H,
    sider_R,
    sider_T0,
    sider_V0,
    sider_t1,
    sider_t_final,
):
    slider_k = mo.ui.slider(
        start=0.001,
        stop=1.0,
        step=0.001,
        value=config.k,
        label="Cooling Rate ($k$)",
        show_value=True,
    )

    slider_T_m = mo.ui.slider(
        start=0.0,
        stop=100.0,
        step=1.0,
        value=config.T_m,
        label="Ambient Temperature of Medium ($T_m$)",
        show_value=True,
    )

    cooling_controls = mo.vstack(
        [
            sider_R,
            sider_H,
            sider_C,
            sider_t1,
            sider_V0,
            sider_T0,
            sider_t_final,
            slider_k,
            slider_T_m,
            dropdown_ode_method,
        ]
    )
    return cooling_controls, slider_T_m, slider_k


@app.cell
def _(cooling_controls, slider_T_m, slider_k):
    k = slider_k.value
    T_m = slider_T_m.value


    cooling_controls
    return T_m, k


@app.cell
def _(T0, V0, ode_cooling, ode_method, plt, solve_ivp, t1, t_eval, t_final):
    sol_cooling = solve_ivp(
        ode_cooling, (0, t_final), [V0, T0], t_eval=t_eval, method=ode_method
    )

    _fig, _ax1 = plt.subplots(figsize=(8, 5))

    # Plot Temperature vs Time

    _ax1.plot(sol_cooling.t, sol_cooling.y[1])
    _ax1.axvline(t1, linestyle="--")
    _ax1.set_xlabel("Time")
    _ax1.set_ylabel("Temperature")
    _ax1.set_title("Temperature vs Time")
    _ax1.grid()

    _fig
    return


if __name__ == "__main__":
    app.run()
