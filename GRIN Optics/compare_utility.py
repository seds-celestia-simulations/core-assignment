"""
compare_utility.py
──────────────────
Benchmark utility for comparing numerical integrators on the GRIN optics
ray problem (Level I — linear gradient medium  n(y) = n0 + alpha*y).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INTEGRATOR SIGNATURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Fixed-step integrators must return new_state:

     def my_integrator(state, deriv_fn, dt):
         # state    : np.ndarray([x, y, px, py])
         # deriv_fn : callable  state -> d(state)/ds
         # dt       : arc-length step size
         ...
         return new_state

 Adaptive integrators must return (new_state, new_dt):

     def my_adaptive(state, deriv_fn, dt):
         ...
         return new_state, new_dt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     from compare_utility import print_results, plot_trajectories, get_compute_accuracy

     # 1. run benchmark — prints to terminal, returns results object
     results = print_results(
         my_euler, my_rk4, my_adaptive,
         dt     = 2.0,
         x_max  = 500.0,
         labels = ['Euler', 'RK4', 'Adaptive RK4'],
     )

     # 2. trajectory plot with deviation analysis
     plot_trajectories(results)

     # 3. get (name, evals, rmse) for your own scatter plot
     data = get_compute_accuracy(results)
     # data = [{'name': 'Euler', 'n_evals': 313, 'rmse': 6.95e-1}, ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TERMINAL OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     Euler           :  RMSE = 6.9555e-01 px   Evals = 313
     RK4             :  RMSE = 6.3419e-04 px   Evals = 1256
     Adaptive RK4    :  RMSE = 2.5488e-03 px   Evals = 1896

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ANALYTIC SOLUTION  (verified to ~1e-11 against fine arc-length RK4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     y(x) = ( p * cosh(alpha*x/p + phi0) - n0 ) / alpha

     p    = n(y0) * cos(theta0)    conserved horizontal optical momentum
     phi0 = arctanh(sin(theta0))   encodes launch angle and direction
"""

import numpy as np
import matplotlib.pyplot as plt


# ── default medium parameters ──────────────────────────────────────────────────
N0    = 1.5
ALPHA = 0.003


# ── analytic solution ──────────────────────────────────────────────────────────

def analytic_y(x, y0=0.0, theta0_deg=0.0, n0=N0, alpha=ALPHA):
    """
    Exact catenary trajectory for n(y) = n0 + alpha*y.
    x may be a scalar or a 1-D numpy array.
    """
    th   = np.radians(theta0_deg)
    p    = (n0 + alpha * y0) * np.cos(th)
    phi0 = np.arctanh(np.sin(th))
    return (p * np.cosh(alpha * x / p + phi0) - n0) / alpha


# ── ODE right-hand side ────────────────────────────────────────────────────────

def make_deriv(n0=N0, alpha=ALPHA):
    """
    Ray equation for n(y) = n0 + alpha*y, arc-length parameterised.
        dx/ds  = px / n(y)
        dy/ds  = py / n(y)
        dpx/ds = 0          (n has no x-dependence → px is conserved)
        dpy/ds = alpha       (= dn/dy)
    """
    def f(state):
        _, y, px, py = state
        nv = max(1e-3, n0 + alpha * y)
        return np.array([px/nv, py/nv, 0.0, alpha])
    return f


# ── call counter (internal) ────────────────────────────────────────────────────

class _CountedFn:
    """
    Wraps deriv_fn and counts every call to it.
    Passed to the integrator in place of the raw deriv_fn, so that
    Euler (1 call/step), RK4 (4 calls/step), and adaptive RK4
    (≥12 calls/attempted step, including rejected steps) are all
    measured on the same footing without any manual bookkeeping.
    """
    def __init__(self, fn):
        self._fn   = fn
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._fn(*args, **kwargs)


# ── core benchmark ─────────────────────────────────────────────────────────────

def benchmark(
    *integrators,
    dt         = 0.5,
    y0         = 0.0,
    theta0_deg = 0.0,
    x_max      = 400.0,
    n_eval     = 200,
    n0         = N0,
    alpha      = ALPHA,
    max_steps  = 40000,
    labels     = None,
):
    """
    Run each integrator and measure accuracy and compute cost.

    Parameters
    ----------
    *integrators  : 1–3 callables  (state, deriv_fn, dt) -> state | (state, dt)
    dt            : arc-length step size; seed for adaptive integrators
    y0            : launch height (pixels, centred coords)
    theta0_deg    : launch angle from horizontal (degrees)
    x_max         : horizontal domain for RMSE evaluation (pixels)
    n_eval        : number of x-sample points used for RMSE
    n0, alpha     : medium parameters
    max_steps     : hard cap on integration steps
    labels        : display names for each integrator (list of str)

    Returns
    -------
    list of dict — one entry per integrator:
        'name'      : str
        'n_evals'   : int    total derivative evaluations  (compute axis)
        'rmse'      : float  RMS position error vs analytic catenary  (px)
        'mom_error' : float  max fractional drift of conserved px
        'trail_x'   : array  numerical x-positions along the ray
        'trail_y'   : array  numerical y-positions along the ray
        'true_x'    : array  x-grid used for RMSE
        'true_y'    : array  analytic y at that grid
    """
    if not (1 <= len(integrators) <= 3):
        raise ValueError("Pass between 1 and 3 integrators.")

    n_init = n0 + alpha * y0
    th     = np.radians(theta0_deg)
    s0     = np.array([0.0, y0, n_init*np.cos(th), n_init*np.sin(th)])
    px0    = s0[2]

    x_grid = np.linspace(0.0, x_max, n_eval)
    y_true = analytic_y(x_grid, y0=y0, theta0_deg=theta0_deg, n0=n0, alpha=alpha)
    deriv  = make_deriv(n0=n0, alpha=alpha)

    results = []

    for i, integ in enumerate(integrators):
        name = (labels[i] if labels and i < len(labels)
                else getattr(integ, '__name__', f'integrator_{i}'))

        counter  = _CountedFn(deriv)
        state    = s0.copy()
        cur_dt   = float(dt)
        trail_x  = [state[0]]
        trail_y  = [state[1]]
        px_log   = [state[2]]
        adaptive = None

        for _ in range(max_steps):
            if state[0] >= x_max * 1.05:
                break

            ret = integ(state, counter, cur_dt)

            if adaptive is None:
                adaptive = isinstance(ret, tuple)

            if adaptive:
                state, cur_dt = ret
            else:
                state = ret

            trail_x.append(state[0])
            trail_y.append(state[1])
            px_log.append(state[2])

        trail_x = np.array(trail_x)
        trail_y = np.array(trail_y)
        px_log  = np.array(px_log)

        # RMSE over the x-domain the ray has covered
        mask = x_grid <= trail_x[-1]
        if mask.sum() >= 2:
            y_num = np.interp(x_grid[mask], trail_x, trail_y)
            rmse  = float(np.sqrt(np.mean((y_num - y_true[mask])**2)))
        else:
            rmse  = np.nan

        # fractional drift of the conserved quantity px = n*cos(theta)
        mom_err = float(np.max(np.abs(px_log - px0)) / (abs(px0) + 1e-12))

        results.append({
            'name'      : name,
            'n_evals'   : counter.count,
            'rmse'      : rmse,
            'mom_error' : mom_err,
            'trail_x'   : trail_x,
            'trail_y'   : trail_y,
            'true_x'    : x_grid,
            'true_y'    : y_true,
        })

    return results


# ── terminal output ────────────────────────────────────────────────────────────

def print_results(*integrators, **kwargs):
    """
    Run benchmark() and print one line per integrator to the terminal:

        {Name}  :  RMSE = {value} px   Evals = {value}

    Returns the full results list so you can pass it directly to
    plot_trajectories() or get_compute_accuracy().
    """
    results = benchmark(*integrators, **kwargs)
    print()
    for r in results:
        rmse_str = f"{r['rmse']:.4e}" if not np.isnan(r['rmse']) else '     nan'
        print(f"  {r['name']:<16}:  RMSE = {rmse_str} px   Evals = {r['n_evals']}")
    print()
    return results


# ── extract values for scatter plot ───────────────────────────────────────────

def get_compute_accuracy(results):
    """
    Extract name, compute cost, and accuracy from benchmark() results.
    Use these values to build your own scatter plot.

    Parameters
    ----------
    results : list of dict returned by benchmark() or print_results()

    Returns
    -------
    list of dict:
        [
            {'name': 'Euler',  'n_evals': 313,  'rmse': 6.95e-1},
            {'name': 'RK4',    'n_evals': 1256, 'rmse': 6.34e-4},
            ...
        ]

    Example scatter plot
    --------------------
        import matplotlib.pyplot as plt
        data = get_compute_accuracy(results)
        for d in data:
            plt.scatter(d['n_evals'], d['rmse'], label=d['name'], s=80)
        plt.xscale('log');  plt.yscale('log')
        plt.xlabel('Derivative evaluations');  plt.ylabel('RMSE (px)')
        plt.legend();  plt.tight_layout();  plt.show()
    """
    return [{'name': r['name'], 'n_evals': r['n_evals'], 'rmse': r['rmse']}
            for r in results]


# ── trajectory plot ────────────────────────────────────────────────────────────

_COLORS = ['#e05c5c', '#5c8fe0', '#5cc46c']

def plot_trajectories(results, title='Ray Trajectories'):
    """
    Two-panel figure from benchmark() / print_results() output.

    Top panel   : true analytic trajectory (black) vs numerical trails (dashed).
    Bottom panel: residual y_num(x) - y_true(x) — the signed offset from the
                  exact path at each x position.

    Includes a zoom inset on the top panel centred on the region of maximum
    deviation. Adjust the three variables below to control the zoom.
    """
    # ── zoom controls ─────────────────────────────────────────────────────────
    ZOOM_X_WIDTH_PCT = 0.08    # zoom window width as fraction of total x-domain
    ZOOM_Y_PADDING   = 2.2     # y half-range = peak_deviation × this value
    ZOOM_INSET_RECT  = [0.55, 0.04, 0.42, 0.40]  # [left, bottom, w, h] axes fraction
    # ─────────────────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={'hspace': 0.06, 'height_ratios': [2, 1]},
    )

    tx = results[0]['true_x']
    ty = results[0]['true_y']
    ax1.plot(tx, ty, 'k-', linewidth=1.6, label='analytic', zorder=5)

    for r, col in zip(results, _COLORS):
        label = f"{r['name']}  (RMSE = {r['rmse']:.3g} px)"
        ax1.plot(r['trail_x'], r['trail_y'], '--',
                 color=col, linewidth=1.2, alpha=0.85, label=label)
        mask = tx <= r['trail_x'][-1]
        if mask.sum() >= 2:
            y_i = np.interp(tx[mask], r['trail_x'], r['trail_y'])
            ax2.plot(tx[mask], y_i - ty[mask], '-',
                     color=col, linewidth=1.2, alpha=0.85, label=r['name'])

    # ── zoom inset ────────────────────────────────────────────────────────────
    _worst   = max(results, key=lambda r: r['rmse'] if not np.isnan(r['rmse']) else 0)
    _y_num   = np.interp(tx, _worst['trail_x'], _worst['trail_y'])
    _abs_dev = np.abs(_y_num - ty)
    _ic      = np.argmax(_abs_dev)
    _xc      = tx[_ic]

    _xspan = (tx[-1] - tx[0]) * ZOOM_X_WIDTH_PCT
    _xl    = max(tx[0],  _xc - _xspan / 2)
    _xr    = min(tx[-1], _xc + _xspan / 2)
    _xmask = (tx >= _xl) & (tx <= _xr)

    _dev_here = _abs_dev[_xmask].max()
    _yc       = ty[_xmask].mean()
    _ypad     = max(_dev_here * ZOOM_Y_PADDING, 0.02)

    axin = ax1.inset_axes(ZOOM_INSET_RECT)
    axin.plot(tx[_xmask], ty[_xmask], 'k-', linewidth=1.5)
    for r, col in zip(results, _COLORS):
        _yi = np.interp(tx[_xmask], r['trail_x'], r['trail_y'])
        axin.plot(tx[_xmask], _yi, '--', color=col, linewidth=1.1)
    axin.set_xlim(_xl, _xr)
    axin.set_ylim(_yc - _ypad, _yc + _ypad)
    axin.tick_params(labelsize=7)
    axin.set_title(f'zoomed  (x ≈ {_xc:.0f},  ±{_ypad:.3f} px)', fontsize=7)
    axin.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)
    ax1.indicate_inset_zoom(axin, edgecolor='gray', alpha=0.55, linewidth=0.8)
    # ─────────────────────────────────────────────────────────────────────────

    ax1.set_ylabel('y  (px)', fontsize=10)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, linestyle='--', linewidth=0.4, alpha=0.45)
    ax1.set_title(title, fontsize=12)

    ax2.axhline(0, color='k', linewidth=0.8, linestyle=':')
    ax2.set_xlabel('x  (px)', fontsize=10)
    ax2.set_ylabel('offset  (px)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', linewidth=0.4, alpha=0.45)

    fig.tight_layout()
    return fig


# ── example (fill in your own integrators) ────────────────────────────────────

if __name__ == '__main__':

    # Replace my_euler, my_rk4, my_adaptive with your implementations.
    #
    # results = print_results(
    #     my_euler, my_rk4, my_adaptive,
    #     dt     = 2.0,
    #     x_max  = 500.0,
    #     labels = ['Euler', 'RK4', 'Adaptive RK4'],
    # )
    #
    # plot_trajectories(results)
    # plt.show()
    #
    # -- scatter plot (implement yourself) ------------------------------------
    # import matplotlib.pyplot as plt
    # data = get_compute_accuracy(results)
    # for d in data:
    #     plt.scatter(d['n_evals'], d['rmse'], label=d['name'], s=80)
    # plt.xscale('log');  plt.yscale('log')
    # plt.xlabel('Derivative evaluations');  plt.ylabel('RMSE (px)')
    # plt.legend();  plt.tight_layout();  plt.show()

    print("Implement your integrators and uncomment the example above.")
