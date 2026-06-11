# GRIN Optics. 

## Solution Pattern. 

For every single task, you'll have to implement the simulation as well as answer the question given in the task. **Provide all the answers, plots, and any extra input in a markdown file.**

## compare_utility.py usage.

To compare and benchmark your integrators using `compare_utility.py`, define them according to the signatures below and run the benchmark.

### Integrator Signatures

- **Fixed-Step Integrator**:
  ```python
  def my_integrator(state, deriv_fn, dt):
      # state    : np.ndarray([x, y, px, py])
      # deriv_fn : callable  state -> d(state)/ds
      # dt       : arc-length step size (float)
      return new_state
  ```


### Usage Example

```python
from compare_utility import print_results, plot_trajectories, get_compute_accuracy

# 1. Run benchmark (prints results to terminal and returns results list)
results = print_results(
    my_euler, my_rk4, my_adaptive,
    dt     = 2.0,
    x_max  = 500.0,
    labels = ['Euler', 'RK4', 'Adaptive RK4'],
)

# 2. Generate trajectory plot with deviation analysis
plot_trajectories(results)

# 3. Extract evaluation data (name, evals, rmse) for plotting
data = get_compute_accuracy(results)
```


## Resources.
[Grin Optics Visual Introduction](https://www.youtube.com/watch?v=XQj97dva6ss)

[Grin Optics Wikipedia](https://en.wikipedia.org/wiki/Gradient-index_optics)
