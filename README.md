# SEDS Celestia Simulations Core Assignment

Welcome! If you're reading this, you’re looking to join the Simulations vertical of SEDS Celestia. We’re a bunch of developers, math nerds, and physics enthusiasts who love building complex systems from scratch. In the past, we've mostly focused on hardcore physics based systems, from building a fluid simulation to a black hole simulations. 

Moving forward, though, we’re branching out way beyond just physics based simulations into verious other domains. If you like seeing complex math come to life visually through code using tools like GPU acceleration, compute shaders, you're in the right place.

## Monte Carlo Simulations

Before diving into the code, your task for this assignment involves a fundamental tool in a simulation engineer's toolkit: the **Monte Carlo simulation**.
Instead of solving a massive, terrifying analytical equation directly, a Monte Carlo simulation uses brute-force randomness to find an answer. Think of it like trying to find the area of an oddly shaped puddle by throwing thousands of random pebbles at the ground and counting how many land in the water versus how many land on the dry grass. By using a random number generator to run a system thousands or millions of times, we can accurately predict probabilities, model chaotic systems, and solve deterministic problems that are otherwise computationally nightmarish.
## The Hardcore Theory
*(TODO)*
## The Simulation Loop (Skeleton Code)
No matter how complex a simulation gets, almost all of them boil down to a central clock loop: you initialize the state, update the physics/logic based on time steps, render or log the data, and repeat.
Here is a basic skeleton structure to get you thinking about how to frame your assignment code:
```python
import numpy as np

def initialize_system():
    # TODO: Set up your initial conditions, parameters, and data structures
    print("Initializing simulation state...")
    return {"step": 0, "data": []}

def update_state(current_state, dt):
    # TODO: Apply your Monte Carlo math, physics steps, or random sampling here
    # This is where the core logic lives
    next_state = current_state.copy()
    next_state["step"] += 1
    return next_state

def run_simulation(total_steps, dt):
    state = initialize_system()
    
    # The Main Simulation Loop
    for _ in range(total_steps):
        state = update_state(state, dt)
        
        # Optional: Add logic to break early if convergence or a specific condition is met
        
    print(f"Simulation finished after {state['step']} steps.")
    return state

if __name__ == "__main__":
    # Quick test run configuration
    DT = 0.01
    TOTAL_STEPS = 10000
    
    final_results = run_simulation(TOTAL_STEPS, DT)

```
