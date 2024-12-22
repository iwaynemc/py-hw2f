from hw2f import HullWhite2Factor
from yield_curve import YieldCurve
import numpy as np
import matplotlib.pyplot as plt

# construct yield curve
yield_curve = YieldCurve(np.array([1, 5, 10, 20, 30]), np.array([0.04211, 0.03809, 0.03764, 0.03791, 0.03603]))

# read in swaption vol surface
swaption_vols = np.loadtxt("USD_Swaption_ATM_IV.csv", delimiter=",", dtype=float)

# define option maturities and swap tenors
option_maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 4, 5])
swap_tenors = np.array([1, 2, 3, 4, 5, 7, 10])

# construct a HW2F model
hw2f = HullWhite2Factor(yield_curve, swaption_vols, option_maturities, swap_tenors)
params = hw2f.calibrate()
print(params)
# plt.plot(yield_curve.get_time, yield_curve.get_rate)
# plt.show()

# Generate and plot interest rate paths
num_paths = 500
num_steps = 600
rates = hw2f.generate_paths(params, num_paths, num_steps)

# Create time points for x-axis
time_points = np.linspace(0, yield_curve.get_time[-1], num_steps)

# Plot paths
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, rates[i,:], alpha=0.1, color='blue')
plt.plot(yield_curve.get_time, yield_curve.get_rate, 'r--', linewidth=2, label='Initial Curve')
plt.xlabel('Time (years)')
plt.ylabel('Interest Rate')
plt.title(f'Hull-White 2F Model: {num_paths} Interest Rate Paths')
plt.legend()
plt.grid(True)
plt.show()
