import matplotlib.pyplot as plt

# # VDN (1,2,3,4,5,6,15)
# sights = [1, 2, 3, 4, 5, 6, 15]
# returns = [0.2898, 0.5586, 0.7099, 0.8891, 0.9291, 0.9596, 0.7133]

# VDN (1,2,3,4,5,6)
sights = [1, 2, 3, 4, 5, 6]
returns = [0.2260, 0.5676, 0.8419, 0.9236, 0.9402, 0.9261]


# Assume observation cost function is linear
def obs_cost_linear(sight):
    return sight


def obs_cost_quadratic(sight):
    return sight ** 2


cost_functions = [obs_cost_linear, obs_cost_quadratic]

# Print all Pareto fronts
# Each function's result is an image
# x-axis is return, y-axis is negative observation cost
# Show each point's sight text; connect neighboring points
plt.style.use('ggplot')
for cost_function in cost_functions:
    plt.figure()
    plt.title(cost_function.__name__)
    plt.xlabel('Return')
    plt.ylabel('Negative Observation Cost')
    for i in range(len(sights) - 1):
        plt.plot([returns[i], returns[i + 1]], [-cost_function(sights[i]), -cost_function(sights[i + 1])], c='gray')
    for i, sight in enumerate(sights):
        # Let the text right more in order not to overlap with the point
        plt.annotate(f'{sight}s', (returns[i], -cost_function(sight)), textcoords="offset points", xytext=(6, 6), ha='center')
    plt.scatter(returns, [-cost_function(sight) for sight in sights], s=100, c=sights, cmap='viridis', zorder=10)
    plt.colorbar()
    plt.show()
