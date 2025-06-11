import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 设置 lambda 值
lambda_param = 1.0

def exp_cdf(x, lam=1.0):
    return 1 - np.exp(-lam * x)

def exp_cdf_with_epsilon(x, lam=1.0, alpha=0.1, beta=1.0):
    x = np.asarray(x)
    base_cdf = 1 - np.exp(-lam * x)
    epsilon_max = alpha / (1 + beta * x)
    epsilon = np.random.rand(*x.shape) * epsilon_max
    result = base_cdf + epsilon
    return np.minimum(result, 1.0)

# 主图数据
x = np.linspace(0, 15, 500)
y = exp_cdf(x, lambda_param)

# 随机点
np.random.seed(0)
x_points_1 = np.random.uniform(0, 3, 5)
x_points_2 = np.random.uniform(4, 6, 5)
x_points = np.concatenate([x_points_1, x_points_2])
y_points = exp_cdf_with_epsilon(x_points, lambda_param)

# 主图
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x, y, label=r'$R(r) = 1 - e^{-\lambda r}$', color='green')
ax.scatter(x_points, y_points, color='red', label='Observation points')
# ax.set_title('The situation of relative changes of indicators on a global scale (exponential distribution function)')
ax.set_xlabel('r')
ax.set_ylabel('R(r)')
ax.set_xlim(0, 8)
ax.set_ylim(-0.1, 1.2)
ax.legend()
ax.grid(False)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)

# 插图 1：放大 [0,3]
axins1 = inset_axes(ax, width="30%", height="30%", loc='lower left')
x_zoom1 = np.linspace(0, 3, 300)
y_zoom1 = exp_cdf(x_zoom1, lambda_param)
axins1.plot(x_zoom1, y_zoom1, color='green')
axins1.scatter(x_points_1, exp_cdf_with_epsilon(x_points_1, lambda_param), color='red')
axins1.set_xlim(1, 3)
axins1.set_ylim(0.6, 0.95)
axins1.set_xticks([])
axins1.set_yticks([])
axins1.set_xticklabels([])
axins1.set_yticklabels([])
mark_inset(ax, axins1, loc1=2, loc2=1, fc="none", ec="gray", lw=1)

# 插图 2：放大 [4,6]
axins2 = inset_axes(ax, width="30%", height="30%", loc='lower right')
x_zoom2 = np.linspace(4, 6, 300)
y_zoom2 = exp_cdf(x_zoom2, lambda_param)
axins2.plot(x_zoom2, y_zoom2, color='green')
axins2.scatter(x_points_2, exp_cdf_with_epsilon(x_points_2, lambda_param), color='red')
axins2.set_xlim(4, 6)
axins2.set_ylim(0.95, 1.02)
axins2.set_xticks([])
axins2.set_yticks([])
axins2.set_xticklabels([])
axins2.set_yticklabels([])
mark_inset(ax, axins2, loc1=2, loc2=1, fc="none", ec="gray", lw=1)

# 保存图像
plt.savefig('draw/pics/exponential_cdf_with_insets.png', dpi=300)
plt.show()