import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# 该测试文件测试的是只输入40s之后的数据计算出的R²,发现只有40s之后的数据的话R²计算正常



# 模型定义
def d_alpha_dt(E, n, alpha, T, A):
    R = 8.314
    return A * np.exp(-E / (R * T)) * (1 - alpha)**n

def alpha_model(t, E, n, T, A):
    dt = t[1] - t[0]  # 时间步长
    alpha = [0.92226]       # 初始 alpha
    for i in range(1, len(t)):
        d_alpha = d_alpha_dt(E, n, alpha[-1], T, A) * dt
        alpha.append(alpha[-1] + d_alpha)
    return np.array(alpha)

# 原始数据
t_data = np.array([40.64492, 46.59147, 51.80235, 57.62629, 64.98283, 71.54242,
                   76.01766, 80.30897, 84.7229, 89.19814, 94.1025, 97.29034, 103.42079, 107.65081,
                   111.69691, 119.23737, 122.60912])
alpha_data = np.array([0.92036,
0.93367,
0.93921,
0.94587,
0.9503,
0.95696,
0.96139,
0.96472,
0.96804,
0.96915,
0.9747,
0.97802,
0.98024,
0.98357,
0.98579,
0.98911,
0.99022
])

T = 338  # 温度 (K)

# # 筛选 t >= 40 的数据（筛选出温度均匀时的数据）
# mask = t_data >= 40
# t_data_filtered = t_data[mask]
# alpha_data_filtered = alpha_data[mask]

# 初始猜测值：E, n, A
initial_guess = [66968, 0.92, 3.22e8]  # E (J/mol), n (dimensionless), A (1/s)

# 进行拟合
params, covariance = curve_fit(
    lambda t, E, n, A: alpha_model(t, E, n, T, A),
    t_data,
    alpha_data,
    p0=initial_guess,
    bounds=(0, [np.inf, 10, 1e12])  # 参数边界：E > 0, n > 0, A < 1e12
)

# 输出拟合结果
E_fit, n_fit, A_fit = params
print(f"Fitted parameters: E = {E_fit:.2f}, n = {n_fit:.2f}, A = {A_fit:.2e}")
print(f"Uncertainties:     ±{np.sqrt(covariance[0,0]):.2f}, ±{np.sqrt(covariance[1,1]):.2f}, ±{np.sqrt(covariance[2,2]):.2e}")

# 使用拟合参数计算拟合曲线（完整时间范围，用于绘图）
alpha_fit_full = alpha_model(t_data, E_fit, n_fit, T, A_fit)


# # 只在 t >= 40 的时间点上计算拟合值，用于 R²
# alpha_fit_filtered = alpha_model(t_data_filtered, E_fit, n_fit, T, A_fit)


# # 确保两者长度一致
# assert len(alpha_fit_filtered) == len(alpha_data_filtered), "Length mismatch between predicted and true values"


# # 计算 t ≥ 40 的 R²
# r2 = r2_score(alpha_data_filtered, alpha_fit_filtered)
# print(f"R² (on filtered data) = {r2:.6f}")
# 计算完整数据的 R²
r2 = r2_score(alpha_data, alpha_fit_full)
print(f"R² (on full data) = {r2:.6f}")


# 绘图：显示所有数据，但拟合只基于 t ≥ 40 的数据
plt.figure(figsize=(10, 6))
plt.plot(t_data, alpha_data, 'o', label='Experimental data')
plt.plot(t_data, alpha_fit_full, '-', label=f'Fitted curve')
# plt.axvline(40, color='r', linestyle='--', label='t = 40s')
plt.xlabel('Time (s)')
plt.ylabel('Alpha')
plt.title('Kinetic Model Fitting (Fitted on t ≥ 40s)')
plt.legend()
plt.grid(True)
plt.text(0.5, 0.95, f'R² = {r2:.4f}\nE = {E_fit:.2f}\nn = {n_fit:.2f}\nA = {A_fit:.2e}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.show()
