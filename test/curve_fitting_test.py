import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 模型定义
def d_alpha_dt(E, n, alpha, T, A):
    R = 8.314
    return A * np.exp(-E / (R * T)) * (1 - alpha)**n

def alpha_model(t, E, n, T, A, alpha0=0):
    dt = t[1] - t[0]  # 时间步长
    alpha = [alpha0]       # 初始 alpha = 0
    for i in range(1, len(t)):
        d_alpha = d_alpha_dt(E, n, alpha[-1], T, A) * dt
        alpha.append(alpha[-1] + d_alpha)
    return np.array(alpha)

# 原始数据
t_data = np.array([0, 3.06523, 5.27219, 7.11133, 8.88916, 10.85091, 12.19961, 13.67092, 15.14223,
                   16.49093, 17.90093, 19.12702, 20.53703, 22.00834, 23.66356, 25.25748, 27.28053,
                   30.16184, 35.06621, 40.64492, 46.59147, 51.80235, 57.62629, 64.98283, 71.54242,
                   76.01766, 80.30897, 84.7229, 89.19814, 94.1025, 97.29034, 103.42079, 107.65081,
                   111.69691, 119.23737, 122.60912])
alpha_data = np.array([0.01109, 0.04879, 0.09647, 0.14859, 0.1996, 0.25171, 0.2994, 0.35151, 0.39808,
                       0.4502, 0.49788, 0.54667, 0.59879, 0.64647, 0.70081, 0.75071, 0.79839, 0.8505,
                       0.90151, 0.92036, 0.93367, 0.93921, 0.94587, 0.9503, 0.95696, 0.96139, 0.96472,
                       0.96804, 0.96915, 0.9747, 0.97802, 0.98024, 0.98357, 0.98579, 0.98911, 0.99022])

T = 338  # 温度 (K)


# 筛选 t >= 40 的数据（筛选出温度均匀时的数据）
mask = t_data >= 40
t_data_filtered = t_data[mask]
alpha_data_filtered = alpha_data[mask]


# 初始猜测值：E, n, A
initial_guess = [70000, 0.88, 1e6, alpha_data_filtered[0]]  # E (J/mol), n (dimensionless), A (1/s)


# 进行拟合
params, covariance = curve_fit(
    lambda t, E, n, A, alpha0: alpha_model(t, E, n, T, A, alpha0),
    t_data_filtered,
    alpha_data_filtered,
    p0=initial_guess,
    bounds=(0, [np.inf, 10, 1e12, 1])  # 参数E，n，A，alpha0的下界统一为0，上界分别为np.inf, 10, 1e12, 1
)


# 输出拟合结果
E_fit, n_fit, A_fit, alpha0_fit = params
print(f"Fitted parameters: E = {E_fit:.2f}, n = {n_fit:.2f}, A = {A_fit:.2e}, alpha0 = {alpha0_fit:.4f}")
print(f"Uncertainties:     ±{np.sqrt(covariance[0,0]):.2f}, ±{np.sqrt(covariance[1,1]):.2f}, ±{np.sqrt(covariance[2,2]):.2e}")


# 使用拟合参数计算拟合曲线（完整时间范围，用于绘图）
alpha_fit_full = alpha_model(t_data, E_fit, n_fit, T, A_fit, alpha0_fit)


# 只在 t >= 40 的时间点上计算拟合值，用于 R²
alpha_fit_filtered = alpha_model(t_data_filtered, E_fit, n_fit, T, A_fit, alpha0_fit)


# 确保两者长度一致
assert len(alpha_fit_filtered) == len(alpha_data_filtered), "Length mismatch between predicted and true values"






# 用sklearn计算 t ≥ 40 的 R²
r2 = r2_score(alpha_data_filtered, alpha_fit_filtered)
print(f"R² (on filtered data) = {r2:.6f}")

# # 用sklearn计算完整数据的 R²
# r2 = r2_score(alpha_data, alpha_fit_full)
# print(f"R² (on full data) = {r2:.6f}")

# # 不使用sklearn，原代码计算 t ≥ 40 的R²
# ss_res = np.sum((alpha_data_filtered - alpha_fit_filtered) ** 2)
# ss_tot = np.sum((alpha_data_filtered - np.mean(alpha_data_filtered)) ** 2)
# r_squared = 1 - (ss_res / ss_tot)
# print("len_alpha_data_filtered:", len(alpha_data_filtered))
# print("len_alpha_pred_filtered:", len(alpha_fit_filtered))



# 输出完整的实验数据以及拟合数据
print("alpha_data:", alpha_data)
print("alpha_fit_full:", alpha_fit_full)

# # 输出 t ≥ 40 的原始实验数据和预测拟合数据
# print("alpha_data_filtered:", alpha_data_filtered)
# print("alpha_pred_filtered:", alpha_fit_filtered)
# print("alpha_data_filtered_avg:", np.mean(alpha_data_filtered))

# # 不使用sklearn，原代码计算R²时输出残差平方和（ss_res）、总平方和（ss_tot）
# print("ss_res:", ss_res)
# print("ss_tot:", ss_tot)
# print("R-squared:", r_squared)






# 绘图：显示所有数据，但拟合只基于 t ≥ 40 的数据
plt.figure(figsize=(10, 6))
plt.plot(t_data, alpha_data, 'o', label='Experimental data')
plt.plot(t_data, alpha_fit_full, '-', label=f'Fitted curve')
plt.axvline(40, color='r', linestyle='--', label='t = 40s')
plt.xlabel('Time (s)')
plt.ylabel('Alpha')
plt.title('Kinetic Model Fitting (Fitted on t ≥ 40s)')
plt.legend()
plt.grid(True)
plt.text(0.5, 0.95, f'E = {E_fit:.2f}\nn = {n_fit:.2f}\nA = {A_fit:.2e}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.tight_layout()
plt.show()
