import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp  # 更精确的积分方法
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 气体常数
R = 8.314  # J/(mol·K)

# 动力学微分方程
def d_alpha_dt(alpha, T, E, n, A):
    return A * np.exp(-E / (R * T)) * (1 - alpha)**n


# 使用 solve_ivp 进行更精确的积分（推荐）
def integrate_alpha(t, T, E, n, A, alpha0=0.0):
    def d_alpha_dt_wrapper(t, y):
        return [d_alpha_dt(y[0], T, E, n, A)]
    sol = solve_ivp(d_alpha_dt_wrapper, [t[0], t[-1]], [alpha0], t_eval=t, method='RK45', rtol=1e-6)
    return sol.y[0]


# 全局模型预测函数
def model_prediction(params, t_list, T_list, alpha0_list=None):
    E, n, A = params
    predictions = []
    for i in range(len(t_list)):
        t = t_list[i]
        T = T_list[i]
        alpha0 = alpha0_list[i] if alpha0_list is not None else 0.0
        pred = integrate_alpha(t, T, E, n, A, alpha0)
        predictions.append(pred)
    return np.concatenate(predictions)

# ========================
# 1. 输入三组实验数据
# ========================

# --- 55°C = 328.15 K ---
t_55 = np.array([
    1544.868, 9931.356, 21186.864, 38180.484, 50098.104, 61574.292,
    72609.12, 82099.08, 91809.72, 100637.568, 108803.34, 118072.584,
    125576.28, 134404.128, 142569.9, 150073.56, 159342.804, 167729.292,
    179426.196, 190461.024, 205026.984, 219151.548, 233938.188, 246076.524,
    257994.108, 275208.444, 299485.044, 326630.7, 341417.376, 360838.656,
    385335.936, 400564.008, 416454.156, 429475.248, 444924.0
])
alpha_55 = np.array([
    0.00998, 0.01774, 0.04435, 0.09093, 0.13972, 0.19073, 0.24284, 0.29496,
    0.34597, 0.39919, 0.45131, 0.49456, 0.54667, 0.59768, 0.64315, 0.69748,
    0.74738, 0.79173, 0.84718, 0.89153, 0.93478, 0.95806, 0.97248, 0.98246,
    0.98579, 0.98690, 0.99355, 0.99355, 0.99133, 0.99688, 0.99355, 0.99355,
    0.99355, 0.99688, 0.99688
])
T_55 = 328.15

# --- 60°C = 333.15 K ---
t_60 = np.array([
    0, 8827.848, 17435.016, 26483.58, 35311.428, 43256.484, 50539.464,
    58484.556, 64443.348, 70622.856, 75698.856, 81657.684, 87616.476,
    93795.984, 99092.7, 105934.284, 112334.472, 120279.564, 129548.808,
    139700.844, 152059.824, 162211.86, 174350.16, 186709.176, 198626.76,
    212972.04, 226655.208, 234600.3, 246297.204, 264835.692, 283815.612,
    302133.384, 318906.324, 338989.716, 358631.676, 375625.296, 394384.5,
    411819.516, 425723.4, 442717.02
])
alpha_60 = np.array([
    0.00776, 0.01885, 0.04325, 0.09093, 0.14083, 0.1874, 0.24395, 0.29829,
    0.34375, 0.39808, 0.44355, 0.49234, 0.54667, 0.59879, 0.64758, 0.70302,
    0.75625, 0.80837, 0.86492, 0.90817, 0.93921, 0.95585, 0.96804, 0.9747,
    0.97802, 0.97581, 0.97802, 0.98357, 0.98246, 0.98579, 0.98579, 0.98579,
    0.98911, 0.99022, 0.99355, 0.99133, 0.99133, 0.99466, 0.99466, 0.99688
])
T_60 = 333.15

# --- 65°C = 338.15 K ---
t_65 = np.array([
    0, 11034.828, 18979.884, 25600.788, 32000.976, 39063.276, 43918.596,
    49215.312, 54512.028, 59367.348, 64443.348, 68857.272, 73933.308,
    79230.024, 85188.816, 90926.928, 98210.908, 108582.624, 126238.356,
    146321.712, 167729.292, 186488.460, 207454.644, 233938.188, 257552.712,
    273663.576, 289112.292, 305002.440, 321113.304, 338769.000, 350245.224,
    372314.844, 387542.916, 402108.876, 429254.532, 441392.832
])
alpha_65 = np.array([
    0.01109, 0.04879, 0.09647, 0.14859, 0.1996, 0.25171, 0.2994, 0.35151,
    0.39808, 0.4502, 0.49788, 0.54667, 0.59879, 0.64647, 0.70081, 0.75071,
    0.79839, 0.8505, 0.90151, 0.92036, 0.93367, 0.93921, 0.94587, 0.9503,
    0.95696, 0.96139, 0.96472, 0.96804, 0.96915, 0.9747, 0.97802, 0.98024,
    0.98357, 0.98579, 0.98911, 0.99022
])
T_65 = 338.15

# ========================
# 🔍 数据筛选：只保留稳定阶段之后的数据
# ========================
threshold_55 = 100000  # s
threshold_60 = 100000  # s
threshold_65 = 100000  # s

mask_55 = t_55 > threshold_55
mask_60 = t_60 > threshold_60
mask_65 = t_65 > threshold_65

t_55_stable = t_55[mask_55]
alpha_55_stable = alpha_55[mask_55]
t_60_stable = t_60[mask_60]
alpha_60_stable = alpha_60[mask_60]
t_65_stable = t_65[mask_65]
alpha_65_stable = alpha_65[mask_65]

# 输出筛选后数据点数量
print(f"筛选后数据点数量:")
print(f"  55°C (> {threshold_55} s): {len(t_55_stable)} 个点")
print(f"  60°C (> {threshold_60} s): {len(t_60_stable)} 个点")
print(f"  65°C (> {threshold_65} s): {len(t_65_stable)} 个点")

# 如果某组没有数据，报错
if len(t_55_stable) == 0 or len(t_60_stable) == 0 or len(t_65_stable) == 0:
    raise ValueError("某组温度下无稳定阶段数据，请检查阈值设置！")

# 组织数据列表
t_list = [t_55_stable, t_60_stable, t_65_stable]
alpha_list = [alpha_55_stable, alpha_60_stable, alpha_65_stable]
T_list = [T_55, T_60, T_65]

# 设置初始 alpha（使用筛选后第一点）
alpha0_list = [alpha_55_stable[0], alpha_60_stable[0], alpha_65_stable[0]]

# 拼接真实数据
y_true = np.concatenate(alpha_list)

# ========================
# 初始猜测和边界
# ========================
initial_guess = [70000, 2, 1e4]  # E, n, A
bounds = ([0, 0, 0], [200000, 5.0, 1e15])

# ========================
# 定义残差函数
# ========================
def residuals(params):
    # print("Current params:", params)      #输出E、n、A的迭代过程
    pred = model_prediction(params, t_list, T_list, alpha0_list)
    return pred - y_true

# ========================
# 执行全局拟合
# ========================
result = least_squares(residuals, initial_guess, bounds=bounds, ftol=1e-8, xtol=1e-8, method='trf')
E_fit, n_fit, A_fit = result.x


print(f"\n✅ 稳定阶段全局拟合完成！")
print(f"活化能 E = {E_fit:.2f} J/mol")
print(f"反应级数 n = {n_fit:.2f}")
print(f"指前因子 A = {A_fit:.2e} 1/s")
print(f"拟合残差平方和 (cost): {result.cost:.6f}")

# ========================
# 计算 R²（全局）
# ========================
y_pred = model_prediction([E_fit, n_fit, A_fit], t_list, T_list, alpha0_list)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² (global on stable phase) = {r2:.6f}")

# ========================
# 分组 R²
# ========================
print("\n🧪 分组 R²（稳定阶段）:")
labels = ['55°C (328.15 K)', '60°C (333.15 K)', '65°C (338.15 K)']
for i in range(3):
    r2_i = r2_score(alpha_list[i], integrate_alpha(t_list[i], T_list[i], E_fit, n_fit, A_fit, alpha0_list[i]))
    print(f"  {labels[i]}: R² = {r2_i:.6f}")


# ========================
# 🔍 输出 65°C 的实验 vs 拟合固化度数据（稳定阶段）
# ========================
print("\n🔬 65°C (338.15 K) 实验数据 vs 拟合数据对比（稳定阶段）:")
print("时间(s)\t\t实验alpha\t拟合alpha\t绝对误差\t\t相对误差 (%)")
print("-" * 80)

# 提取 65°C 数据
i = 2  # 因为 65°C 是第 3 组（索引为 2）
t_65_stable = t_list[i]
alpha_65_exp = alpha_list[i]
T_65_val = T_list[i]
alpha0_65 = alpha0_list[i]

# 计算拟合值
alpha_65_fit = integrate_alpha(t_65_stable, T_65_val, E_fit, n_fit, A_fit, alpha0_65)

# 逐点输出
for t, exp, fit in zip(t_65_stable, alpha_65_exp, alpha_65_fit):
    abs_error = abs(exp - fit)
    rel_error = (abs_error / exp * 100) if exp != 0 else abs_error * 100
    print(f"{t:.1f}\t\t{exp:.6f}\t{fit:.6f}\t{abs_error:.6f}\t\t{rel_error:8.4f}%")



# ========================
# 绘图
# ========================
plt.figure(figsize=(10, 7))
colors = ['tab:blue', 'tab:orange', 'tab:green']

for i in range(3):
    t_full = [t_55, t_60, t_65][i]         # 原始完整时间
    alpha_full = [alpha_55, alpha_60, alpha_65][i]
    t_stable = t_list[i]
    alpha_stable = alpha_list[i]
    sim_stable = integrate_alpha(t_stable, T_list[i], E_fit, n_fit, A_fit, alpha0_list[i])

    # 绘制完整实验数据（灰点）
    plt.plot(t_full, alpha_full, 'o', color=colors[i], label=f'Exp (stable) - {labels[i]}', markersize=6)
    # # 高亮稳定阶段数据（彩色）
    # plt.plot(t_stable, alpha_stable, 'o', color=colors[i], label=f'Exp (stable) - {labels[i]}', markersize=6)
    # 拟合曲线
    plt.plot(t_stable, sim_stable, '-', color=colors[i], linewidth=2, label=f'Fit - {labels[i]}')

plt.xlabel('Time (s)')
plt.ylabel('Conversion alpha')
plt.title(r'Kinetic Fitting on Stable Phase Only')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# ========================
# 参数总结
# ========================
print("\n📊 拟合参数总结（仅稳定阶段）:")
print(f"E  = {E_fit:8.2f} J/mol")
print(f"n  = {n_fit:8.2f}")
print(f"A  = {A_fit:8.2e} 1/s")
print(f"R² = {r2:8.6f}")




# # 输出拟合后的数据（含实验值对比）
# print("\n📈 拟合后的固化度 (alpha) 预测数据（实验值 vs 拟合值）:")

# for i in range(3):
#     t_stable = t_list[i]
#     alpha_exp = alpha_list[i]  # 实验 alpha
#     alpha_pred = integrate_alpha(t_stable, T_list[i], E_fit, n_fit, A_fit, alpha0_list[i])  # 拟合 alpha
#     label = labels[i]

#     print(f"\n--- {label} ---")
#     print("时间(s)\t\t实验alpha\t拟合alpha\t偏差 (exp - fit)")
#     print("-" * 60)
#     for t, exp, pred in zip(t_stable, alpha_exp, alpha_pred):
#         diff = abs(exp - pred)/exp*100
#         print(f"{t:.1f}\t\t{exp:.6f}\t{pred:.6f}\t{diff:+.6f}%")
