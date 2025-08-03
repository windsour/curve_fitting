import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 气体常数
R = 8.314  # J/(mol·K)

# 动力学微分方程（带保护）
def d_alpha_dt(alpha, T, E, n, A):
    if alpha >= 1.0:
        return 0.0  # 反应完成，速率为0
    term = 1.0 - alpha
    if term < 0:
        term = 0.0  # 防止浮点误差导致负数
    try:
        return A * np.exp(-E / (R * T)) * (term)**n
    except (OverflowError, ValueError):
        return 0.0  # 安全兜底

# 使用 solve_ivp 进行积分（确保返回长度一致）
def integrate_alpha(t, T, E, n, A, alpha0=0.0):
    def d_alpha_dt_wrapper(t, y):
        return [d_alpha_dt(y[0], T, E, n, A)]

    # 确保时间数组单调递增
    if not np.all(np.diff(t) > 0):
        t = np.sort(t)

    # 设置合理的 max_step，防止跳过关键点
    max_step = (t[-1] - t[0]) / 200 if len(t) > 1 else 1000

    sol = solve_ivp(
        d_alpha_dt_wrapper,
        [t[0], t[-1]],
        [alpha0],
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-9,
        max_step=max_step
    )

    if sol.success and len(sol.y[0]) == len(t):
        return sol.y[0]
    else:
        print(f"⚠️ solve_ivp 失败 (T={T:.2f} K)，使用线性外推回退")
        # 简单回退：从 alpha0 开始线性增长（仅用于调试，实际应避免）
        return np.linspace(alpha0, min(alpha0 + 0.5, 0.99), len(t))

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
# 1. 输入实验数据（55°C 和 60°C）
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

# ========================
# 🔍 数据筛选：只保留稳定阶段之后的数据
# ========================
threshold_55 = 20000  # s (可调整)
threshold_60 = 20000  # s (可调整)


mask_55 = t_55 >= threshold_55  # 使用 >= 避免浮点误差
mask_60 = t_60 >= threshold_60

t_55_stable = t_55[mask_55]
alpha_55_stable = alpha_55[mask_55]
t_60_stable = t_60[mask_60]
alpha_60_stable = alpha_60[mask_60]

# 输出筛选后数据点数量
print(f"筛选后数据点数量:")
print(f"  55°C (≥ {threshold_55} s): {len(t_55_stable)} 个点")
print(f"  60°C (≥ {threshold_60} s): {len(t_60_stable)} 个点")

if len(t_55_stable) == 0 or len(t_60_stable) == 0:
    raise ValueError("某组温度下无稳定阶段数据，请检查阈值设置！")

# 组织数据列表
t_list = [t_55_stable, t_60_stable]
alpha_list = [alpha_55_stable, alpha_60_stable]
T_list = [T_55, T_60]

# 设置初始 alpha（使用筛选后第一点）
alpha0_list = [alpha_55_stable[0], alpha_60_stable[0]]

# 拼接真实数据
y_true = np.concatenate(alpha_list)

# ========================
# 调试：检查预测长度是否匹配
# ========================
initial_guess = [70000, 0.9, 1e5]  # 建议 A 更大，如 1e8
test_pred = model_prediction(initial_guess, t_list, T_list, alpha0_list)

print(f"\n🔍 调试信息:")
print(f"  y_true 长度: {len(y_true)}")
print(f"  test_pred 长度: {len(test_pred)}")

if len(y_true) != len(test_pred):
    print("❌ 长度不匹配！请检查数据或积分设置")
else:
    print("✅ 长度匹配，准备拟合...")

# ========================
# 初始猜测和边界
# ========================
bounds = ([0, 0, 0], [200000, 5.0, 1e15])  # 更合理的边界

# ========================
# 定义残差函数
# ========================
def residuals(params):
    pred = model_prediction(params, t_list, T_list, alpha0_list)
    if len(pred) != len(y_true):
        print(f"⚠️ 预测长度 {len(pred)} ≠ 真实长度 {len(y_true)}")
        # 补全或截断（临时处理）
        if len(pred) < len(y_true):
            pad_len = len(y_true) - len(pred)
            pred = np.pad(pred, (0, pad_len), mode='edge')
        else:
            pred = pred[:len(y_true)]
    return pred - y_true

# ========================
# 执行全局拟合
# ========================
print("\n🚀 开始全局拟合...")
result = least_squares(
    residuals,
    initial_guess,
    bounds=bounds,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    method='trf',
    verbose=2
)

if result.success:
    E_fit, n_fit, A_fit = result.x
    print(f"\n✅ 拟合成功！")
    print(f"活化能 E = {E_fit:.2f} J/mol")
    print(f"反应级数 n = {n_fit:.2f}")
    print(f"指前因子 A = {A_fit:.2e} 1/s")
    print(f"拟合残差平方和: {result.cost:.6f}")

    # ========================
    # 计算 R²（全局）
    # ========================
    y_pred = model_prediction([E_fit, n_fit, A_fit], t_list, T_list, alpha0_list)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"R² (global) = {r2:.6f}")

    # ========================
    # 分组 R²
    # ========================
    print("\n🧪 分组 R²:")
    labels = ['55°C (328.15 K)', '60°C (333.15 K)']
    for i in range(2):
        sim = integrate_alpha(t_list[i], T_list[i], E_fit, n_fit, A_fit, alpha0_list[i])
        r2_i = r2_score(alpha_list[i], sim)
        print(f"  {labels[i]}: R² = {r2_i:.6f}")

    # ========================
    # 绘图
    # ========================
    plt.figure(figsize=(10, 7))
    colors = ['tab:blue', 'tab:orange']

    for i in range(2):
        t_full = [t_55, t_60][i]
        alpha_full = [alpha_55, alpha_60][i]
        t_stable = t_list[i]
        alpha_stable = alpha_list[i]
        sim_stable = integrate_alpha(t_stable, T_list[i], E_fit, n_fit, A_fit, alpha0_list[i])

        plt.plot(t_full, alpha_full, 'o', color=colors[i], label=f'Exp - {labels[i]}', alpha=0.7, markersize=4)
        plt.plot(t_stable, sim_stable, '-', color=colors[i], linewidth=2, label=f'Fit - {labels[i]}')

    plt.xlabel('Time (s)')
    plt.ylabel('Conversion α')
    plt.title('Kinetic Fitting on Stable Phase (Global Fit)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ========================
    # 参数总结
    # ========================
    print("\n📊 拟合参数总结:")
    print(f"E  = {E_fit:8.2f} J/mol")
    print(f"n  = {n_fit:8.2f}")
    print(f"A  = {A_fit:8.2e} 1/s")
    print(f"R² = {r2:8.6f}")

else:
    print("❌ 拟合失败，请检查初始值或数据！")
    print(result.message)
