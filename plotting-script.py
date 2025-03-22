import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 默认绘图风格和调色板
sns.set(style='whitegrid')  # 使用 Seaborn 的默认样式替代 plt.style.use('seaborn')
sns.set_palette("deep")

# 读取CSV文件
df = pd.read_csv('vehicle_control_data.csv')

# 创建一个4x2的子图网格
fig, axs = plt.subplots(4, 1, figsize=(12, 16))
fig.suptitle('Vehicle Control Performance Analysis', fontsize=16)

# 1. 横向位置追踪
axs[0].plot(df['time'], df['y'], label='Actual Y')
axs[0].plot(df['time'], df['y_ref'], label='Reference Y', linestyle='--')
axs[0].set_title('Lateral Position Tracking')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Lateral Position (m)')
axs[0].legend()
axs[0].grid(True)

# 2. 航向角追踪
axs[1].plot(df['time'], df['psi'], label='Actual Heading')
axs[1].plot(df['time'], df['psi_ref'], label='Reference Heading', linestyle='--')
axs[1].set_title('Heading Angle Tracking')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Heading Angle (rad)')
axs[1].legend()
axs[1].grid(True)

# 3. 速度追踪
axs[2].plot(df['time'], df['vx'], label='Actual Speed')
axs[2].plot(df['time'], df['speed_ref'], label='Reference Speed', linestyle='--')
axs[2].set_title('Speed Tracking')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Speed (m/s)')
axs[2].legend()
axs[2].grid(True)

# 4. 误差分析
axs[3].plot(df['time'], df['lateral_error'], label='Lateral Error')
axs[3].plot(df['time'], df['heading_error'], label='Heading Error')
axs[3].plot(df['time'], df['speed_error'], label='Speed Error')
axs[3].set_title('Tracking Errors')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Error')
axs[3].legend()
axs[3].grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('vehicle_control_performance.png', dpi=300)

# 打印一些基本统计信息
print("横向误差统计:")
print(f"平均绝对误差: {df['lateral_error'].abs().mean():.4f} m")
print(f"最大绝对误差: {df['lateral_error'].abs().max():.4f} m")

print("\n航向角误差统计:")
print(f"平均绝对误差: {df['heading_error'].abs().mean():.4f} rad")
print(f"最大绝对误差: {df['heading_error'].abs().max():.4f} rad")

print("\n速度误差统计:")
print(f"平均绝对误差: {df['speed_error'].abs().mean():.4f} m/s")
print(f"最大绝对误差: {df['speed_error'].abs().max():.4f} m/s")