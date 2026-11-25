import numpy as np
import matplotlib.pyplot as plt

# 设置字体支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义函数
def f(x):
    return (np.exp(x) - np.exp(-x)) / (x**2)

# 生成数据（避开 x=0）
x1 = np.linspace(-5, -0.05, 1000)
x2 = np.linspace(0.05, 5, 1000)
x = np.concatenate((x1, x2))
y = f(x)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, y, color='black', linewidth=2)

# 绘制坐标轴
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# 设置坐标轴范围
plt.ylim(-5, 10)
plt.xlim(-5, 5)

# 去除顶部和右侧的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.text(0.1, 1, 'O', fontsize=12)
plt.text(1.1, -0.5, '1', fontsize=12)
plt.text(-0.6, 9, 'y', fontsize=14, fontstyle='italic')
plt.text(4.8, -0.8, 'x', fontsize=14, fontstyle='italic')

plt.title(r'\(f(x) = \frac{e^x - e^{-x}}{x^2}\)', fontsize=16)
plt.grid(False)

# 显示或保存图片
plt.savefig('function_image.png', dpi=300, bbox_inches='tight')
plt.show()