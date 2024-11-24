import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd

# Вихідні дані
x_min, x_max = 0, np.pi  # Проміжок [0, π]
n_points = 15            # Кількість точок для таблиці
custom_point = 0.7
x_table = np.linspace(x_min, x_max, n_points) # Рівномірний розподіл точок
x_table = np.append(x_table, custom_point)
x_table = np.sort(x_table)
y_table = np.sin(x_table)                      # Значення аналітичної функції sin(x)

# Обчислення різниць для методу Ньютона
def divided_differences(x, y):
    n = len(x)
    coef = np.zeros(n)
    coef[0] = y[0]
    for j in range(1, n):
        coef[j] = (y[j] - coef[j - 1]) / (x[j] - x[j - 1])
    return coef

# Поліном Ньютона
def newton_polynomial(x, x_data, coef):
    n = len(coef)
    result = coef[0]
    product = 1
    for i in range(1, n):
        product *= (x - x_data[i - 1])
        result += coef[i] * product
    return result

# Обчислення коефіцієнтів
newton_coef = divided_differences(x_table, y_table)

# Графік
x_fine = np.linspace(x_min, x_max, 500)  # Точки для гладкого графіка
y_exact = np.sin(x_fine)                # Аналітична функція
y_newton = [newton_polynomial(x, x_table, newton_coef) for x in x_fine]  # Поліном Ньютона

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_exact, label="Аналітична функція (sin(x))", color="blue")
plt.plot(x_fine, y_newton, label="Інтерполяція методом Ньютона", linestyle="--", color="red")
plt.scatter(x_table, y_table, label="Табличні значення", color="black", zorder=5)
plt.title("Інтерполяція методом Ньютона")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# Задача оберненої інтерполяції
y_target = 0.5  # Значення, яке шукаємо в таблиці
# Розв'язання рівняння методом Ньютона для табличної функції
def inverse_interpolation(y_target, x_table, y_table):
    # Побудова інтерполяційної функції
    def newton_interp_func(x):
        return newton_polynomial(x, x_table, newton_coef) - y_target

    # Використовуємо scipy fsolve для пошуку кореня
    x_guess = np.mean(x_table)  # Початкове наближення
    x_solution = fsolve(newton_interp_func, x_guess)[0]
    return x_solution

# Знаходимо розв'язок
x_solution = inverse_interpolation(y_target, x_table, y_table)

# Візуалізація результату оберненої інтерполяції
plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_exact, label="Аналітична функція (sin(x))", color="blue")
plt.scatter(x_solution, y_target, color="green", label=f"Розв'язок: x = {x_solution:.4f}", zorder=5)
plt.scatter(x_table, y_table, label="Табличні значення", color="black", zorder=5)
plt.axhline(y=y_target, color="orange", linestyle="--", label=f"y = {y_target}")
plt.title("Обернена інтерполяція для sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

table_data = {"x": x_table, "sin(x)": y_table}
table_df = pd.DataFrame(table_data)

print(table_df)