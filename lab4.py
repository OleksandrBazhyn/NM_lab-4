import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd

# Вихідні дані
x_min, x_max = 0, np.pi  # Проміжок [0, π]
n_points = 15            # Кількість точок для таблиці
custom_point = 0.7

# Табличні значення
x_table = np.linspace(x_min, x_max, n_points)  # Рівномірний розподіл точок
x_table = np.append(x_table, custom_point)
x_table = np.sort(x_table)                     # Сортування точок
y_table = np.sin(x_table)                      # Значення аналітичної функції sin(x)

# Обчислення розділених різниць для методу Ньютона
def divided_differences(x, y):
    n = len(x)
    coef = np.copy(y)  # Початкові значення коефіцієнтів
    for j in range(1, n):
        coef[j:] = (coef[j:] - coef[j - 1]) / (x[j:] - x[j - 1])
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

def inverse_interpolation(y_target, x_table, y_table):
    # Побудова інтерполяційної функції
    def newton_interp_func(x):
        return newton_polynomial(x, x_table, newton_coef) - y_target

    # Початкове наближення ближче до очікуваного розв'язку
    x_guess = 0.5
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

# Таблиця значень
table_data = {"x": x_table, "sin(x)": y_table}
table_df = pd.DataFrame(table_data)

print(table_df)



# Функція для обчислення розділених різниць
def compute_divided_differences(x_table, y_table):
    n = len(x_table)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_table  # Перший стовпець — значення функції
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / (x_table[i + j] - x_table[i])
    
    return divided_diff

# Обчислення коефіцієнтів полінома Ньютона
divided_diff_table = compute_divided_differences(x_table, y_table)
newton_coefficients = divided_diff_table[0, :] 

divided_diff_df = pd.DataFrame(np.round(divided_diff_table, 4), 
                               columns=[f"Lv {i}" for i in range(len(x_table))], 
                               index=[f"x{i}" for i in range(len(x_table))])


def newton_interpolation(x_target, x_nodes, coefficients):
    n = len(coefficients)
    interpolated_value = coefficients[0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x_target - x_nodes[i - 1])
        interpolated_value += coefficients[i] * product_term
    return interpolated_value

interpolated_value = newton_interpolation(0.7, x_table, newton_coefficients)


# Вивід результатів
print("Таблиця розділених різниць:")
print(divided_diff_df)

print("\nКоефіцієнти полінома Ньютона:")
for i, coef in enumerate(newton_coefficients):
    print(f"a[{i}] = {coef:.6f}")

print(f"\nІнтерпольоване значення в точці x = {0.7}: {interpolated_value:.6f}")