import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

# Завантажуємо дані, пропускаємо 5 рядків заголовка і ігноруємо 0-й стовпець
data = np.loadtxt("SCPUnion2.1_mu_vs_z.txt", skiprows=5, converters={0: lambda s: 0})

# Масив червоних зсувів з другого стовпця
z = data[:, 1]
# Масив distance modulus з третього стовпця та помилки з четвертого
DM = data[:, 2]
DM_err = data[:, 3]


# Функція перетворення distance modulus в світлове відстань
def DM2DL(DM):
    return 10 ** (DM / 5 - 1) / 1e4


# Перетворюємо весь масив
DL = DM2DL(DM)

# Швидкість світла в см/с
c = 29979245800  # см/с


# Функція перетворення швидкості в червоний зсув
def v2z(v):
    return np.sqrt((1.0 + v / c) / (1.0 - v / c)) - 1.0


# Лінійний простір швидкостей
v_list = np.linspace(0, c, 100)
z_list = v2z(v_list)


# Функція перетворення червоного зсуву в швидкість
def z2v(z):
    return np.interp(z, z_list, v_list) / 1e5  # Повертаємо швидкість в км/с


# Швидкості наднових
v = z2v(z)

# Створюємо графік даних
plt.plot(DL, v, ".")
plt.xlabel(r"$D_{L}\;\mathrm{[Mpc]}$", size=18)
plt.ylabel(r"$v\;\mathrm{[km/s]}$", size=18)


# Функція E_z для розрахунку космологічного інтегралу
def e_Z(z, OmegaM):
    OmegaL = 1.0 - OmegaM
    return (OmegaM * (1.0 + z) ** 3 + OmegaL) ** (-0.5)


# Функція розрахунку світлового відстані
def D_L(z, OmegaM):
    dh = 4286.0  # c / H0, де H0 = 70 км/с/Мпс
    D_c = dh * si.quad(e_Z, 0.0, z, args=(OmegaM,))[0]
    return D_c * (1.0 + z)


# Функція для роботи з масивами
def D_L_batch(z, OmegaM):
    return np.array([D_L(z_i, OmegaM) for z_i in z])


# Створення графіків для різних значень OmegaM
temp_z = np.linspace(0, 1.5, 100)  # Масив червоних зсувів
temp_v = z2v(temp_z)  # Швидкості

# Створюємо дуги для трьох моделей з різними значеннями OmegaM
plt.plot(D_L_batch(temp_z, 0.01), temp_v, "r")
plt.plot(D_L_batch(temp_z, 0.27), temp_v, "g")
plt.plot(D_L_batch(temp_z, 0.99), temp_v, "m")

plt.legend(("дані", r"$\Omega_M=0.01$", r"$\Omega_M=0.27$", r"$\Omega_M=0.99$"), loc=4)

# Відображаємо графік
plt.show()


# Розрахунок часу Всесвіту для заданого OmegaM
def age_of_universe(OmegaM, OmegaL=0.7):
    # Космологічна стала (H0) в км/с/Мпс
    H0 = 70  # км/с/Мпс
    H0_si = H0 / 3.085677581e19  # Переведемо в 1/с

    # Інтеграл для віку Всесвіту
    def integrand(z):
        return 1 / ((1 + z) * H0_si * np.sqrt(OmegaM * (1 + z) ** 3 + OmegaL))

    # Інтегруємо від 0 до безкінечності
    age, _ = si.quad(integrand, 0, np.inf)
    return age  # Вік у секундах


# Розраховуємо вік Всесвіту для різних OmegaM
OmegaM_values = [0.01, 0.27, 0.99]
for OmegaM in OmegaM_values:
    age = age_of_universe(OmegaM)
    age_years = age / (60 * 60 * 24 * 365.25)  # Переводимо в роки
    print(f"Вік Всесвіту для Omega_M={OmegaM}: {age_years:.2f} років")
