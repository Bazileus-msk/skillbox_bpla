import math
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MathModel:
    def __init__(self, I_y, k_b, l, T_cmd):
        self.k_b = k_b
        self.I_y = I_y
        self.l = l
        self.T_cmd = T_cmd
        self.acceleration = 0.0
        self.velocity = 0.0
        self.position = 0.0

    def main_formule(self, tetta_ddot_cmd):
        # Физически корректная формула момента силы
        M_y = self.I_y * tetta_ddot_cmd
        self.acceleration = M_y / self.I_y

    def integrate(self, dt=0.01):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def calculatePosition(self, tetta_ddot_cmd, dt):
        self.main_formule(tetta_ddot_cmd)
        self.integrate(dt)

    def getAcceleration(self):
        return self.acceleration

    def getVelocity(self):
        return self.velocity

    def getPosition(self):
        return self.position


class Pid:
    def __init__(self, k_p, k_i, k_d,
                 tetta_dot_cmd_lower, tetta_dot_cmd_upper,
                 tetta_ddot_cmd_lower, tetta_ddot_cmd_upper,
                 angle_cmd=0.0):
        """
        :param tetta_dot_cmd_lower: нижний предел скорости (град/с)
        :param tetta_dot_cmd_upper: верхний предел скорости (град/с)
        :param tetta_ddot_cmd_lower: нижний предел ускорения (рад/с²)
        :param tetta_ddot_cmd_upper: верхний предел ускорения (рад/с²)
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.angle_cmd = angle_cmd

        # Ошибки
        self.error_angle = 0.0
        self.integration_error_angle = 0.0
        self.differentiation_error_angle = 0.0
        self.previous_error_angle = 0.0

        self.error_tetta_dot = 0.0
        self.integration_error_tetta_dot = 0.0
        self.differentiation_error_tetta_dot = 0.0
        self.previous_error_tetta_dot = 0.0

        # Пределы скорости (конвертируем градусы/с → радианы/с)
        self.tetta_dot_cmd_lower = math.radians(tetta_dot_cmd_lower)
        self.tetta_dot_cmd_upper = math.radians(tetta_dot_cmd_upper)

        # Пределы ускорения (уже в радианах/с²)
        self.tetta_ddot_cmd_lower = tetta_ddot_cmd_lower
        self.tetta_ddot_cmd_upper = tetta_ddot_cmd_upper

        self.last_tetta_dot_cmd = 0.0

    def Pid_1(self, angle, dt=0.01):
        self.error_angle = self.angle_cmd - angle
        self.integration_error_angle += self.error_angle * dt
        self.differentiation_error_angle = (self.error_angle - self.previous_error_angle) / dt if dt != 0 else 0
        self.previous_error_angle = self.error_angle

        tetta_dot_cmd = (self.k_p * self.error_angle +
                         self.k_i * self.integration_error_angle +
                         self.k_d * self.differentiation_error_angle)

        # Применяем несимметричное ограничение скорости
        self.last_tetta_dot_cmd = self.saturation(
            tetta_dot_cmd,
            self.tetta_dot_cmd_lower,
            self.tetta_dot_cmd_upper
        )
        return self.last_tetta_dot_cmd

    def Pid_2(self, tetta_dot, dt=0.01):
        self.error_tetta_dot = self.last_tetta_dot_cmd - tetta_dot
        self.integration_error_tetta_dot += self.error_tetta_dot * dt
        self.differentiation_error_tetta_dot = (
                                                           self.error_tetta_dot - self.previous_error_tetta_dot) / dt if dt != 0 else 0
        self.previous_error_tetta_dot = self.error_tetta_dot

        tetta_ddot_cmd = (self.k_p * self.error_tetta_dot +
                          self.k_i * self.integration_error_tetta_dot +
                          self.k_d * self.differentiation_error_tetta_dot)

        # Применяем несимметричное ограничение ускорения
        return self.saturation(
            tetta_ddot_cmd,
            self.tetta_ddot_cmd_lower,
            self.tetta_ddot_cmd_upper
        )

    def saturation(self, inputVal, lower, upper):
        """Универсальное несимметричное ограничение"""
        if inputVal < lower:
            return lower
        elif inputVal > upper:
            return upper
        return inputVal


class Simulator:
    def __init__(self, Tend, dt, Pid, mathModel):
        self.dt = dt
        self.Tend = Tend
        self.Pid = Pid
        self.mathModel = mathModel
        self.accList = []
        self.velList = []
        self.posList = []
        self.timeList = []
        self.cmdAccList = []  # Для записи команд ускорения
        self.cmdVelList = []  # Для записи команд скорости

    def runSimulation(self):
        time = 0.0
        while time <= self.Tend:
            pose = self.mathModel.getPosition()
            vel = self.mathModel.getVelocity()
            acc = self.mathModel.getAcceleration()

            self.posList.append(pose)
            self.velList.append(vel)
            self.accList.append(acc)
            self.timeList.append(time)

            # Последовательный вызов ПИД-регуляторов
            tetta_dot_cmd = self.Pid.Pid_1(pose, self.dt)
            tetta_ddot_cmd = self.Pid.Pid_2(vel, self.dt)

            self.cmdVelList.append(tetta_dot_cmd)
            self.cmdAccList.append(tetta_ddot_cmd)

            self.mathModel.calculatePosition(tetta_ddot_cmd, self.dt)
            time += self.dt

    def showPlots(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))

        # Константы преобразования
        rad_to_deg = 180 / math.pi

        # Преобразование в градусы для положения и скорости
        deg_pos = [p * rad_to_deg for p in self.posList]
        deg_vel = [v * rad_to_deg for v in self.velList]
        deg_cmd_vel = [cv * rad_to_deg for cv in self.cmdVelList]

        # Ускорения оставляем в радианах
        rad_acc = self.accList
        rad_cmd_acc = self.cmdAccList

        # Целевые значения и ограничения
        target_deg = self.Pid.angle_cmd * rad_to_deg
        vel_lower_deg = self.Pid.tetta_dot_cmd_lower * rad_to_deg
        vel_upper_deg = self.Pid.tetta_dot_cmd_upper * rad_to_deg
        acc_lower_rad = self.Pid.tetta_ddot_cmd_lower
        acc_upper_rad = self.Pid.tetta_ddot_cmd_upper

        # 1. График углового положения
        ax1.plot(self.timeList, deg_pos, label='Actual')
        ax1.grid()
        ax1.set_ylabel('Position [deg]')
        ax1.axhline(y=target_deg, color='r', linestyle='--', label='Target')
        ax1.set_title(f'Target: {target_deg:.1f}°')
        ax1.legend()

        # 2. График угловой скорости
        ax2.plot(self.timeList, deg_vel, "g", label='Actual')
        ax2.plot(self.timeList, deg_cmd_vel, "b--", alpha=0.7, label='Commanded')
        ax2.grid()
        ax2.set_ylabel('Velocity [deg/s]')
        ax2.axhline(y=vel_lower_deg, color='r', linestyle='--', alpha=0.5, label='Limits')
        ax2.axhline(y=vel_upper_deg, color='r', linestyle='--', alpha=0.5)
        ax2.set_title(f'Velocity Limits: {vel_lower_deg:.1f} - {vel_upper_deg:.1f} deg/s')
        ax2.legend()

        # 3. График углового ускорения (в радианах/с²)
        ax3.plot(self.timeList, rad_acc, "b", label='Actual')
        ax3.grid()
        ax3.set_ylabel('Acceleration [rad/s²]')
        ax3.axhline(y=acc_lower_rad, color='r', linestyle='--', alpha=0.5, label='Limits')
        ax3.axhline(y=acc_upper_rad, color='r', linestyle='--', alpha=0.5)
        ax3.set_title(f'Acceleration Limits: {acc_lower_rad:.1f} - {acc_upper_rad:.1f} rad/s²')
        ax3.legend()

        # 4. График командного ускорения (в радианах/с²)
        ax4.plot(self.timeList, rad_cmd_acc, "m", label='Commanded')
        ax4.grid()
        ax4.set_ylabel('Commanded Accel [rad/s²]')
        ax4.set_xlabel('Time [s]')
        ax4.axhline(y=acc_lower_rad, color='r', linestyle='--', alpha=0.5, label='Limits')
        ax4.axhline(y=acc_upper_rad, color='r', linestyle='--', alpha=0.5)
        ax4.set_title(f'Commanded Acceleration')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('simulation_plot.png', dpi=150)
        plt.close()

        # Проверка соблюдения ограничений
        self.check_limits(deg_vel, deg_cmd_vel, rad_acc, rad_cmd_acc,
                          vel_lower_deg, vel_upper_deg,
                          acc_lower_rad, acc_upper_rad)

    def check_limits(self, deg_vel, deg_cmd_vel, rad_acc, rad_cmd_acc,
                     vel_lower, vel_upper, acc_lower, acc_upper):
        """Проверка соблюдения ограничений"""
        print("\n=== Проверка соблюдения ограничений ===")

        # Проверка ограничений скорости
        vel_min = min(deg_vel)
        vel_max = max(deg_vel)
        vel_cmd_min = min(deg_cmd_vel)
        vel_cmd_max = max(deg_cmd_vel)

        vel_ok = (vel_min >= vel_lower) and (vel_max <= vel_upper)
        vel_cmd_ok = (vel_cmd_min >= vel_lower) and (vel_cmd_max <= vel_upper)

        print(
            f"Скорость: факт {vel_min:.2f}-{vel_max:.2f}° (огранич: {vel_lower:.2f}-{vel_upper:.2f}°) {'✅' if vel_ok else '❌'}")
        print(
            f"Скорость: команд {vel_cmd_min:.2f}-{vel_cmd_max:.2f}° (огранич: {vel_lower:.2f}-{vel_upper:.2f}°) {'✅' if vel_cmd_ok else '❌'}")

        # Проверка ограничений ускорения
        acc_min = min(rad_acc)
        acc_max = max(rad_acc)
        acc_cmd_min = min(rad_cmd_acc)
        acc_cmd_max = max(rad_cmd_acc)

        acc_ok = (acc_min >= acc_lower) and (acc_max <= acc_upper)
        acc_cmd_ok = (acc_cmd_min >= acc_lower) and (acc_cmd_max <= acc_upper)

        print(
            f"Ускорение: факт {acc_min:.2f}-{acc_max:.2f} рад/с² (огранич: {acc_lower:.2f}-{acc_upper:.2f}) {'✅' if acc_ok else '❌'}")
        print(
            f"Ускорение: команд {acc_cmd_min:.2f}-{acc_cmd_max:.2f} рад/с² (огранич: {acc_lower:.2f}-{acc_upper:.2f}) {'✅' if acc_cmd_ok else '❌'}")


# Параметры моделирования

k_p = 5.0 #коэффициент Пропорционального регулирования
k_i = 0.1 #коэффициент Интегрального регулирования
k_d = 0.5 #коэффициент Дифференциального регулирования

dt = 0.01
Tend = 2.0

# Пределы скорости (в градусах/сек)
VEL_LOWER = 50  # нижний предел 50 °/с
VEL_UPPER = 300  # верхний предел 300 °/с

# Пределы ускорения (в радианах/с²)
ACC_LOWER = 5.0  # нижний предел 5 рад/с²
ACC_UPPER = 15.0  # верхний предел 15 рад/с²

# Физические параметры
Iy = 7.16914e-10
k_b = 3.9865e-08
l = 0.17
T_cmd = 10

# Создание объектов
target_angle_deg = 30.0
target_angle_rad = math.radians(target_angle_deg)
controller = Pid(
    k_p, k_i, k_d,
    tetta_dot_cmd_lower=VEL_LOWER,
    tetta_dot_cmd_upper=VEL_UPPER,
    tetta_ddot_cmd_lower=ACC_LOWER,
    tetta_ddot_cmd_upper=ACC_UPPER,
    angle_cmd=target_angle_rad
)
uavSimpleDynamic = MathModel(I_y=Iy, k_b=k_b, l=l, T_cmd=T_cmd)

# Запуск симуляции
sim = Simulator(Tend, dt, controller, uavSimpleDynamic)
sim.runSimulation()
sim.showPlots()

# Автоматическое открытие изображения
try:
    os.startfile('simulation_plot.png')
except:
    print("Graph saved as 'simulation_plot.png'")