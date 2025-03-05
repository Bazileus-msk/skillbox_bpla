#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Regislab
"""

import numpy as np
from matplotlib import pyplot as plt


class VehicleSimpleDynamic:
    def __init__(self, mass: float, k_b: float,
                 rotor_count: int, acc_init=0.0,
                 vel_init=0.0, altitude_init=0.0):
        '''

        :param mass: масса аппарата
        :type mass: float
        :param k_b: коэффициент тяги двигателя
        :type k_b: float
        :param rotor_count: количество двигателей в системе
        :type rotor_count: int
        :param acc_init: начальное значение ускорения ЛА
        :type acc_init: float
        :param vel_init: начальное значение скорости ЛА
        :type vel_init: float
        :param altitude_init: начальное значение положения ЛА
        :type altitude_init: float

        '''

        self._mass = mass
        self._k_b = k_b
        self._rotor_count = rotor_count
        self._acceleration = acc_init
        self._velocity = vel_init
        self._altitude = altitude_init
        # Величина ускорения свободного падения
        self._g = 9.81 

    def _rightParts(self, rotors_angular_vel):
        '''

        :param rotorsAngularVel: угловая скорость двигателей
        :type rotorsAngularVel: float

        '''
        
        # Проверяем что количество двигателей совпадает с размерностью команды
        if (len(rotors_angular_vel) != self._rotor_count):
            raise ValueError("Incorrect number of motors")

        # Для всех двигателей рассчитаем суммарную тягу
        rotors_angular_vel_sum = np.sum(np.square(rotors_angular_vel))
        
        # Вычисляем ускорение нашей системы.
        # Получаем силу тяги создаваемую всеми двигателями 
        # согласно нашей математической модели Fтяги = Kb * omega^2
        # После для получения текущего ускорения разделим полученное значение 
        # на массу и вычтем ускорение свободного падения действующие на аппарат
        self._acceleration = (self._k_b * rotors_angular_vel_sum) \
                            / self._mass - self._g

    def _integrate(self, dt: float):
        '''

        :param dt: шаг моделирования
        :type dt: float

        '''
        # интегрируем ускорение методом эйлера
        self._velocity += self._acceleration * dt
        # Полученную скорость интегрируем для определения местоположения
        self._altitude += self._velocity * dt

    def computeAltitude(self, u, dt: float):
        '''

        :param u: управляющие воздействие
        :type u: float
        :param dt: шаг моделирования
        :type dt: float

        '''
        # Для определения положения вызываем метод для правых частей(то есть наших приращений от перемещения)
        # В данном случае приращением выступает ускорение нашей системы.
        # Для следования заданному целевому значению высоты передаем в метод наше управляющие воздействие u,
        # которое характеризует необходимую угловую скорость двигателей.
        self._rightParts(u)
        # Далее вызываем метод интегрирования
        # Интегрируем полученное при помощи функции правых частей ускорение и получаем скорость,
        # после интегрируем скорость и получаем положение.
        self._integrate(dt)


    @property
    def getAltitude(self):
        '''

        :return: положение ЛА
        :rtype: float

        '''
        return self._altitude


    @property
    def getVelocity(self):
        '''

        :return: скорость ЛА
        :rtype: float

        '''
        return self._velocity

    @property
    def getAcceleration(self):
        '''

        :return: ускорение
        :rtype: float

        '''
        return self._acceleration

class Pid():
    def __init__(self, k_p, k_i, k_d):
        '''

        :param k_p: пропорциональный коэффициент
        :type k_p: float
        :param k_i: интегральный коэффициент
        :type k_i: float
        :param k_d: дифференциальный коэффициент
        :type k_d: float

        '''

        # объявим приватные поля класса
        self._k_p = k_p
        self._k_i = k_i
        self._k_d = k_d
        self._error_cmd = 0.0
        self._integration__error_cmd = 0.0
        self._differentiation_error_cmd = 0.0
        self._previous_error_cmd = 0.0


    def updatedPid(self, cmd_des, cmd_current, dt):
        self._error_cmd = cmd_des - cmd_current
        self._integration__error_cmd += self._error_cmd * dt
        # проверка деления на ноль
        try:
            self._differentiation_error_cmd = (self._error_cmd -
                                        self._previous_error_cmd) / dt
        except ZeroDivisionError as e:
            print("Error: Division by zero is not allowed.")

        self._previous_error_cmd = self._error_cmd

        return (self._k_p * self._error_cmd +
                self._k_i * self._integration__error_cmd +
                self._k_d * self._differentiation_error_cmd)

class ControlSystem():
    def __init__(self, k_p, k_i, k_d, motor_speed_limit):
        '''

        :param k_p: коэффициент П регулятора
        :type k_p: float
        :param k_i: коэффициент И регулятора
        :type k_i: float
        :param k_d: коэффициент Д регулятора
        :type k_d: float
        :param controlLimit: ограничение по управляющему воздействию
        :type controlLimit: float

        '''
        self._pid = Pid(k_p, k_i, k_d)
        self._k_p = k_p
        self._k_i = k_i
        self._k_d = k_d
        self._position_des = 0.0
        self._error = 0.0
        self._error_past = 0.0
        self._integral = 0.0
        self._motor_speed_limit = motor_speed_limit


    @property
    def desiredPosition(self):
        return self._position_des


    @desiredPosition.setter
    def desiredPosition(self, position_des):
        '''

        :param position_des: целевое положение ЛА
        :type position_des: float

        '''
        # данный метод устанавливает целевое положение ЛА,
        # к которому система с течением времени будет стремиться(в нашем примере это высота)
        self._position_des = position_des


    def compute_motor_velocity(self, position_current, dt):
        '''

        :param currentPosition: текущее положение ЛА
        :type currentPosition: float
        :param dt: шаг моделирования
        :type dt: float

        '''
        u = self._pid.updatedPid(self._position_des, position_current, dt)

        # Вызовем звено насыщения для ограничения максимального управляющего воздействия
        u = np.clip(u, -self._motor_speed_limit, self._motor_speed_limit)

        u = self._mixer(u)

        return u


    def _mixer(self, cmd):
        # Этот метод обеспечивает перевод команд от системы управления.
        # в желаемую угловую скорость для каждого двигателя.
        res_motors_vel = [cmd, cmd, cmd, cmd]

        return (res_motors_vel)


class Quadcopter():
    def __init__(self, control_system: ControlSystem,
                 math_model: VehicleSimpleDynamic):

        self._control_system = control_system
        self._math_model = math_model
        self._current_altitude = 0.0

    @property
    def currentAltitude(self):
        return (self._current_altitude)

    def goto(self, point):
        self._control_system.desiredPosition = point[2]


    def updatedState(self, dt):
        # рассчитываем новое управляющие воздействие
        # на основе текущей высоты(_current_altitude) ЛА 
        u = self._control_system.compute_motor_velocity(self._current_altitude, dt)
        # Рассчитываем положение ЛА с учетом полученного
        # управляющего воздействия 
        self._math_model.computeAltitude(u, dt)
        self._current_altitude = self._math_model.getAltitude


    @property
    def getPosition(self):
        pass


    @property
    def getVelocity(self):
        return (self._math_model.getVelocity)
    
    @property
    def getAcceleration(self):
        return (self._math_model.getAcceleration)


class Simulator:

    def __init__(self, Tend: float, dt: float,
                 quadcopter: Quadcopter):
        '''

        :param Tend: конечное время моделирования
        :type Tend: float
        :param dt: шаг моделирования
        :type dt: float
        :param quadcopter: объект квадрокоптера
        :type quadcopter: Quadcopter

        '''
        self.dt = dt
        self.T_end = Tend
        self.quadcopter = quadcopter
        self.accList = []
        self.velList = []
        self.posList = []
        self.timeList = []

    def runSimulation(self):
        '''

       метод запускает моделирование системы от 0 до конечного времени Tend 
       с шагом dt

        '''
        # Задаем 0 время и начинаем рассчет до тех пор пока 
        # время не достигнет конечного значения T_end
        time = 0
        for time in np.arange(0, self.T_end, self.dt):
            quadcopter.updatedState(dt)
            # Записываем полученные значения в списки
            # для дальнейшего построения графиков
            self.posList.append(quadcopter.currentAltitude)
            self.velList.append(quadcopter.getVelocity)
            self.accList.append(quadcopter.getAcceleration)
            self.timeList.append(time)

            # увеличиваем время на dt, то есть на шаг моделирования
            time += self.dt

    def showPlots(self):
        '''
        метод строит графики на основе измерений полученных в 
        ходе моделирования системы

        '''
        f = plt.figure(constrained_layout=True)
        gs = f.add_gridspec(3, 5)
        ax1 = f.add_subplot(gs[0, :-1])
        ax1.plot(self.timeList, self.posList)
        ax1.grid()
        ax1.set_title('position')

        ax2 = f.add_subplot(gs[1, :-1])
        ax2.plot(self.timeList, self.velList, "g")
        ax2.grid()
        ax2.set_title('velocity')

        ax3 = f.add_subplot(gs[2, :-1])
        ax3.plot(self.timeList, self.accList, "r")
        ax3.grid()
        ax3.set_title('acceleration')

        plt.show()


'''
 Объявим параметры для моделирования
'''
k_p = 300 #коэффициент Пропорционального регулирования
k_i = 35 #коэффициент Интегрального регулирования
k_d = 180 #коэффициент Дифференциального регулирования
dt = 0.01 # шаг моделирования системы (например одна сотая секунды)

Tend = 10 # конечное время моделирования (например 20 сек)

# Масса ЛА
mass = 0.006
# Коэффициент тяги двигателя ЛА
k_b = 3.9865e-08
# Количество двигателей ЛА
rotor_count = 4
# Ограничение на угловую скорость двигателей рад/сек
motor_speed_limit = 1000

'''
Создадим объект контроллера и объект для нашей математической модели
'''
control_system = ControlSystem(k_p, k_i, k_d, motor_speed_limit)
math_model =  VehicleSimpleDynamic(mass, k_b, rotor_count)

quadcopter = Quadcopter(control_system, math_model)

'''
Установим целевое положение для нашей системы
'''
point = [0.0, 0.0, 10.0]

quadcopter.goto(point)

"""
Создадим объект симулятора и передадим в него контроллер
 и математическую модель
"""
sim = Simulator(Tend, dt, quadcopter)
sim.runSimulation()  # запуск симулятора
sim.showPlots()  # построение графиков
