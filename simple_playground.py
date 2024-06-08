import random as r
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from simple_geometry import *
import numpy as np
import matplotlib.patches
from PyQt5 import QtWidgets, QtCore
matplotlib.use('Qt5Agg')

# 隱藏層的激活函數
def relu(x):
    return np.maximum(0, x)

# 多層感知機MLP
def MLP(car_state, Whid, Wout):
    '''
    Whid (numpy array): 輸入層到隱藏層的權重矩陣。
    Wout (numpy array): 隱藏層到輸出層的權重矩陣。
    '''
    front_dist, right_dist, left_dist = car_state
    # 將車輛狀態數據轉換為NumPy數組，作為神經網路的輸入層
    input_layer = np.array([[front_dist, right_dist, left_dist]])
    # 計算隱藏層的輸入與輸出
    hidden_input = input_layer @ Whid
    hidden_output = relu(hidden_input)

    # 計算輸出層的輸入
    SUMout = hidden_output @ Wout

    # 使用tanh因為可以幫我轉換為-1~1之間的值
    wheel_angle = 40 * np.tanh(SUMout * 1/5)

    return wheel_angle[0, 0]


class Car:
    def __init__(self) -> None:
        self.diameter = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def radius(self):
        return self.diameter / 2

    # reset the car to the beginning line
    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.diameter)
        left_xpos = self.xini_min + self.diameter // 2
        self.xpos = r.random() * xini_range + left_xpos  # random x pos [-3, 3]
        # self.xpos = 0
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle < self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    # this is the function returning the coordinate on the right, left, front or center points
    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.diameter / 2, 0).rotate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.diameter / 2, 0).rotate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle / 180 * m.pi) * self.diameter / 2 + self.xpos
            fy = m.sin(self.angle / 180 * m.pi) * self.diameter / 2 + self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    # set the car state from t to t+1
    def tick(self):
        car_angle = self.angle / 180 * m.pi
        wheel_angle = self.wheel_angle / 180 * m.pi
        new_x = self.xpos + m.cos(car_angle + wheel_angle) + \
                m.sin(wheel_angle) * m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle + wheel_angle) - \
                m.sin(wheel_angle) * m.cos(car_angle)
        new_angle = (car_angle - m.asin(2 * m.sin(wheel_angle) / self.diameter)) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground:
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]
        self.complete = False

        self.car = Car()
        self.reset()

    def succeeded(self):
        return self.complete  # 返回模擬是否成功 
     
    def _setDefaultLine(self):
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    @property
    def n_actions(self):  
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
        ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
        ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
        ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')  # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        radius = self.car.radius

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        # if we finish the tour
        done = False if not isAtDestination else True
        self.complete = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # check every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos - p1).length, (cpos - p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < radius)
            p2_touch = (dp2 < radius)
            body_touch = (
                    dToLine < radius and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1) * front_u + p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1) * right_u + p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1) * left_u + p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.complete = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)
        self._checkDoneIntersects()

    def step(self, action=None):
        if action:
            # angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(action)

        if not self.done:
            self.car.tick()
            self._checkDoneIntersects()
            return self.state
        else:
            return self.state

    def run(self, weight):
        Whid = weight[:-1]
        Wout = weight[-1].reshape(-1, 1)
        wheel_angle = MLP(self.state, Whid, Wout)

        self.step(wheel_angle)

# 粒子
class Particle:
    def __init__(self, neuronNumber_hid=100):
        self.neuronNumber_hid = neuronNumber_hid
        self.Whid = np.random.uniform(-8.0, 8.0, size=(3, self.neuronNumber_hid))
        self.Wout = np.random.uniform(-8.0, 8.0, size=(self.neuronNumber_hid, 1))
        self.pre_velocity = 0 #前一時間速度
        self.cur_velocity = 0 #當前速度
        self.route_history = [] #紀錄車輛行駛路徑
        self.succeeded = False #紀錄是否抵達終點

    # 更新權重
    def update_weight(self, previous_best, global_best, phi1=2.0, phi2=2.0):
        self.pre_velocity = self.cur_velocity
        #把權重堆疊成一個數組
        combined = np.vstack((self.Whid, self.Wout.reshape(1, -1)))
        # 計算新的速度，包含自我認知分量和社會認知分量
        self.cur_velocity = self.pre_velocity + phi1 * (previous_best - combined) + phi2 * (global_best - combined)
        #更新權重
        self.Whid = self.Whid + self.cur_velocity[:-1]
        self.Wout = self.Wout + self.cur_velocity[-1].reshape(-1, 1)

    # 執行car在playground上的計算
    def runInPlayground(self, playground: Playground):
        playground.reset()
        self.route_history.clear()
        self.route_history.append([playground.car.xpos, playground.car.ypos])
        while not playground.done:
            wheel_angle = MLP(playground.state, self.Whid, self.Wout)
            playground.step(wheel_angle)
            self.route_history.append([playground.car.xpos, playground.car.ypos])
        self.succeeded = playground.complete

    # 回傳車子的路徑紀錄
    def get_route(self):
        return self.route_history

    # 回傳這個權重是否可以到達終點
    def can_finish(self):
        return self.succeeded

    # 回傳車子的神經元權重
    def get_weight(self):
        return np.vstack((self.Whid, self.Wout.reshape(1, -1)))

# PSO
class PSO:
    def __init__(self, init_particleNumber=50):
        self.previous_best_particle_weight = None
        self.previous_best_particle_fitnessVal = float("-inf")
        self.find_successParticle = False

        self.playground = Playground()
        self.particles = [Particle() for i in range(init_particleNumber)]

    # 適應度函數
    def fittness_func(self, car_traj: list, succeeded: bool):
        if succeeded:
            self.find_successParticle = True  

        func = ( 0.8 * car_traj[-1][0] +   # 最終的x值
                 1 * car_traj[-1][1] -     # 最終的y值
                 0.4 * len(car_traj) +     # 移動路線長度
                (200 if succeeded else 0))     # 成功的路徑出現 

        return func

    def train_model(self):
        while not self.find_successParticle:
            # 先求得每台車子的運行結果
            for particle in self.particles:
                particle.runInPlayground(self.playground)

            # 計算歷史最佳和鄰近最佳
            for ind_out, particle in enumerate(self.particles):
                # 更新歷史最佳
                if (self.fittness_func(particle.get_route(), particle.can_finish()) >self.previous_best_particle_fitnessVal):
                    self.previous_best_particle_fitnessVal = self.fittness_func(particle.get_route(),
                                                                                particle.can_finish())
                    self.previous_best_particle_weight = particle.get_weight()
                    if (particle.can_finish()):
                        self.find_successParticle = True

                # 更新鄰居最佳
                global_best_particle_weight = particle.get_weight()
                global_best_particle_fitnessVal = self.fittness_func(particle.get_route(), particle.can_finish())
                for ind_in, neighbor in enumerate(self.particles):
                    # 若找到鄰近最佳
                    if (self.fittness_func(neighbor.get_route(), neighbor.can_finish()) >
                            global_best_particle_fitnessVal):
                        global_best_particle_weight = neighbor.get_weight()
                        global_best_particle_fitnessVal = self.fittness_func(neighbor.get_route(),
                                                                               neighbor.can_finish())
                # 更新粒子的速度和位置
                particle.update_weight(self.previous_best_particle_weight, global_best_particle_weight)

    # 取得最佳的粒子
    def get_weight(self):
        return self.previous_best_particle_weight

    def del_particles(self):
        for particle in self.particles:
            del particle
        del self.playground

class Animation(QtWidgets.QMainWindow):
    '''
    state: 當前狀態
    QtCore.QTimer: 控制動畫的執行頻率和狀態
    '''
    def __init__(self, play: Playground, pso: PSO):
        super().__init__()
        self.play = play
        self.state = self.play.reset()
        self.now_running = False
        self.timer = QtCore.QTimer(self)
        self.path_points = []

        # 訓練模型
        pso.train_model()
        self.weight = pso.get_weight()
        pso.del_particles()

        # 建立主視窗
        self.setWindowTitle("操作介面")
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)

        # 建立開始按鈕
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_animation)
        # 停止動畫按鈕
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_animation)

        # 建立繪畫動畫的區域
        # 動畫的底板
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)

        # 建立畫面配置
        # 主要畫面
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.canvas)
        self.setup_animation()

    def setup_animation(self):
        '''
        start_line: 起點
        finish_line: 終點
        car_path: 明確路徑
        direction_line: 顯示車子所面向的方向
        text: 感測器偵測到的距離    
        '''
        self.ax = self.figure.add_subplot(111)
        self.background = self.play.lines
        self.start_line = self.play.decorate_lines[0]
        self.finish_line = self.play.destination_line
        self.car_radius = self.play.car.radius
        self.direction_line, = self.ax.plot([], [], 'r-')  # 指引方向的線
        self.car_path, = self.ax.plot([], [], 'g-', linewidth=2)
        self.text = self.ax.text(15, 0, '', fontsize=10)

        self.draw_background()

    def draw_background(self):
        for line in self.background:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], "k-")

        # 起點
        self.ax.plot([self.start_line.p1.x, self.start_line.p2.x],
                     [self.start_line.p1.y, self.start_line.p2.y], "b-")
        # 終點的長方形
        self.ax.plot([self.finish_line.p1.x, self.finish_line.p2.x],
                     [self.finish_line.p1.y, self.finish_line.p1.y], "r-")
        self.ax.plot([self.finish_line.p1.x, self.finish_line.p2.x],
                     [self.finish_line.p2.y, self.finish_line.p2.y], "r-")

        self.ax.axis('equal')

    # 初始化各項變數後開始動畫
    def start_animation(self):
        if self.now_running:
            self.timer.stop()

        self.clean()
        self.play.reset()
        self.now_running = True

        # 更新動畫的函數
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # 每 50 毫秒更新一次

    # 停止動畫
    def stop_animation(self):
        self.timer.stop()
        self.now_running = False
        self.clean()

    # 更新動畫的畫面
    def update_animation(self):
        car_pos = self.play.car.getPosition("center")
        self.path_points.append((car_pos.x, car_pos.y))
        self.update_path()
        self.draw_car(car_pos)
        self.text.set_text(
            f'front sensor: {self.play.state[0]:.{3}f}\n'
            f'right sensor: {self.play.state[1]:.{3}f}\n'
            f'left sensor: {self.play.state[2]:.{3}f}'
        )

        # 抵達終點的話就算成功
        if self.play.done:
            if self.play.complete:
                self.show_message("Reach Destination!")
            # else:
            #     self.show_message("Failed!")
            self.timer.stop()
            self.now_running = False

        self.play.run(self.weight)

        # 畫出所有移動畫面
        self.canvas.draw()

    def update_path(self):
        if self.path_points:
            x, y = zip(*self.path_points)
            self.car_path.set_data(x, y)  # 更新路徑數據

    # 秀訊息
    def show_message(self, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(message)
        msg_box.exec_()

    # 畫出車子
    def draw_car(self, car_pos):
        self.car = plt.Circle((car_pos.x, car_pos.y), self.car_radius, color="green", fill=False)
        self.ax.add_patch(self.car)
        front_sensor = self.play.car.getPosition("front")
        self.direction_line.set_data([car_pos.x, front_sensor.x], [car_pos.y, front_sensor.y])

    # 清理之前的車子移動軌跡
    def clean(self):
        for trace in self.ax.patches:
            trace.remove()
        self.path_points.clear()
        self.car_path.set_data([], [])

    # 實際顯示整個動畫
    def run(self):
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    GUI = Animation(Playground(), PSO())
    GUI.run()
    # 啟動 PyQt5 事件循環。
    app.exec_()
