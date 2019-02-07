import numpy as np


class EnvRoad:
    action_amount = 3

    def __init__(self, config):
        self.config = config

        self.map = Road()
        self.robot = Robot(self.map.width/2., 1., 0., self.map, config)
        self.car1 = Car(-4., self.map.sidewalkwidth + self.map.lanewidth/2., config.car_speed)

        self.cars = [self.car1]

        self.up = True  # Goal

    def act(self, act):
        collision = False
        reward = 0.

        if act == 0:
            self.robot.move(0.0, self.robot.turn_speed)
            reward = -0.01
        elif act == 1:
            self.robot.move(self.robot.speed, 0.)
            reward = self.config.reward_forward
            if self.robot.x > self.map.width or self.robot.x < 0 or self.robot.y > self.map.length or self.robot.y < 0:
                self.robot.move(-self.robot.speed, 0.)
                reward = self.config.reward_wall
                collision = False
        elif act == 2:
            self.robot.move(0.0, -self.robot.turn_speed)
            reward = -0.01

        if self.up and self.robot.y > self.map.length - 2:
            reward = self.config.reward_goal
            self.up = False
        if not self.up and self.robot.y < 2:
            reward = self.config.reward_goal
            self.up = True

        for car in self.cars:
            if ((car.x + car.length * 0.5) > self.robot.x > (car.x - car.length * 0.5)) and (
                    (car.y + car.width * 0.5) > self.robot.y > (car.y - car.width * 0.5)):  # Crash
                reward = self.config.reward_car
                collision = True
                break

            car.drive()

            if ((car.x + car.length * 0.5) > self.robot.x > (car.x - car.length * 0.5)) and (
                    (car.y + car.width * 0.5) > self.robot.y > (car.y - car.width * 0.5)):  # Crash
                reward = self.config.reward_car
                collision = True
                break

        x, y, dist = self.robot.get_laser_distance(self.cars)

        dist = dist / self.robot.max_dist_laser

        for car in self.cars:
            if car.x > (self.map.width+5.) or car.x < -5.:  # Respawn car if out of bounce
                car.respawn()

        return dist, reward, collision

    def new_game(self):
        self.robot.x = self.map.width/2.
        self.robot.y = 1.
        self.robot.theta = 0.

        for car in self.cars:
            car.respawn()

        self.up = True


class Robot:
    def __init__(self, x, y, theta, map, config):
        self.x = x
        self.y = y
        self.theta = theta

        self.speed = config.speed
        self.turn_speed = config.turn_speed

        self.lasers = config.laser_amount
        self.max_dist_laser = config.laser_max_dist
        self.min_angle_laser = config.min_angle
        self.max_angle_laser = config.max_angle

        self.noise = 0.0001  # std

        self.map = map

    def move(self, v, dtheta):
        self.theta = self.theta + dtheta
        self.x = self.x + v * np.sin(self.theta)
        self.y = self.y + v * np.cos(self.theta)

    def get_laser_distance(self, cars):
        # Get the laser ranges
        thetas = np.linspace(self.theta+np.deg2rad(self.min_angle_laser), self.theta+np.deg2rad(self.max_angle_laser), self.lasers)
        x = np.ones(len(thetas))*self.max_dist_laser+1.
        y = np.ones(len(thetas))*self.max_dist_laser+1.
        r = np.ones(len(thetas))*self.max_dist_laser+1.

        # For each laser check if it hits a car or wall
        for n, theta in enumerate(thetas):
            for car in cars:
                xt, yt, rt = self.calc_intersect_point_car(self.x, self.y, theta, car)
                if rt < r[n]:
                    x[n], y[n], r[n] = xt, yt, rt
            if r[n] > self.max_dist_laser:  # When car not hit by laser check for wall
                x[n], y[n], r[n] = self.calc_intersect_point_map(theta)

        noise = np.random.normal(0, self.noise, len(r))
        r = r + noise
        r = r.clip(0, self.max_dist_laser)

        return x, y, r

    def calc_intersect_point_car(self, x1, y1, theta, car):
        theta = theta % (2*np.pi)
        x2 = car.x - car.length / 2
        y2 = car.y + car.width / 2
        x3 = car.x + car.length / 2
        y3 = car.y - car.width / 2

        a = -np.tan(theta-(np.pi/2))
        if a == 0:
            a = 0.0001
        b = y1 - x1*a

        ycltest = a*x2 + b
        if y2 > ycltest > (y2 - car.width) and theta < np.pi and self.x < x2:
            ycl = ycltest
            xcl = x2
        else:
            ycl = np.inf
            xcl = np.inf

        ycrtest = a*x3 + b
        if (y3 + car.width) > ycrtest > y3 and theta > np.pi and self.x > x3:
            ycr = ycrtest
            xcr = x3
        else:
            ycr = np.inf
            xcr = np.inf

        xcttest = (y2-b)/a
        if (x2 + car.length) > xcttest > x2 and 3*np.pi/2 > theta > np.pi/2 and self.y > y2:
            yct = y2
            xct = xcttest
        else:
            yct = np.inf
            xct = np.inf

        xcbtest = (y3-b)/a
        if x3 > xcbtest > (x3 - car.length) and (np.pi/2 > theta or theta > 3*np.pi/2) and self.y < y3:
            ycb = y3
            xcb = xcbtest
        else:
            ycb = np.inf
            xcb = np.inf

        points = [xcl, ycl, xcr, ycr, xct, yct, xcb, ycb]
        r = np.zeros(4)
        for i in range(4):
            r[i] = self.pythagoras(points[i*2], points[(2*i)+1])

        minrange = np.min(r)
        ind = np.argmin(r)

        return points[ind*2], points[(2*ind)+1], minrange

    def calc_intersect_point_map(self, theta):
        theta = theta % (2*np.pi)

        xmax = self.map.width
        ymax = self.map.length

        a = np.tan(0.5*np.pi-theta)
        if a == 0:
            a = 0.0001
        b = self.y - self.x * a

        ytest = a * xmax + b
        if 0 <= ytest <= ymax and 0 <= theta <= np.pi:
            x = xmax
            y = ytest

        ytest = a * 0 + b
        if 0 <= ytest <= ymax and np.pi <= theta <= 2*np.pi:
            x = 0
            y = ytest

        xtest = (ymax - b) / a
        if 0 <= xtest <= xmax and (0 <= theta <= 0.5*np.pi or 1.5*np.pi <= theta <= 2*np.pi):
            x = xtest
            y = ymax

        xtest = (0 - b) / a
        if 0 <= xtest <= xmax and 0.5*np.pi <= theta <= 1.5*np.pi:
            x = xtest
            y = 0

        r = self.pythagoras(x, y)

        return x, y, r

    def pythagoras(self, a, b):
        c = np.sqrt((a - self.x) ** 2 + (b - self.y) ** 2)
        return c

class Road:
    def __init__(self):
        self.width = 40
        self.lanes = 2
        self.lanewidth = 3
        self.sidewalkwidth = 2
        self.length = self.lanes * self.lanewidth + 2*self.sidewalkwidth

class Car:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.width = 2
        self.length = 4
        self.startx = x
        self.starty = y

        self.speed = v

    def drive(self):
        self.x += self.speed

    def respawn(self):
        self.x = self.startx
        self.y = self.starty

