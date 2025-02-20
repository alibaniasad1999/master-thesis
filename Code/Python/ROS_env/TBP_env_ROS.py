import numpy as np
# three body problem env
class ThreeBodyEnv:
    def __init__(self, trajectory_, error_range=0.1, final_range=0.1):
        self.trajectory = trajectory_
        self.state = np.zeros(4)
        self.dt = 0.01
        self.mu = 0.012277471
        # self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.position = trajectory_[0]
        self.steps = 0
        self.max_steps = 1000
        self.final_range = final_range
        self.error_range = error_range
        self.reward_range = (-float('inf'), float('inf'))
        self.render_logic = False
        # second player
        self.second_player = False
        self.reset()

    def step(self, action):
        x = self.position[0]
        y = self.position[1]
        xdot = self.position[2]
        ydot = self.position[3]

        # force = action[0] * env.state[2:] + action[1] * env.state[:2]
        a_x = action[0]/10
        a_y = action[1]/10
        # add second player action
        a_x_2 = action[2]/10 if self.second_player else 0
        a_y_2 = action[3]/10 if self.second_player else 0


        r1 = np.sqrt((x+self.mu)**2 + y**2)
        r2 = np.sqrt((x-1+self.mu)**2 + y**2)

        xddot = 2*ydot + x -(1-self.mu)*((x+self.mu)/(r1**3)) - self.mu*(x-1+self.mu)/(r2**3) + a_x + a_x_2
        yddot = -2*xdot + y - (1-self.mu)*(y/(r1**3)) - self.mu*y/(r2**3) + a_y + a_y_2

        x = x + xdot*self.dt
        y = y + ydot*self.dt

        xdot = xdot + xddot*self.dt
        ydot = ydot + yddot*self.dt

        self.position = np.array([x, y, xdot, ydot])

        self.steps += 1

        self.position2state()
        #
        # # plot position
        # if self.render_logic:
        #     plt.plot(x, y, 'ro')
        #     plt.plot(self.trajectory[:,0], self.trajectory[:,1])
        #     plt.show()

        reward = 1 - np.linalg.norm(self.state, axis=0) + self.steps /1000 - (a_x/10)**2 - (a_y/10)**2 + (a_x_2/10)**2 + (a_y_2/10)**2
        done = self.steps >= self.max_steps
        if np.linalg.norm(self.position[0:2] - self.trajectory[-1, 0:2]) < self.final_range:
            done = True
            reward = 10
            print("done 🥺")
        if self.steps > 1000:
            done = True
            reward = -10
            print("end time")
        if self.error_calculation() > self.error_range:
            done = True
            reward = -10
            print("too much error 🥲😱")


        # print(self.state, reward, done, self.position)
        return 1000*self.state, reward, done, False, self.position

    def position2state(self):
        # find the nearest point from position to trajectory
        distance = np.linalg.norm(self.trajectory - self.position, axis=1)
        nearest_idx = np.argmin(distance)
        # estate = position - nearest(index)
        self.state = self.position - self.trajectory[nearest_idx]

    def error_calculation(self):
        normalized_error = np.copy(self.state[0:2]) * np.array([1, 10])
        return np.linalg.norm(normalized_error)

    def reset(self,
              *,
              seed: 5 = None,
              return_info: bool = False,
              options: 6 = None):
        self.position = self.trajectory[0]
        self.steps = 0
        self.position2state()
        return 10000*self.state, {}