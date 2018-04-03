from breakout_env.utils import GameObject, Bricks, FRAME_X, aabb, FRAME_Y, actions_meaning, obs_base, render_bb, digits, \
    BRICKS_COLS, BRICKS_SIZE
import numpy as np


class BreakoutState(object):

    def __init__(self, conf):
        self.conf = conf
        self.reset()

    def reset(self):
        self.score = 0
        self.reward = 0
        self.step_count = 0
        self.lifes = self.conf["lifes"]
        self.max_step = self.conf["max_step"]
        self.terminal = False
        self.started = False
        # self.ball = GameObject(self.conf['ball_pos'], self.conf['ball_size'], self.conf['ball_color'])
        # self.ball_v = list(self.conf['ball_speed'])
        self.ball, self.ball_v = self._get_random_ball()

        self.paddle = GameObject([189, 70], [4, self.conf['paddle_width']], self.conf['paddle_color'], self.conf['catch_reward'])
        self.paddle_v = [0, self.conf['paddle_speed']]
        self.bricks = Bricks(self.conf['bricks_rows'], BRICKS_COLS, BRICKS_SIZE, self.conf['bricks_color'],
                             self.conf['bricks_reward'])


    def _get_random_ball(self):

        # if random.random()<0.5:
        #
        #     ball = GameObject([100, 10], self.conf['ball_size'], self.conf['ball_color'])
        #     random_v = list(self.conf['ball_speed'])
        #     random_v[1] = abs(random_v[1])
        #
        # else:
        #
        #     ball = GameObject([100, 120], self.conf['ball_size'], self.conf['ball_color'])
        #     random_v = list(self.conf['ball_speed'])
        #     random_v[1] = -abs(random_v[1])

        ball = GameObject([100, 10], self.conf['ball_size'], self.conf['ball_color'])
        random_v = list(self.conf['ball_speed'])
        random_v[1] = abs(random_v[1])

        ball_v = random_v


        # ball = GameObject([100, np.random.randint(10, 120)], self.conf['ball_size'], self.conf['ball_color'])
        # random_v = list(self.conf['ball_speed'])
        # if random.random()<0.5:
        #     random_v = random_v[::-1]
        # random_v[1] *= np.random.choice([-1, 1])
        # ball_v = random_v

        return ball, ball_v

    def _edge_collision(self):
        bb1 = self.ball.boundingbox
        if aabb(bb1, [0, 999, 0, FRAME_X[0]]):  # Left edge
            self.ball_v = [self.ball_v[0], -self.ball_v[1]]
            self.ball.translate([0, 2 * self.ball_v[1]])
        elif aabb(bb1, [0, 999, FRAME_X[1], 999]):  # Right edge
            self.ball_v = [self.ball_v[0], -self.ball_v[1]]
            self.ball.translate([0, 2 * self.ball_v[1]])
        elif aabb(bb1, [0, FRAME_Y[0], 0, 999]):  # Top edge
            self.ball_v = [-self.ball_v[0], self.ball_v[1]]
            self.ball.translate([2 * self.ball_v[0], 0])
        elif aabb(bb1, [FRAME_Y[1], 999, 0, 999]):  # Bottom edge
            self.lifes -= 1
            self.terminal = self.started and self.lifes == 0
            # self.ball = GameObject(self.conf['ball_pos'], self.conf['ball_size'], self.conf['ball_color'])
            # self.ball_v = self.conf['ball_speed']
            self.ball, self.ball_v = self._get_random_ball()


    def _paddle_collision(self):
        bb1 = self.ball.boundingbox
        if aabb(bb1, self.paddle.boundingbox):
            vy, vx = self.ball_v
            # print(self.ball_v)
            bigger, smaller = (vx, vy) if abs(vx) > abs(vy) else (vy, vx)
            bigger, smaller = abs(bigger), abs(smaller)

            pos_x = (self.ball.pos[1] - (self.paddle.pos[1] + self.paddle.size[1] / 2)) / (self.paddle.size[1] / 2)
            # print(pos_x, self.ball.pos[1], self.paddle.pos[1] + self.paddle.size[1]/2, self.ball.pos[1] - self.paddle.pos[1] + self.paddle.size[1]/2)
            vx_sign = 1 if vx > 0 else -1
            # pos_x = -vx_sign if pos_x < -vx_sign*0.3 else vx_sign
            pos_x *= vx_sign
            vx, vy = (-bigger, smaller) if pos_x >= -1.0 and pos_x < -0.75 else \
                     (-smaller, bigger) if pos_x >= -0.75 and pos_x < -0.25 else \
                     (vx_sign * vx, vy) if pos_x >= -0.25 and pos_x < 0.25 else \
                     (smaller, bigger) if pos_x >= 0.25 and pos_x < 0.75 else \
                     (bigger, smaller)

            # print(vx, vy)
            vx = vx_sign * vx

            # self.ball_v = [-self.ball_v[0], self.ball_v[1]]
            self.ball_v = [-vy, vx]

            self.ball.translate([2 * self.ball_v[0], self.ball_v[1]])

        # Re-new bricks if all clear
        if len(self.bricks.deleted_indexes) == len(self.bricks.bricks):
            self.terminal = True
            self.bricks = Bricks(self.conf['bricks_rows'], BRICKS_COLS, BRICKS_SIZE, self.conf['bricks_color'],
                                     self.conf['bricks_reward'])

        return 0

    def _bricks_collision(self):
        bb1 = self.ball.boundingbox
        outer_bb = self.bricks.outer_boundingbox

        # Early return if not inside outer bounding box
        if not aabb(bb1, outer_bb):
            return 0

        x2 = (bb1[2] + bb1[3]) / 2.0
        # y2 = (bb1[0] + bb1[1]) / 2.0
        x1 = x2 - self.ball_v[1]
        # y1 = y2 - self.ball_v[0]

        for idx, brick in enumerate(self.bricks.bricks):
            if idx in self.bricks.deleted_indexes:
                continue
            bb2 = brick.boundingbox

            if not aabb(bb1, bb2):
                continue

            if (x1 < bb2[2] and x2 > bb2[2]) or (x1 > bb2[3] and x2 < bb2[3]):
                self.ball_v = [self.ball_v[0], -self.ball_v[1]]
                self.ball.translate([0, 2 * self.ball_v[1]])
                # continue
            else:
                self.ball_v = [-self.ball_v[0], self.ball_v[1]]
                self.ball.translate([2 * self.ball_v[0], 0])

            r = brick.reward
            # del self.bricks.bricks[idx]
            self.bricks.deleted_indexes.append(idx)
            self.bricks.bricks_status_matrix[idx//BRICKS_COLS, idx % BRICKS_COLS] = 0
            return r

        return 0

    def encode(self):
        obs_type = self.conf["observation"]
        res = None
        if obs_type == "number":
            res = self.encode_number()
        elif obs_type == "vector":
            res = self.encode_vector()
        elif obs_type == "pixels":
            res = self.encode_pixels()
        elif obs_type == "number_discretized":
            res = self.encode_number_discretized()
        else:
            raise Exception
        return res


    def encode_number(self):
        """encode the state as a number"""
        i = self.paddle.pos[1]
        i *= 160
        i += self.ball.pos[0]
        i *= 210
        i += self.ball.pos[1]
        i *= 160
        i += self.ball_v[0]
        i *= 4
        i += self.ball_v[1]
        i *= 4
        return i

    def encode_number_discretized(self):
        """encode the state as a number with an efficient state discretization"""
        paddle_x = self.paddle.pos[1]//2 #4
        ball_x = self.ball.pos[1]//2
        ball_y = self.ball.pos[0]//2
        # ball_y = 0 if ball_y in range(0, 93) else (ball_y-93)//2 if ball_y in range(93, FRAME_Y[1]) else None
        # if ball_y is None:
        #     raise Exception

        ball_v = self.ball_v
        velocities = [-2, -1, 1, 2]
        ball_vy = velocities.index(ball_v[0])
        ball_vx = velocities.index(ball_v[1])


        i = paddle_x
        i *= 160//2
        i += ball_x
        i *= 160//2
        # i += ball_vy
        # i *= 4
        # i += ball_vx
        # i *= 4

        i += ball_y

        # i = paddle_x - ball_x + 160
        # print(paddle_x, ball_x, paddle_x-ball_x, i)
        # i*=320
        # i+= ball_y

        return i


    def encode_vector(self):
        # paddle_x, ball_y, ball_x, ball_speed_y, ball_speed_x
        return np.asarray([self.paddle.pos[1], self.ball.pos[0], self.ball.pos[1], self.ball_v[0], self.ball_v[1]])

    def encode_pixels(self):
        obs = np.copy(obs_base)

        # Draw paddle
        paddle_bb = self.paddle.boundingbox
        obs[paddle_bb[0]:paddle_bb[1], paddle_bb[2]:paddle_bb[3]] = self.paddle.color

        # Draw bricks
        for idx, brick in enumerate(self.bricks.bricks):
            if idx in self.bricks.deleted_indexes:
                continue
            bb = brick.boundingbox
            obs[bb[0]:bb[1], bb[2]:bb[3]] = brick.color

        # Draw ball
        if self.started:
            ball_bb = self.ball.boundingbox
            obs[ball_bb[0]:ball_bb[1], ball_bb[2]:ball_bb[3]] = self.ball.color

        # Draw info (score, lifes)
        life_bb = render_bb['lifes']
        obs[life_bb[0]:life_bb[1], life_bb[2]:life_bb[3]] = digits[self.lifes]

        scores_bb = render_bb['scores']
        scores = [(self.score // 10 ** i) % 10 for i in range(2, -1, -1)]
        for idx, bb in enumerate(scores_bb):
            obs[bb[0]:bb[1], bb[2]:bb[3]] = digits[scores[idx]]

        return obs

    def _next_state(self, action):
        previous_lives = self.lifes
        act = actions_meaning[action]

        if act == 'RIGHT' and self.paddle.boundingbox[3] + self.paddle_v[1] < FRAME_X[1] + 8:
            self.paddle.translate(self.paddle_v)
        if act == 'LEFT' and self.paddle.boundingbox[2] - self.paddle_v[1] > FRAME_X[0] - 8:
            self.paddle.translate([-x for x in self.paddle_v])

        if self.started:
            self.ball.translate(self.ball_v)

            # Check collision
            self._edge_collision()
            self.reward = self._paddle_collision()
            self.reward += self._bricks_collision()
            self.score += self.reward

        # Check is FIRE
        if actions_meaning[action] == 'FIRE':
            self.started = True

        self.step_count += 1
        if self.step_count >= self.max_step:
            self.terminal = True

        if self.lifes < previous_lives:
            self.reward = -10
            # self.reward = 0
        return self

