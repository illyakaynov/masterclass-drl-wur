import pygame
from QLearning.GridWorld.grid_env import (
    GridEnv,
)

from QLearning.GridWorld.dynamic_programming import DPAgent, DPRandomAgent
from QLearning.GridWorld.QAgent import DoubleQAgent, QAgent

from os.path import join

from QLearning.GridWorld.monte_carlo import MonteCarloValue, MonteCarloQValue, MonteCarloValueRandom

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)

GAMMA_STEP = 0.01
EPSILON_STEP = 0.05
ALPHA_STEP = 0.05

TOGGLE_LEARNING = "toggle_learning"
RESET_PLAYER = "reset_player"
RESET_VALUES = "reset_agent"
GAMMA_DOWN = "gamma_down"
GAMMA_UP = "gamma_up"
EPSILON_DOWN = "epsilon_down"
EPSILON_UP = "epsilon_up"
ALPHA_DOWN = "alpha_down"
ALPHA_UP = "alpha_up"
DO_STEP = "do_step"
TOGGLE_LABELS = "toggle_q_labels"
LEARN_ONE_ITERATION = 'learn_one_iteration'
POLICY_EVALUATION = 'policy_evaluation'
POLICY_IMPROVEMENT = 'policy_improvement'

# NOOP = 0
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class GridWorld:
    def __init__(self,
                 agent_type='q_learning',
                 # agent_type='value_iteration',
                 # agent_type='value_evaluation_random',
                 # agent_type='monte_carlo_value',
                 # agent_type='monte_carlo_q_value',
                 # agent_type='monte_carlo_value_evaluation_random',

                 layout_id=0,
                 screen_size=(800, 800),
                 display_logo=False):
        pygame.init()
        self.display_logo = display_logo
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.flip()
        pygame.display.set_caption(f"GridWorld-{agent_type}")
        self.player_size = 10

        env_params = {'layout_id': layout_id, 'terminate_after': 20}
        self.env = GridEnv(**env_params)
        state = self.env.get_state()
        if agent_type == 'q_learning':
            self.agent = QAgent(self.env.observation_space.n, self.env.action_space.n)
        elif agent_type == 'value_iteration':
            self.agent = DPAgent(GridEnv(**env_params))
        elif agent_type == 'monte_carlo_value':
            self.agent = MonteCarloValue(GridEnv(**env_params))
        elif agent_type == 'monte_carlo_q_value':
            self.agent = MonteCarloQValue(self.env.observation_space.n, self.env.action_space.n)
        elif agent_type == 'value_evaluation_random':
            self.agent = DPRandomAgent(GridEnv(**env_params))
        elif agent_type == 'monte_carlo_value_evaluation_random':
            self.agent = MonteCarloValueRandom(GridEnv(**env_params))

        self.margins = (
            self.screen_size[0] // (state.shape[0] + 1),
            self.screen_size[1] // (state.shape[1] + 1),
        )

        self.score = 0
        self.auto_learn = False

        self.display_labels = True
        if self.agent.get_type() == "q_value":
            self.display_q_labels = True
            self.display_values = False
        elif self.agent.get_type() == "value":
            self.display_q_labels = False
            self.display_values = True
        else:
            raise ValueError("Unknown type of the agent")

    def reset(self):
        self.score = 0
        self.env.reset()
        self.agent.reset()
        pass

    def draw_player(self):
        state = self.env.get_state()
        y, x = state.get_player_pos()
        robot_image = pygame.image.load(
            join("QLearning", "GridWorld", "images", "robot.png")
        )
        img_x, img_y = robot_image.get_size()
        pos = (int(x) + 1) * self.margins[0] - img_x // 2, (int(y) + 1) * self.margins[
            1
        ] - img_y // 2
        self.screen.blit(robot_image, pos)
        # pygame.draw.circle(self.screen, BLUE, pos, self.player_size)
        # pygame.display.update()

    def draw_q_labels(self, x, y, width, q_values, height=None):
        height = height or width
        offset = 5

        half = width // 2
        small = width // offset
        big = (offset - 1) * width // offset

        xy_pos = [
            [x + small, y + half],  # left
            [x + half, y + small],  # up
            [x + big, y + half],  # right
            [x + half, y + big],  # down
        ]

        max_q_value = max(q_values)
        for (x, y), q_value in zip(xy_pos, q_values):
            if max_q_value == q_value:
                self.draw_text(x, y, text="{:.3f}".format(q_value), color_text=RED)
            else:
                self.draw_text(x, y, text="{:.3f}".format(q_value))

    def draw_grid(self):
        cell_width = self.margins[0]
        cell_height = self.margins[1]
        state = self.env.get_state()
        for x in range(0, state.shape[0]):
            for y in range(0, state.shape[0]):
                posx = (x + 1) * self.margins[0] - 0.5 * self.margins[0]
                posy = (y + 1) * self.margins[1] - 0.5 * self.margins[1]
                if state.get_state_transpose(x, y) == 1:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, GREY, rect, 0)
                elif state.get_state_transpose(x, y) == 2:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, RED, rect, 0)
                    if self.display_q_labels or self.display_values:
                        self.draw_text(
                            posx + cell_width // 2,
                            posy + cell_width // 2,
                            text=str(self.env.REWARDS["trap"]),
                        )

                elif state.get_state_transpose(x, y) == 3:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, GREEN, rect, 0)
                    if self.display_q_labels or self.display_values:
                        self.draw_text(
                            posx + cell_width // 2,
                            posy + cell_height // 2,
                            text=str(self.env.REWARDS["goal"]),
                        )
                else:
                    rect = pygame.Rect(posx, posy, cell_width, cell_height)
                    pygame.draw.rect(self.screen, BLACK, rect, 1)
                    if self.display_labels:
                        if self.display_q_labels:
                            self.draw_q_labels(
                                posx,
                                posy,
                                width=self.margins[0],
                                q_values=self.agent.q_table[10 * y + x].tolist()
                            )
                        if self.display_values:
                            self.draw_value_label(
                                posx,
                                posy,
                                width=self.margins[0],
                                value=self.agent.value_fn[self.env.coord_to_state(y, x)],
                            )
                        if self.display_values or self.display_q_labels:
                            self.draw_arrow(posx + 2, posy + 2, action=self.agent.compute_greedy_action(self.env.coord_to_state(y, x)))

    def draw_value_label(self, posx, posy, width, value):
        offset = 5

        half = width // 2
        small = width // offset
        self.draw_text(posx + half, posy + small, text="{:.3f}".format(value), color_text=BLACK)



    def draw_arrow(self, posx, posy, action):
        action_name = self.env.ACTIONS[action]
        image = pygame.image.load(
            join("QLearning", "GridWorld", "images", f"{action_name}.png")
        )
        size = image.get_size()
        size = (int(size[0] / 2.), int(size[1] / 2.))
        image = pygame.transform.scale(image, size)
        self.screen.blit(image, (posx, posy))

    def draw_agent_params(self):

        text_values = []
        for attr in ['epsilon', 'gamma', 'alpha', 'episode_reward', 'return_', 'episode_count']:
            value = getattr(self.agent, attr, 0)
            if value != 0:
                t = '{}: {:.2f}'.format(attr, value)
                text_values.append(t)

        self.draw_text(
            50,
            40,
            pos="topleft",
            text=' ,'.join(text_values),
        )

    def draw_text(
        self,
        x,
        y,
        pos="center",
        text="",
        rotate_degrees=None,
        color_text=BLACK,
        color_background=WHITE,
    ):
        font = pygame.font.Font("freesansbold.ttf", 16)
        text = font.render(text, True, color_text, color_background)
        textRect = text.get_rect()
        if rotate_degrees:
            pygame.transform.rotate(text, rotate_degrees)

        if pos == "center":
            textRect.center = (x, y)
        if pos == "topleft":
            textRect.topleft = (x, y)
        self.screen.blit(text, textRect)

    def draw_logo(self):
        logo_image = pygame.image.load(
            join("QLearning", "GridWorld", "images", "logo_transparant.png")
        )
        size = logo_image.get_size()
        size = (int(size[0] / 4.1), int(size[1] / 4.1))
        logo_image = pygame.transform.scale(logo_image, size)
        pos = (
            self.screen_size[0] // 2 - logo_image.get_size()[0] // 2 - 70,
            self.screen_size[1] // 2 - logo_image.get_size()[1] // 2 - 50,
        )
        self.screen.blit(logo_image, pos)

    def draw_game(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_player()
        self.draw_agent_params()
        if self.display_logo:
            self.display_logo()
        pygame.display.update()

    def get_action_from_input(self):
        return_event = None
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = LEFT
                if event.key == pygame.K_UP:
                    action = UP
                if event.key == pygame.K_RIGHT:
                    action = RIGHT
                if event.key == pygame.K_DOWN:
                    action = DOWN

                if event.key == pygame.K_z:
                    return_event = DO_STEP

                if event.key == pygame.K_x:
                    return_event = LEARN_ONE_ITERATION

                if event.key == pygame.K_c:
                    return_event = POLICY_EVALUATION

                if event.key == pygame.K_v:
                    return_event = POLICY_IMPROVEMENT

                if event.key == pygame.K_SPACE:
                    return_event = TOGGLE_LEARNING

                if event.key == pygame.K_r:
                    return_event = RESET_PLAYER

                if event.key == pygame.K_t:
                    return_event = RESET_VALUES

                if event.key == pygame.K_q:
                    return_event = EPSILON_UP
                if event.key == pygame.K_a:
                    return_event = EPSILON_DOWN

                if event.key == pygame.K_w:
                    return_event = GAMMA_UP
                if event.key == pygame.K_s:
                    return_event = GAMMA_DOWN

                if event.key == pygame.K_e:
                    return_event = ALPHA_UP
                if event.key == pygame.K_d:
                    return_event = ALPHA_DOWN
                if event.key == pygame.K_l:
                    return_event = TOGGLE_LABELS

        return return_event, action

    def step(self):
        done = False

        while not done:
            # pygame.time.delay(100)
            done = False

            event, action = self.get_action_from_input()
            obs = self.env.get_obs()
            if event == TOGGLE_LEARNING:
                self.auto_learn = not self.auto_learn

            if event == TOGGLE_LABELS:
                self.display_labels = not self.display_labels

            if event == LEARN_ONE_ITERATION:
                self.agent.learn_one_iteration()

            if event == POLICY_EVALUATION:
                print(self.agent.policy_evaluation())

            if event == POLICY_IMPROVEMENT:
                print(self.agent.policy_improvement())


            if self.auto_learn:
                action = self.agent.compute_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.agent.update(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=reward,
                    done=done,
                )
            else:
                if event:
                    if event == RESET_PLAYER:
                        self.reset()
                    if event == RESET_VALUES:
                        self.agent.reset_values()

                    if event == EPSILON_UP:
                        self.agent.epsilon += EPSILON_STEP
                    if event == EPSILON_DOWN:
                        self.agent.epsilon -= EPSILON_STEP

                    if event == GAMMA_UP:
                        self.agent.gamma += GAMMA_STEP
                    if event == GAMMA_DOWN:
                        self.agent.gamma -= GAMMA_STEP

                    if event == ALPHA_UP:
                        self.agent.alpha += ALPHA_STEP
                    if event == ALPHA_DOWN:
                        self.agent.alpha -= ALPHA_STEP

                    if event == DO_STEP:
                        action = self.agent.compute_action(obs)
                if action is not None:
                    if action >= 0:
                        next_obs, reward, done, info = self.env.step(action)
                        self.agent.update(
                            obs=obs,
                            action=action,
                            next_obs=next_obs,
                            reward=reward,
                            done=done,
                        )
                        print(
                            "Time:",
                            self.env.time,
                            "Score:",
                            self.score,
                            "Obs:",
                            next_obs,
                            "reward: ",
                            reward,
                        )
                        self.score += reward

            self.draw_game()
            if done:
                self.reset()
                done = False
        return done


if __name__ == "__main__":
    env = GridWorld(layout_id=0)
    env.reset()
    done = env.step()
# print(env.agent.q_table)
