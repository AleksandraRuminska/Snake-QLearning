import csv

import numpy as np

import pandas as pd
from gym_snake.envs.objects import Apples
from gym_snake.envs.constants import ObjectColor
from gym_snake.envs.objects import Snake


def rotate_color(r, g, b, hue_rotation):
    if hue_rotation == 0:
        return r, g, b

    import colorsys, math
    hue, lightness, saturation = colorsys.rgb_to_hls(r / 255., g / 255., b / 255.)
    r2, g2, b2 = colorsys.hls_to_rgb((hue + hue_rotation) % 1, lightness, saturation)

    return int(math.floor(r2 * 255)), int(math.floor(g2 * 255)), int(math.floor(b2 * 255))


class BaseGrid:

    def __init__(
        self,
        np_random,
        width,
        height,
        num_snakes=1,
        initial_snake_size=2,
        num_apples=1,
        reward_apple=200,
        reward_none=0,
        reward_collision=-100,
        done_apple=False,
        always_expand=False
    ):
        assert width >= initial_snake_size
        assert height >= initial_snake_size
        assert initial_snake_size >= 1

        self.np_random = np_random
        self.num_snakes = num_snakes
        self.width = width
        self.height = height

        self.reward_apple = reward_apple
        self.reward_none = reward_none
        self.reward_collision = reward_collision

        self.done_apple = done_apple
        self.always_expand = always_expand
        self.forward_action = self.get_forward_action()

        self.snakes = None
        self.apples = Apples()
        self.all_done = False

        self.add_snakes(num_snakes, initial_snake_size)
        self.add_apples(num_apples)

    def move(self, actions):
        assert not self.all_done

        rewards = [self.reward_none] * self.num_snakes
        num_new_apples = 0

        # Move live snakes and eat apples
        if not self.always_expand:
            for snake, action in zip(self.snakes, actions):
                if snake.alive:
                    # Only contract if not about to eat apple
                    next_head = snake.next_head(action)
                    if next_head not in self.apples:
                        snake.contract()

        for i, snake, action in zip(range(self.num_snakes), self.snakes, actions):
            if not snake.alive:
                continue

            next_head = snake.next_head(action)
            if self.is_blocked(next_head):
                snake.kill()
                rewards[i] = self.reward_collision
            else:
                snake.expand(action)
                if next_head in self.apples:
                    if self.done_apple:
                        snake.kill()
                    self.apples.remove(next_head)
                    num_new_apples += 1
                    rewards[i] = self.reward_apple

        # If all agents are done, mark grid as done (and prevent future moves)
        dones = [not snake.alive for snake in self.snakes]
        self.all_done = False not in dones

        # Create new apples
        self.add_apples(num_new_apples)

        return rewards, dones

    def encode(self):
        return [self.encode_agent(i) for i in range(self.num_snakes)]

    def __eq__(self, other):
        self_encode = self.encode()
        other_encode = other.encode()

        if len(self_encode) != len(other_encode):
            return False

        for x, y in zip(self_encode, other_encode):
            if not np.array_equal(x, y):
                return False

        return True

    def get_forward_action(self):
        raise NotImplementedError()

    def add_snakes(self, num_snakes=1, initial_snake_size=4):
        self.snakes = []

        for i in range(num_snakes):
            x = self.np_random.randint(0, self.width)
            y = self.np_random.randint(0, self.height)
            direction = self.get_random_direction()

            rotated_green = rotate_color(0, 255, 0, i / num_snakes)
            rotated_blue = rotate_color(0, 0, 255, i / num_snakes)

            new_snake = Snake(x, y, direction, color_head=rotated_blue, color_body=rotated_green)
            self.snakes.append(new_snake)
            for _ in range(initial_snake_size):
                next_head = new_snake.next_head(self.forward_action)
                if self.is_blocked(next_head):
                    # give up and try again to place snakes
                    return self.add_snakes(num_snakes=num_snakes, initial_snake_size=initial_snake_size)

                new_snake.expand(self.forward_action)

    def add_apples(self, num_apples):
        num_open_spaces = self.width * self.height - sum(len(s) for s in self.snakes) - len(self.apples)
        num_new_apples = min(num_apples, num_open_spaces)
        for _ in range(num_new_apples):
            self._add_one_apple()

    def _add_one_apple(self):
        while True:
            p = (self.np_random.randint(0, self.width), self.np_random.randint(0, self.height))
            if self.is_blocked(p) or p in self.apples:
                continue

            self.apples.add(p)
            # print("Add apple p: ", p)
            break

    def is_blocked(self, p):
        x, y = p
        if x < 0 or x >= self.width:
            return True
        if y < 0 or y >= self.height:
            return True

        for snake in self.snakes:
            if p in snake:
                return True

        return False

    def encode_agent(self, agent_number):
        result = np.zeros((self.width, self.height, 3), dtype='uint8')

        for p in self.apples:
            result[p] = ObjectColor.apple

        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                body_color = ObjectColor.dead_body
                head_color = ObjectColor.dead_head
            elif i == agent_number:
                body_color = ObjectColor.own_body
                head_color = ObjectColor.own_head
            else:
                body_color = ObjectColor.other_body
                head_color = ObjectColor.other_head

            last_p = None
            for p in snake:
                result[p] = body_color
                last_p = p

            result[last_p] = head_color

        return result

    def get_random_direction(self):
        raise NotImplementedError()

    def get_renderer_dimensions(self, tile_size):
        raise NotImplementedError()

    def render(self, r, tile_size, cell_pixels):
        raise NotImplementedError()

    #[wallY, wallX, appleY, appleX, bodyUP, bodyDOWN, bodyLEFT, bodyRIGHT]
    #get current state
    def define_state(self):
        xs, ys = self.snakes[0]._deque[-1]
        # print("SNAKE HEAD Define state x,y: ", xs, ",", ys)

        #WALLS
        if xs== 0:
            #left
            wallX = '1'
        elif xs == self.width - 1:
            #right
            wallX = '2'
        else:
            #none
            wallX = '0'

        if ys== 0:
            #up
            wallY = '1'
        elif ys == self.height - 1:
            #down
            wallY = '2'
        else:
            #none
            wallY = '0'

        #APPLE
        xa, ya = self.apples._set.pop()
        pa = xa, ya
        self.apples.add(pa)
        # print(" APPLE Define state x,y: ", xa, ",", ya)

        if xs < xa:
            #right
            appleX = '2'
            # print("Apple on right ")
        elif xs > xa:
            #left
            appleX = '1'
            # print("Apple on left ")
        else:
            #none
            appleX = '0'

        if ys > ya:
            #up
            appleY = '1'
            # print("Apple up ")
        elif ys < ya:
            #down
            appleY = '2'
            # print("Apple down ")
        else:
            #none
            appleY = '0'

        #BODY
        if (xs, ys - 1) in self.snakes[0]:
            #up
            bodyUp = '1'
            # print("Body up ")
        else:
            #none
            bodyUp = '0'

        if (xs, ys + 1) in self.snakes[0]:
            #down
            bodyDown = '1'
            # print("Body down ")
        else:
            #none
            bodyDown = '0'

        if (xs - 1, ys) in self.snakes[0]:
            # left
            bodyLeft = '1'
            # print("Body left ")
        else:
            # none
            bodyLeft = '0'

        if (xs + 1, ys) in self.snakes[0]:
            # right
            bodyRight = '1'
            # print("Body right ")
        else:
            # none
            bodyRight = '0'

        # [wallX, wallY, appleX, appleY, bodyUP, bodyDOWN, bodyLEFT, bodyRIGHT]
        state = wallX+wallY+appleX+appleY+bodyUp+bodyDown+bodyLeft+bodyRight
        print("State: ", state)

        return state


    def combine_states(self):

        bodyStates = ['0', '1']
        appleWallStates = ['0', '1', '2']

        combinations = {}

        for wX in appleWallStates:
            for wY in appleWallStates:
                for aX in appleWallStates:
                    for aY in appleWallStates:
                        for bU in bodyStates:
                            for bD in bodyStates:
                                for bL in bodyStates:
                                    for bR in bodyStates:
                                        state = wX+wY+aX+aY+bU+bD+bL+bR

                                        # initial Q-table
                                        combinations[state] = [0, 0, 0, 0]
        k = 1

        self.save_table(combinations)


    def save_table(self, combinations):
        with open('qtable.csv', 'w') as Qtable:
            writer = csv.writer(Qtable)
            keys = list(combinations.keys())

            for c in range(len(keys)):
                actualKey = keys[c]
                line = ''
                k = 0
                for i in combinations[actualKey]:
                    if k == 3:
                        line = line + str(i) 
                    else:
                         line = line + str(i) + ', '
                    k = k+1
                
                bigline = list()
                bigline.append(actualKey)
                bigline.append(line)
                writer.writerow(bigline)
    

    def read_table():
        table = pd.read_csv('qtable.csv', dtype=str, header=None)

        combinations = {}

        state = table[0]
        value = table[1]

        for i in range(len(value)):
            valuex = value[i].split(",")
            statex = state[i]
            listVal = list()
            for i in valuex:
              listVal.append(float(i))  
            combinations[statex] = listVal

        return combinations


















