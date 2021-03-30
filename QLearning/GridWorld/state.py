import numpy as np

LAYOUTS = {
    0: {
        "size": (5, 5),
        "player_start": (2, 0),
        "objects": {
            "walls": [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (3, 1)],
            "traps": [(4, 4), (2, 3)],
            "goal": [(2, 4)],
        },
    },
    1: {
        "size": (4, 4),
        "player_start": (0, 0),
        "objects": {
            "walls": [(1, 1), (2, 3), (3, 0), (1, 3)],
            "traps": [],
            "goal": [(3, 3)],
        },
    },
    2: {
        "size": (5, 5),
        "player_start": (4, 0),
        "objects": {
            "walls": [],
            "traps": [(4, 1), (4, 2), (4, 3)],
            "goal": [(4, 4)],
        },
    },
}


class State:
    CELLS = {"empty": 0, "wall": 1, "trap": 2, "goal": 3}

    def __init__(self, layout_id=0):
        self.layout_id = layout_id
        self.layout = LAYOUTS[self.layout_id]
        self.size = self.size_x, self.size_y = self.layout["size"]
        self.player = np.zeros(self.size, dtype=np.bool)
        self.maze = np.zeros(self.size, dtype=np.int)
        self.reset()

    def reset(self):
        for i, (obj_name, pos_xy) in enumerate(self.layout["objects"].items(), start=1):
            for x, y in pos_xy:
                self.maze[x, y] = i
        self.player.fill(0)
        player_start_x, player_start_y = self.layout["player_start"]
        self.player[player_start_x, player_start_y] = 1

    def move(self, dx=0, dy=0):
        old_x, old_y = self.get_player_pos()

        new_x = dx + old_x
        if new_x >= self.size_x:
            new_x = self.size_x - 1
        if new_x < 0:
            new_x = 0

        new_y = dy + old_y
        if new_y >= self.size_y:
            new_y = self.size_y - 1
        if new_y < 0:
            new_y = 0

        if self.maze[new_x, new_y] == self.CELLS['wall']:
            new_x, new_y = old_x, old_y

        self.player[old_x][old_y] = 0
        self.player[new_x][new_y] = 1

    def get_state_transpose(self, x, y):
        return self.maze[y, x]

    def get_state(self, x, y):
        return self.maze[x, y]

    def set_state(self, x, y):
        old_x, old_y = self.get_player_pos()
        if self.maze[x, y] == self.CELLS['wall']:
            print('Cannot set the player on the wall')
        else:
            self.player[old_x][old_y] = 0
            self.player[x][y] = 1

    def get_player_pos(self):
        x, y = np.where(self.player)
        return int(x), int(y)

    def get_allowed_player_positions(self):
        x, y = np.where(self.maze == 0)
        return x, y

    def get_allowed_positions(self):
        x, y = np.where(self.maze != 1)
        return x, y

    def render(self):
        maze_copy = self.maze.copy()
        maze_copy[self.get_player_pos()] = 9
        print(maze_copy.astype(np.int8))

    def get_terminal_positions(self):
        x, y = np.where(np.logical_or(self.maze == self.CELLS['goal'], self.maze == self.CELLS['trap']))
        return x, y

    @property
    def shape(self):
        return self.maze.shape
