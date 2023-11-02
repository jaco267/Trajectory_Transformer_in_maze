import pdb
# pdb.set_trace()
import torch as tc
Ten = tc.Tensor
#0: no_op, 1: up, 2: left, 3: down, 4: right
move_map = tc.tensor([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])

def ravel_multi_index(x,y,gridsize):
    return gridsize*tc.min(x,gridsize-1)+tc.min(y,gridsize-1)   #flat_cell = 10x+y
def unravel_index(flat_cell,grid_size):
    return flat_cell//grid_size, flat_cell%grid_size
def random_choice_num(num_samples,p):
    #https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/6
    return  tc.multinomial(p,num_samples=num_samples,replacement=True) 
class ProcMaze:
    def __init__(self, grid_size=10, goal_reward=False,seed=None, timeout=None,device='cuda'):
        self.move_map = tc.tensor([[0, 0], [-1,0], [0,-1], [1,0], [0,1]],device=device)
        self.move_map_cpu = tc.tensor([[0, 0], [-1,0], [0,-1], [1,0], [0,1]],device="cpu")
        self.timeout = tc.tensor(timeout,device=device) if timeout else None
        self._num_actions = 5
        self.grid_size = tc.tensor(grid_size)  #!!   #ex. 10 -> 10x10 board
        self.channels ={'player':0,'goal':1,'wall':2,'empty':3}
        self.device = device
        self.seed = seed
    @property
    def action_map_dict(self):
        action_map = {
            0:'no_op',
            1:'up',
            2:'left',
            3:'down',
            4:'right'
        }
        return action_map 
    def reset(self)->any:
        def push(stack, top, x):
            stack[top] = x
            return stack, top + 1
        def pop(stack, top):
            top-=1
            return stack[top], top
        def neighbours(flatten_cell):
            #takes and flattened index, returns neighbours as (x,y) pair
            coord_array:tuple = unravel_index(flatten_cell,self.grid_size)
            return tc.tensor(coord_array)+self.move_map_cpu  # 10x+y -> [x,y] -> [[x,y],..,[x+1,y+1]]
        def can_expand(cell, visited):
            """ input cell = [x,y],or [x+1,y-1]  ...or[x+1,y+1]  """
            #* A neighbour can be expanded as long as it is on the grid, it has not been visited, and it's only visited neighbour is the current cell
            flat_cell = ravel_multi_index(cell[0],cell[1],self.grid_size) #flat_cell = 10x+y
            not_visited = tc.logical_not(visited[flat_cell])
            ns = neighbours(flat_cell)  #[[x,y]] -> [[x,y],..,[x+1,y+1]]
            ns_on_grid = tc.all(tc.logical_and(ns>=0,ns<=self.grid_size-1), dim=1)#[T,F,F,T,F]
            flat_ns = ravel_multi_index(ns[:,0],ns[:,1],self.grid_size) #[10x+y,10(x+1)+y,...10(x-1)-y]
            # Only count neighbours which are actually on the grid
            only_one_visited_neighbor = tc.equal(tc.sum(tc.logical_and(visited[flat_ns],ns_on_grid))
                                                 ,tc.tensor(1))
            on_grid = tc.all(tc.logical_and(cell>=0,cell<=self.grid_size-1))
            return tc.logical_and(tc.logical_and(not_visited,tc.tensor(only_one_visited_neighbor)),on_grid)
        wall_grid = tc.ones((self.grid_size, self.grid_size), dtype=bool)#[10,10]
        visited = tc.zeros(self.grid_size*self.grid_size, dtype=bool)#Visited node map#[]*100
        stack = tc.zeros(self.grid_size*self.grid_size, dtype=int)# []*100#big enough to hold every location in the grid, indices should be flattened to store here
        top = 0
        curr = random_choice_num(num_samples=2,p = tc.ones(self.grid_size)/self.grid_size)# (x,y)#Location of current cell being searched#ex. 7,5 
        flat_curr = ravel_multi_index(curr[0],curr[1],self.grid_size)#  7*10+5# curr[0]*self.grid_size+curr[1]  #*** ravel_numti_index
        wall_grid[curr[0], curr[1]]=False  #expand start point
        visited[flat_curr]=True
        stack, top = push(stack,top, flat_curr)
        while (top != tc.tensor(0) ): 
            curr, top = pop(stack,top)
            ns = neighbours(curr)
            """ [[2 3], [1 3], [2 2], [3 3], [2 4]]"""
            flat_ns = ravel_multi_index(ns[:,0],ns[:,1],self.grid_size)#[46 36 45 56 47]
            expandable = []
            for cell in ns:   
                expandable.append(can_expand(cell,visited))
            expandable = tc.stack(expandable)#expandable = [False False False False False]
            has_expandable_neighbour = tc.any(expandable)  #True False
            if tc.any(expandable) == False:
                p = tc.ones(len(expandable))/len(expandable)
            else:
                p = expandable/(tc.sum(expandable))  #[0.5,0.5,0,0,0]
            # This will all be used only conditioned on has_expandable neighbor
            _stack, _top = push(stack, top, curr)

            idx = tc.multinomial(p,num_samples=1,replacement=True)[0] 
            selected = flat_ns[idx]   # selected = np.random.choice(flat_ns,p=p)
            _stack, _top = push(_stack, _top, selected)
            _wall_grid = wall_grid
            # print(_wall_grid[0],"_wall_grid__a")
            # print("selected: ", selected,unravel_index(selected, self.grid_size))
            _wall_grid[unravel_index(selected, self.grid_size)]=False
            # print(_wall_grid[0],"_wall_grid__b")
            _visited = visited
            _visited[selected] = True

            stack =  _stack if has_expandable_neighbour else stack
            top = _top if has_expandable_neighbour else top
            # print(top,has_expandable_neighbour,)#_wall_grid)
            # pdb.set_trace()
            wall_grid = _wall_grid if has_expandable_neighbour else wall_grid
            visited = _visited if has_expandable_neighbour else visited
  
        flat_open = tc.logical_not(wall_grid.reshape(-1))
        pos = tc.multinomial(flat_open/tc.sum(flat_open),num_samples=1,replacement=True)[0]  # p = flat_open/sum(flat_open)
        pos:tuple = unravel_index(pos,self.grid_size) # in  [[x,y]]

        goal = random_choice_num( num_samples=1,p=flat_open/tc.sum(flat_open))[0]  #  len(100) p  -> sample 0~100
        goal:tuple = unravel_index(goal, self.grid_size)
        wall_grid[goal[0], goal[1]] = False
        wall_grid[pos[0], pos[1]]   = False
        t=0
        goal:Ten = tc.stack(goal).to(self.device)  #(a,b) -> [a,b]
        pos:Ten = tc.stack(pos).to(self.device)
        wall_grid:Ten = wall_grid.to(self.device)
        t:Ten = tc.tensor(t).to(self.device)
        env_state = goal, wall_grid, pos, t
        return env_state
  
    def step(self, action, env_state):
        goal, wall_grid, pos, t = env_state
        new_pos = tc.clip(pos+self.move_map[action], 0, self.grid_size-1)  # x, min ,max
        pos =  tc.where(tc.logical_not(wall_grid[new_pos[0], new_pos[1]]), new_pos, pos)
        
        terminal:Ten = tc.tensor(tc.equal(pos, goal))# Treminated if we reach the goal
        if(self.timeout is not None):
            terminal = tc.logical_or(terminal, t>=self.timeout)
        reward = -1;   t+=1;
        env_state = goal, wall_grid, pos, t
        return env_state, self.get_observation(env_state), reward, terminal, {}
  
    def get_observation(self, env_state):
        goal, wall_grid, pos, t = env_state
        obs = tc.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs[pos[0],pos[1],self.channels['player']] = True
        obs[goal[0],goal[1],self.channels['goal']] = True
        obs[:,:,self.channels['wall']] = wall_grid
        obs[:,:,self.channels['empty']] = tc.logical_not(wall_grid)
        return obs

    def num_actions(self):
        return self._num_actions
