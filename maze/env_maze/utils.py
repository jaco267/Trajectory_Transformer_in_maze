import torch as tc

def play_with_env(env):
	returns=[]
	for i in range(100):
		print(f"=============game{i}==============")
		G=0
		env_state = env.reset()
		terminal = False
		while(not terminal):
			goal, wall_grid, pos, t = env_state
			# a = tc.tensor(tc.where(wall_grid,1,0))
			obs = tc.where(wall_grid,1,0)
			obs[pos[0] ,pos[1] ] = 8
			obs[goal[0],goal[1]] = 4
			print(f"{obs} current pos: {pos.numpy()}, \
	goal: {goal.numpy()} acc_reward: {G}")
			key = input('action 0:pass,w:up,a:left,s:down,d:right     ')
			if key == 'w':		action = 1
			elif key == 's':	action = 3
			elif key == 'a':	action = 2
			elif key == 'd':	action = 4
			else:	action = 0
			env_state, obs, reward, terminal, _ = env.step( action, env_state)
			
			G+=reward
		returns+=[G]
	avg_returns  = tc.mean(tc.tensor(returns))
	return avg_returns