import numpy as np
from copy import deepcopy

LENGTH = 3
NUM_ROWS = LENGTH
NUM_COLS = LENGTH
NUM_STATES = 3**(NUM_ROWS*NUM_COLS) # Num states can be x,o or empty for each position in the board

class Environment:
	def __init__(self, length_r = NUM_ROWS, length_c = NUM_COLS,print_board=False):
		self.board = np.zeros((length_r,length_c)); # Game board, init as empty
		self.x = -1 # player 1
		self.o = 1 # player 2
		self.winner = None
		self.ended = False
		self.print_board = print_board # Defines if board is printed doring games

	def is_empty(self,i,j):
		return self.board[i,j] == 0

	def reward(self, player):
		if not self.ended or self.winner is None:
			return 0
		elif self.winner == player:
			return 1
		else:
			return -1

	def get_state(self):
		# returns the current state, represented as an int
	    # from 0... NUM_STATES-1
	    # NUM_STATES = 3^(NUM_ROWS*NUM_COLS), since each cell can have 3 possible values - empty, x, o
	    # some states are not possible, e.g. all cells are x, but we ignore that detail
	    # this is like finding the integer represented by a base-3 number

		cell = 0
		state = 0

		for i in range(NUM_ROWS):
			for j in range(NUM_COLS):
				if self.board[i,j] == self.x:
					val = 1
				elif self.board[i,j] == self.o:
					val = 2
				else:
					val = 0

				state += (3^cell)*val
				cell+=1

		return state

	def is_game_over(self, recalculate=False):
		if not recalculate and self.ended:
			return self.ended
		# check rows
		for i in range(NUM_ROWS):
			for player in (self.x, self.o):
				if self.board[i].sum() == player*NUM_ROWS:
					self.winner = player
					self.ended = True
					return True

		# check columns
		for j in range(NUM_COLS):
			for player in (self.x, self.o):
				if self.board[:,j].sum() == player*NUM_COLS:
					self.winner = player
					self.ended = True
					return True

		# check diagonals
		for player in (self.x, self.o):
		  # top-left -> bottom-right diagonal
			if self.board.trace() == player*LENGTH:
				self.winner = player
				self.ended = True
				return True
		  # top-right -> bottom-left diagonal
			if np.fliplr(self.board).trace() == player*LENGTH:
				self.winner = player
				self.ended = True
				return True

		# check if draw
		if np.all((self.board == 0) == False):
		  # winner stays None
			self.winner = None
			self.ended = True
			return True

		# game is not over
		self.winner = None
		return False

	def is_draw(self):
		return self.ended and self.winner is None

	def draw_board(self):
		print()
		for i in range(NUM_ROWS):
			for j in range(NUM_COLS):
				# Print column dividers
				if j > 0:
					print('|',end='')

				# Gets each block value
				if self.board[i,j] == self.x:
					print('X',end='')
				elif self.board[i,j] == self.o:
					print('O',end='')
				else:
					print(' ',end='')

			# Prints line divider
			if i < NUM_ROWS-1:
				print()
				print('-+-+-')

		print()

	def play_game(self,p1,p2):
		current_player = None
		#loop until game is over
		while not self.is_game_over():
			if current_player == p1:
				current_player = p2
			else:
				current_player = p1

			if self.print_board: # draw the board before the player's action
				self.draw_board()

			# takes action
			current_player.take_action(env=self)

			# update state history for both players
			state = self.get_state()
			p1.update_state_history(state)
			p2.update_state_history(state)

		if self.print_board: # draw the final state of the board
			self.draw_board()

		# updates value function
		p1.update_value_function(env=self)
		p2.update_value_function(env=self)

		if self.print_board:
			print()
			if self.winner is None:
				print("It is a draw!")
			elif self.winner == self.x:
				print("X player won!")
			else:
				print("O player won!")
			print()

class Agent:
	def __init__(self, sym = 1, eps=0.1, alpha=0.5, states_results=None):
		self.eps = eps # probability of choosing random action instead of greedy
		self.alpha = alpha # learning rate
		self.state_history = []
		self.sym = sym
		if states_results is not None:
			self.init_value_function(states_results)

	def set_eps(self,eps):
		self.eps = eps

	def init_value_function(self, states_results):
		# initialize value function
		# states_results is an array of state_winner_tripples
		self.value_fun = np.zeros(NUM_STATES)

		for state, ended, winner in states_results:
			val = 0
			if ended:
				if winner == self.sym:
					val = 1
				# elif winner is None:
				# 	val = 0.5
				else:
					val = 0
			else:
				val = 0.5

			self.value_fun[state] = val

		return self.value_fun

	def update_value_function(self, env):
		# we want to BACKTRACK over the states, so that:
		# V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
		# where V(next_state) = reward if it's the most current state
		#
		# NOTE: we ONLY do this at the end of an episode

		if not env.ended:
			return

		reward = env.reward(self.sym)
		target = reward
		for prev in reversed(self.state_history):
			self.value_fun[prev] += self.alpha*(target - self.value_fun[prev])
			target = self.value_fun[prev]
		self.reset_history()

	def reset_history(self):
		self.state_history = []

	def update_state_history(self, state):
		# cannot put this in take_action, because take_action only happens
		# once every other iteration for each player
		# state history needs to be updated every iteration
		# s = env.get_state() # don't want to do this twice so pass it in
		self.state_history.append(state)

	def take_action(self,env):
		if env.print_board:
			print()
			print('Agent action:')
		# choose an action based on epsilon-greedy strategy
		r = np.random.rand()
		best_move = None # It is the number corresponding to the best next stage

		# get all possible valid actions
		valid_actions = []
		for i in range(NUM_ROWS):
			for j in range(NUM_COLS):
				if env.is_empty(i,j): # If cell is empty then it is a valid action
					valid_actions.append((i,j))

		action = None
		if r < self.eps: # takes a random action
			# Chooses a random action based on the possible available actions
			idx = np.random.choice(len(valid_actions))
			action = valid_actions[idx]

		else:
			# choose the best action based on current values of states
			# loop through all possible moves, get their values
			# keep track of the best value
			best_value = -10000
			for val_act in valid_actions:
				env.board[val_act[0],val_act[1]] = self.sym
				state = env.get_state()
				env.board[val_act[0],val_act[1]] = 0

				if self.value_fun[state] > best_value:
					action = val_act
					best_value = self.value_fun[state]
					best_move = state

		# takes the action
		if action is not None:
			env.board[action[0],action[1]] = self.sym

class Random_Action_Player:
	def __init__(self,sym):
		self.sym = sym

	def take_action(self, env):
		# get all possible valid actions
		valid_actions = []
		for i in range(NUM_ROWS):
			for j in range(NUM_COLS):
				if env.is_empty(i,j): # If cell is empty then it is a valid action
					valid_actions.append((i,j))

		idx = np.random.choice(len(valid_actions))
		action = valid_actions[idx]
		env.board[action[0],action[1]] = self.sym

	def update_value_function(self, env):
		pass

	def update_state_history(self, state):
		pass

class Default_Player:
	def __init__(self,sym):
		self.sym = sym

	def take_action(self, env):
		while True:
			# break if we make a legal move
			print()
			move = input("Enter coordinates i,j for your next move (i,j=[0...2]): ")
			i, j = move.split(',')
			i = int(i)
			j = int(j)
			if env.is_empty(i, j):
				env.board[i,j] = self.sym
				break
			else:
				print()
				print('Move is invalid. Type the coordinates (i,j) of an empty cell in the board.')

	def update_value_function(self, env):
		pass

	def update_state_history(self, state):
		pass

# Gets initial values for every possible state
def get_initial_states_results(env, i=0, j=0):
	# recursive function that will return all
	# possible states (as ints) and who the corresponding winner is for those states (if any)
	# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
	# impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
	# since that will never happen in a real game
	
	results = [] # results as an array of state_winner_triples

	for val in (0,env.x,env.o):
		env.board[i,j] = val

		if j == NUM_COLS - 1:
			# if j = NUM_COLS and i < NUM_ROWS then j=0 and i+=1
			if i == NUM_ROWS - 1:
				# break point
				# get the results for a state
				state = env.get_state()
				ended = env.is_game_over(recalculate=True)
				winner = env.winner
				results.append((state,ended,winner))
			else:
				results += get_initial_states_results(env, i+1, 0)
		else:
			results += get_initial_states_results(env, i, j+1)

	return results


if __name__ == '__main__':
	# init the environment
	e = Environment()

	# get state values before training
	states_results = get_initial_states_results(Environment())

	# init the players as agents
	a1 = Agent(eps = 1, sym = e.x, states_results=states_results)
	a2 = Agent(sym = e.o, states_results=states_results)

	print('Training Agent...')
	print()
	T = 50000 # number of games that will be played for training
	for t in range(T):
		if t%800==0:
			print('Training: {}% Complete'.format(t*100/T))
		
		if t < 1000: # min eps=0.01
			a1.set_eps(1/(t+1))
			a2.set_eps(1/(t+1))

		Environment().play_game(a1,a2)

	print('Training Complete')
	print()

	print('Begining validation of the agent trained by pairing it to an agent with a random approach...')
	print()

	rand_p = Random_Action_Player(e.o)

	G = 1000 # Number of games played for validation
	agent_wins = 0
	rand_player_wins = 0
	draws = 0
	for g in range(G):
		val_env = Environment()
		val_env.play_game(a1,rand_p)

		if val_env.winner == a1.sym:
			agent_wins += 1
		elif val_env.winner == rand_p.sym:
			rand_player_wins += 1
		else:
			draws += 1

	print('Validation results:')
	print('Trained Agent Wins: {}%;     Random Action Player Wins: {}%;      Draws:{}%'.format(agent_wins*100/G,rand_player_wins*100/G,draws*100/G))

	# print('Begining game against human player. Human is "X", angent is "O":')
	# print()

	# human = Default_Player(e.o)

	# while True:
	# 	Environment(print_board=True).play_game(a1,human)
	# 	print()
	# 	answer = input("Play again? [y/n]: ")
	# 	if answer and answer.lower()[0] == 'n':
	# 		break


