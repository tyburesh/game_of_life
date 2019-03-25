
# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
MATRIX_SIZE = 100 # board dimensions
INTERVAL = 250 # in milliseconds

class Game:
	def __init__(self, size):
		self.size = size
		self.initialize_board(size)
		self.run()

	# Each cell is randomly set to either 1 (live) or 0 (dead)
	def initialize_board(self, size):
		self.board = np.random.randint(2, size = (self.size, self.size)).astype(np.float32)

	# Update each cell of the grid
	# Any live cell with less than two live neighbors dies
	# Any live cell with two or three live neighbors lives
	# Any live cell with four or more live neighbors dies
	# Any dead cell with three neighbors becomes a live cell
	def step(self, frame, img):
		self.next_board = self.board.copy()
		for i in range(self.size):
			for j in range(self.size):

				# Game of life classically takes place on an infinite grid
				# I've used a toirodal geometry for the problem
				# The matrix wraps from top to bottom and from left to right
				num = int(self.board[(i-1)%self.size][(j-1)%self.size] + \
					self.board[(i+1)%self.size][(j+1)%self.size] + \
					self.board[(i-1)%self.size][(j+1)%self.size] + \
					self.board[(i+1)%self.size][(j-1)%self.size] + \
					self.board[i][(j-1)%self.size] + \
					self.board[i][(j+1)%self.size] + \
					self.board[(i-1)%self.size][j] + \
					self.board[(i+1)%self.size][j])

				# Live cell
				if self.board[i][j] == 1:
					if (num < 2 or num > 3):
						self.next_board[i][j] = 0

				# Dead cell
				else:
					if (num == 3):
						self.next_board[i][j] = 1

		# Update animation and save updated board
		img.set_data(self.next_board)
		self.board[:] = self.next_board[:]
		return img

	# Main driver function
	# Setup animation and begin game
	def run(self):
		fig, ax = plt.subplots()
		img = ax.imshow(self.board, interpolation='nearest') 
		ani = animation.FuncAnimation(fig, self.step, fargs = (img,), frames = None, interval = INTERVAL) 
		plt.show()


if __name__ == '__main__':
	Game(MATRIX_SIZE)



