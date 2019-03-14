
# Conway's game of life using PyCUDA
# This program is designed to be run on GPU

# imports
import numpy as np
import matplotlib.pyplot as plt
#import pycuda.autoinit
#import pycuda.driver as drv
#import pycuda.gpuarray as gpuarray
#from pycuda.compiler import SourceModule

# constants
MATRIX_SIZE = 50 # board dimensions
RUNTIME = 5 # in seconds

class Game:
	def __init__(self, size):
		self.size = size
		self.initialize_board(size)
		self.initialize_kernel()
		self.reveal()

	def initialize_board(self, size):
		self.board = np.random.randint(2, size = (MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32)

	def initialize_kernel(self):
		kernel_template = """

		"""
		#self.board_gpu = gpuarray.to_gpu(self.board)
		#self.kernel_code = kernel_template % {
		#	'MATRIX_SIZE': MATRIX_SIZE
		#}
		#self.mod = SourceModule(kernel_template)
		#self.game = mod.get_function("")
		pass

	def reveal(self):
		plt.imshow(self.board, cmap = 'binary')
		plt.show(block = False)
		plt.pause(RUNTIME)
		plt.close()

if __name__ == '__main__':
	Game(MATRIX_SIZE)



