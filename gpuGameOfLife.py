# Conway's game of life using PyCUDA
# This program is designed to be run on a GPU
# Due to CUDA limitations, the maximum allowable board has 
# 65536 x 32768 items

# imports
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

# constants
MATRIX_SIZE = 32768 # size of square board
BLOCK_SIZE = 32 # number of blocks
N_ITERS = 1 # number of iterations

class Game:
	def __init__(self, matrix, iters, block):
		self.size = matrix
		self.n_iters = iters
		self.n_blocks = matrix // block
		self.n_threads = block
		self.initialize_board()
		self.initialize_kernel()
		self.run()

	# Each cell is randomly set to either 1 (live) or 0 (dead)
	def initialize_board(self):
		self.board = np.random.randint(2, size = (self.size, self.size)).astype(np.float32)
		#self.board = np.zeros((self.size, self.size)).astype(np.float32)

	# PyCUDA implementation of Conway's game of life
	# Kernel is written in C
	def initialize_kernel(self):
		self.kernel_code = """
			// Update each cell of the grid
			// Any live cell with less than two live neighbors dies
			// Any live cell with two or three live neighbors lives
			// Any live cell with four or more live neighbors dies
			// Any dead cell with three neighbors becomes a live cell
			__global__ void life_step(float *board, float *board2)
			{{

				// Matrix size
				unsigned int m_size = {};
				unsigned int num_cells = {};

				// Column index of the element
				unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
				// Row index of the element
				unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
				// Thread ID in the board array
				unsigned int thread_id = y * m_size + x;

				// Game of life classically takes place on an infinite grid
				// I've used a toroidal geometry for the problem
				// The matrix wraps from top to bottom and from left to right
				unsigned int above = (thread_id - m_size) % num_cells;
				unsigned int below = (thread_id + m_size) % num_cells;
				unsigned int left;
				if (thread_id % m_size == 0) {{
					left = thread_id + m_size - 1;
				}} else {{
					left = thread_id - 1;
				}}
				unsigned int right;
				if (thread_id % m_size == m_size - 1) {{
					right = thread_id - m_size + 1;
				}} else {{
					right = thread_id + 1;
				}}
				unsigned int above_left;
				if (thread_id % m_size == 0) {{
					above_left = (thread_id - 1) % num_cells;
				}} else {{
					above_left = (thread_id - m_size - 1) % num_cells;
				}}
				unsigned int above_right;
				if (thread_id % m_size == m_size - 1) {{
					above_right = (thread_id - blockDim.x * m_size + 1) % num_cells;
				}} else {{
					above_right = (thread_id - m_size + 1) % num_cells;
				}}
				unsigned int below_left;
				if (thread_id % m_size == 0) {{
					below_left = (thread_id + blockDim.x * m_size - 1) % num_cells;
				}} else {{
					below_left = (thread_id + m_size - 1) % num_cells;
				}}
				unsigned int below_right;
				if (thread_id % m_size == m_size - 1) {{
					below_right = (thread_id + 1) % num_cells;
				}} else {{
					below_right = (thread_id + m_size + 1) % num_cells;
				}}

				unsigned int num_neighbors = board[above] + board[below] + board[left] + board[right] +
					board[above_left] + board[above_right] + board[below_left] + board[below_right];

				unsigned int live_and2 = board[thread_id] && (num_neighbors == 2);			// Live cell with 2 neighbors
				unsigned int live_and3 = board[thread_id] && (num_neighbors == 3);			// Live cell with 3 neighbors
				unsigned int dead_and3 = !(board[thread_id]) && (num_neighbors == 3);		// Dead cell with 3 neighbors
				board2[thread_id] = live_and2 || live_and3 || dead_and3;
			}}
		"""

		# Transfer CPU memory to GPU memory
		self.board_gpu = gpuarray.to_gpu(self.board)
		self.next_board = gpuarray.empty((self.size, self.size), np.float32)

		self.kernel = self.kernel_code.format(self.size, self.size * self.size)

		# Compile kernel code
		self.mod = SourceModule(self.kernel)

		# Get kernel function from compiled module
		self.game = self.mod.get_function('life_step')

	# Prints runtime kernel info
	def mem_info(self):
	    shared = self.game.shared_size_bytes
	    regs = self.game.num_regs
	    local = self.game.local_size_bytes
	    const = self.game.const_size_bytes
	    mbpt = self.game.max_threads_per_block
	    print("=MEM=\nLocal:{},\nShared:{},\nRegisters:{},\nConst:{},\nMax Threads/B:{}".format(local, shared, regs, const, mbpt))

	# Prints static device info
	def usage_info(self):
		(free,total) = drv.mem_get_info()
		print("Global memory occupancy:{}% free".format(free*100/total))

		for devicenum in range(drv.Device.count()):
		    device = drv.Device(devicenum)
		    attrs = device.get_attributes()

	    	#Beyond this point is just pretty printing
		    print("\n===Attributes for device {}".format(device))
		    for (key,value) in attrs.items():
		        print("{}:{}".format(str(key),str(value)))

	# Main driver function
	# Life_step function is called N_ITERS times
	def run(self):
		self.mem_info()
		self.usage_info()
		i = 0
		print('\nStarting board: \n', self.board_gpu)
		while i < N_ITERS: 
			self.game(
				# input
				self.board_gpu,
				# output
				self.next_board,
				# grid of n_blocks x n_blocks
				grid = (self.n_blocks, self.n_blocks, 1),
				# block of n_threads x n_threads
				block = (self.n_threads, self.n_threads, 1),
				)
			self.board_gpu, self.next_board = self.next_board, self.board_gpu
			i += 1
		print('Final board: \n', self.board_gpu)

if __name__ == '__main__':
	Game(MATRIX_SIZE, N_ITERS, BLOCK_SIZE)