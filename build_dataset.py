#!/usr/bin/python

import os, sys, random, zlib
import chess
import chess.pgn
import numpy as np
import utils

test_set_one_in = 1000
chunk_count = 12
output_directory = sys.argv[1]
input_paths = sys.argv[2:]

print "Processing %i files into %s" % (len(input_paths), output_directory)

total_game_count = 0
total_move_count = 0

class ZlibWriter:
	def __init__(self, path):
		self.f = open(path, "wb")
		self.comp = zlib.compressobj()

	def write(self, s):
		self.f.write(self.comp.compress(s))

	def close(self):
		self.f.write(self.comp.flush())
		self.f.close()

	def __del__(self):
		self.close()

class RoundRobinWriter:
	def __init__(self, writer_list):
		self.writer_list = writer_list
		self.index = 0

	def write(self, s):
		self.writer_list[self.index].write(s)

	def advance(self):
		self.index += 1
		self.index %= len(self.writer_list)

	def close(self):
		for w in self.writer_list:
			w.close()

def process_game(game, writer):
	global total_move_count
	board = game.board()
	features  = []
	moves = []
	for move in game.main_line():
		# Get the pre-move board features.
		features.append(utils.extract_features(board))
		moves.append(utils.encode_move(move))
		board.push(move)
		total_move_count += 1
	# Stream the moves out.
	assert len(features) == len(moves)
	features = np.array(features, dtype=np.int8)
	moves = np.array(moves, dtype=np.int8)
	writer[0].write(features.tostring())
	writer[1].write(moves.tostring())
	writer[0].advance()
	writer[1].advance()

if __name__ == "__main__":
	P = lambda path: os.path.join(output_directory, path)
	train_writer = \
		RoundRobinWriter([ZlibWriter(P("features_%03i.z" % i)) for i in xrange(chunk_count)]), \
		RoundRobinWriter([ZlibWriter(P("moves_%03i.z" % i)) for i in xrange(chunk_count)])
	test_writer = \
		RoundRobinWriter([ZlibWriter(P("test_features.z"))]), \
		RoundRobinWriter([ZlibWriter(P("test_moves.z"))])

	# Step 1: Read in and generate pairs for every game.
	for path in input_paths:
		print "Processing:", path
		f = open(path)
		while True:
			game = chess.pgn.read_game(f)
			if game is None:
				break
			if total_game_count % test_set_one_in == 0:
				process_game(game, test_writer)
			else:
				process_game(game, train_writer)
			total_game_count += 1
			if total_game_count % 5000 == 0:
				print "Games: %6i  Positions: %8i" % (total_game_count, total_move_count)

	train_writer[0].close()
	train_writer[1].close()
	test_writer[0].close()
	test_writer[1].close()

	print "Total move count:", total_move_count

