#!/usr/bin/python

import os, sys, random, zlib, re
import chess
import chess.pgn
import numpy as np
import utils

# Moves must be by players at least this Elo to be included.
elo_threshold = -2000
test_set_one_in = 1000
#chunk_count = 22
chunk_count = 1

total_game_count = 0
total_move_count = 0
rejected_move_count = 0

class ZlibWriter:
	def __init__(self, path):
		self.f = open(path, "wb")
		self.comp = zlib.compressobj(9)

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
	global total_move_count, rejected_move_count

	game_outcome = {
		"1-0": 1,
		"1/2-1/2": 0,
		"0-1": -1,
	}[game.headers["Result"]]

	elos = [int(game.headers.get("WhiteElo", -1)), int(game.headers.get("BlackElo", -1))]

	board = game.board()
	#for move in game.main_line():
	if True:
		# Get the pre-move board features.
		features = utils.extract_features(board)
		#move_map = utils.encode_move(move)
		# Extract who is to win from this position.
		flip = 1 if board.turn else -1
		outcome = {0: "\0", 1: "\1", -1: "\xff"}[game_outcome * flip]

		# Write an entry in.
		current_player_elo = elos[board.turn == chess.BLACK]
		if current_player_elo >= elo_threshold:
			writer[0].write(features.tostring())
			#writer[1].write(move_map.tostring())
			writer[2].write(outcome)
			writer[0].advance()
			writer[1].advance()
			writer[2].advance()
			total_move_count += 1
		else:
			rejected_move_count += 1

		# Advance the game.
		#board.push(move)

		# XXX: For value_generation parsing, only extract the first position!
		#break

#	# Stream the moves out.
#	assert len(features) == len(moves)
#	features = np.array(features, dtype=np.int8)
#	moves = np.array(moves, dtype=np.int8)
#	outcomes = np.array(outcomes, dtype=np.int8)

def phony_read_pgn(f):
	# First, skip any blank lines before the [Event] header.
	while True:
		event_header = f.readline()
		if not event_header:
			return "EOF"
		if event_header != "\n":
			break
	assert event_header.startswith("[Event")

	# Next, read headers until we're done.
	fen = result = None
	while fen is None or result is None:
		line = f.readline()
		if line == "\n":
			assert False, "No entry!"
#			assert fen == result == None
#			fen = result = None
		if not line:
			return "EOF"
		if line.startswith("[Result "):
#			print "Hitting result:", `line`
			m = re.search('Result "([^"]*)"', line)
			result, = m.groups()
		elif line.startswith("[FEN "):
#			print "Hitting FEN:", `line`
			m = re.search('FEN "([^"]*)"', line)
			fen, = m.groups()
	# Now skip the body of the game.
	blank_lines = 2
	while blank_lines:
		line = f.readline()
		if not line:
			assert False, "Bad end of file between headers and moves!"
		if line == "\n":
			blank_lines -= 1

	b = chess.Board(fen)
	if b.result() != "*":
		print "Rejecting:", fen
		return None
	game = chess.pgn.Game()
#	print "Got:", `fen`, `result`
	game.setup(b)
	game.headers["Result"] = result
	return game

if __name__ == "__main__":
	output_directory = sys.argv[1]
	input_paths = sys.argv[2:]
	print "Processing %i files into %s" % (len(input_paths), output_directory)

	P = lambda path: os.path.join(output_directory, path)
	train_writer = \
		RoundRobinWriter([ZlibWriter(P("features_%03i.z" % i)) for i in xrange(chunk_count)]), \
		RoundRobinWriter([ZlibWriter(P("moves_%03i.z" % i)) for i in xrange(chunk_count)]), \
		RoundRobinWriter([ZlibWriter(P("outcomes_%03i.z" % i)) for i in xrange(chunk_count)])
	test_writer = \
		RoundRobinWriter([ZlibWriter(P("test_features.z"))]), \
		RoundRobinWriter([ZlibWriter(P("test_moves.z"))]), \
		RoundRobinWriter([ZlibWriter(P("test_outcomes.z"))])

	# Step 1: Read in and generate pairs for every game.
	for path in input_paths:
		print "Processing:", path
		f = open(path)
		this_file_game_count = 0
		while True:
			#game = chess.pgn.read_game(f)
			game = phony_read_pgn(f)
			if game == "EOF":
				break
			elif game is None:
				continue
			if total_game_count % test_set_one_in == 0:
				process_game(game, test_writer)
			else:
				process_game(game, train_writer)
			total_game_count += 1
			this_file_game_count += 1
			if total_game_count % 5000 == 0:
				print "Games: %6i  Positions: %8i  (Rejected: %8i))" % (total_game_count, total_move_count, rejected_move_count)
		print "Found %i games in %s" % (this_file_game_count, path)

	train_writer[0].close()
	train_writer[1].close()
	train_writer[2].close()
	test_writer[0].close()
	test_writer[1].close()
	test_writer[2].close()

	print "Total move count:", total_move_count
	print "Rejected moves:  ", rejected_move_count

