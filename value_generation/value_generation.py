#!/usr/bin/python

import os, sys, time, json, re, random, subprocess
import chess, chess.pgn

STOCKFISH_PATH = "stockfish"
STOCKFISH_HASH_MB = 1024
STOCKFISH_MS_PER_MOVE = 500

all_spots = []
for path in sys.argv[1:]:
	if not os.path.exists(path + ".index"):
		continue
	print "Loading:", path
	fd = open(path)
	spots = json.loads(open(path + ".index").read())
	for spot in spots:
		all_spots.append((fd, spot))

print "Parsed in:", len(all_spots)

def sample_game():
	fd, spot = random.choice(all_spots)
	fd.seek(spot)
	game = chess.pgn.read_game(fd)
	return game

def extract_move(process, board, movetime):
	process.stdin.write("position fen %s\ngo movetime %i\n" % (board.fen(), movetime))
	process.stdin.flush()
	for _ in xrange(movetime / 10):
		while True:
			line = process.stdout.readline()
			m = re.search("bestmove ([^ \\n]+)", line)
			if m:
				return chess.Move.from_uci(m.groups()[0])
			if not line:
				break
		time.sleep(0.020)
	print "WARNING: Stockfish timed out on:", board.fen()

def generate_playout(output_file):
	# First, we sample a game.
	g = sample_game()
	# Next, we choose a random ply count.
	board = chess.Board()
	all_moves = list(g.main_line())
	# Skip very short games.
	if len(all_moves) < 10:
		return False
	# We truncate at some random point, always truncating off the last move (which might be a mate).
	truncate = random.randrange(len(all_moves))
	for m in all_moves[:truncate]:
		board.push(m)
	legal_moves = list(board.legal_moves)
	# Assert that there is at least one legal move!
	if not legal_moves:
		print "WARNING: Illegal state in:"
		print g
		print truncate, board.fen()
		return False
	# Make a single uniformly random move.
	board.push(random.choice(legal_moves))
	# Clear all 3 move repetition history intentionally, because Stockfish won't have access to it.
	board = chess.Board(board.fen())
	output_game = chess.pgn.Game()
	output_game.setup(board)
	# Begin playing out this game.
	process = subprocess.Popen([STOCKFISH_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	process.stdin.write("setoption name Hash value %i\n" % (STOCKFISH_HASH_MB,))
	process.stdin.flush()
	all_generated_moves = []
	while board.result(claim_draw=True) == "*":
		move = extract_move(process, board, STOCKFISH_MS_PER_MOVE)
		all_generated_moves.append(move)
		board.push(move)
	process.kill()
	# Finally, write out the game.
	for k in ("White", "Black", "WhiteElo", "BlackElo", "Time", "FICSGamesDBGameNo"):
		if k in g.headers:
			output_game.headers[k] = g.headers[k]
	output_game.headers["Result"] = board.result(claim_draw=True)
	output_game.add_line(all_generated_moves)
	print >>output_file, output_game
	print >>output_file
	output_file.flush()
	return True

output_path = "out/games-hs%i-mt%i-%s.pgn" % (STOCKFISH_HASH_MB, STOCKFISH_MS_PER_MOVE, os.urandom(16).encode("hex"))
output_file = open(output_path, "a+")

success = 0
while True:
	success += generate_playout(output_file)
	if success % 10 == 0:
		print "Generated: %i" % success

