#!/usr/bin/python

import os, sys, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir("/home/snp/proj/TinyMLChess")

import chess
import engine
import logging

LOG_PATH = "/tmp/chess.log" 
logging.basicConfig(
	format="%(message)s",
	filename=LOG_PATH,
	filemode="a+",
	level=logging.DEBUG,
)
logging.debug("Starting log.")

go_options = {
	"wtime": "wtime ([0-9]+)",
	"btime": "btime ([0-9]+)",
}

def parse_options(options, line):
	results = {}
	for k, v in options.iteritems():
		m = re.search(v, line)
		if m:
			results[k], = m.groups()
	return results

default_ms_per_move = 5000

def main():
	print "TinyMLChess"
	sys.stdout.flush()
	board = chess.Board()

	manager = engine.MCTSEngine()

	while True:
		line = raw_input()
		logging.debug("Got: %r" % (line,))
		if line == "quit":
			exit()
		elif line == "uci":
			print "id name TinyMLChess"
			print "id author Peter Schmidt-Nielsen"
			print "option name Mode type spin default 1 min 1 max 3"
			print "uciok"
		elif line == "isready":
			print "readyok"
		elif line == "ucinewgame":
			board = chess.Board()
		elif line.startswith("position fen"):
			_, _, fen = line.split(" ", 2)
			board = chess.Board(fen)
			manager.set_state(board)
			logging.debug(">>> Setting board:\n%s" % board)
		elif line.startswith("position startpos moves"):
			_, _, _, moves = line.split(" ", 3)
			board = chess.Board()
			for m in moves.split():
				board.push(chess.Move.from_uci(m))
			manager.set_state(board)
			logging.debug(">>> Setting board:\n%s" % board)
		elif line == "go" or line.startswith("go "):
			option_values = parse_options(go_options, line)
			logging.debug("Got option values: %r" % (option_values,))
			our_time = float(option_values.get({True: "wtime", False: "btime"}[board.turn], default_ms_per_move)) * 1e-3
			logging.debug(">>> Thinking with %.4f seconds." % (our_time,))
			move = manager.genmove(our_time)
			print "bestmove", move

#			moves = engine.score_moves(board)
#			# TODO: What if no legal moves?
#			move_score, top_move = moves[0]
#			print "bestmove", top_move
##		elif line.startswith("usermove"):
##			_, move_uci = line.split(" ", 1)
##			board.push(chess.Move.from_uci(move_uci))
		elif line == "showboard":
			print board
		else:
			try:
				move = chess.Move.from_uci(line)
				board.push(move)
			except ValueError:
				print "Unknown command."
				logging.debug(">>> Unknown command.")
		sys.stdout.flush()

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		logging.error(e, exc_info=True)

