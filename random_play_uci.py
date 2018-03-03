#!/usr/bin/python

import chess
import random, sys, logging

LOG_PATH = "/tmp/random_chess.log" 
logging.basicConfig(
	format="%(message)s",
	filename=LOG_PATH,
	filemode="a+",
	level=logging.DEBUG,
)
logging.debug("Starting log.")

#play_tablebase = "--tb" in sys.argv
play_tablebase = True

def main():
	print "RandomChess"
	sys.stdout.flush()
	board = chess.Board()

	while True:
		line = raw_input()
		logging.debug("Got: %s" % (line,))
		if line == "quit":
			exit()
		elif line == "uci":
			print "id name RandomChess"
			print "id author Peter Schmidt-Nielsen"
			print "uciok"
		elif line == "isready":
			print "readyok"
		elif line == "ucinewgame":
			board = chess.Board()
		elif line.startswith("position fen"):
			_, _, fen = line.split(" ", 2)
			board = chess.Board(fen)
		elif line.startswith("position startpos moves"):
			_, _, _, moves = line.split(" ", 3)
			board = chess.Board()
			for m in moves.split():
				board.push(chess.Move.from_uci(m))
		elif line == "go" or line.startswith("go "):
			move = None
			if play_tablebase:
				import tb
				move = tb.play_dtz(board)
				if move != None:
					logging.debug("Playing tablebase move.")
			if move is None:
				move = random.choice(list(board.legal_moves))
			print "bestmove", move
		else:
			try:
				move = chess.Move.from_uci(line)
				board.push(move)
			except ValueError:
				print "Unknown command."
		sys.stdout.flush()

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		logging.error(e, exc_info=True)

