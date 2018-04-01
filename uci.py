#!/usr/bin/python

import os, sys, re, argparse, logging
import chess
import engine

RED  = "\x1b[91m"
ENDC = "\x1b[0m"

go_options = {
	"wtime": "wtime ([0-9]+)",
	"btime": "btime ([0-9]+)",
	"winc": "winc ([0-9]+)",
	"binc": "binc ([0-9]+)",
}
option_options = {
	"name": "name ([^ ]+)",
	"value": "value ([^ ]+)",
}

def parse_options(options, line):
	results = {}
	for k, v in options.iteritems():
		m = re.search(v, line)
		if m:
			results[k], = m.groups()
	return results

default_ms_per_move = 5000

def color_board_string(board):
	s = str(board)
	for c in "prnbqk":
		s = s.replace(c, RED + c + ENDC)
	return s

def main():
	print "TinyMLChess"
	sys.stdout.flush()

	all_options = {}

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
			print "option name PolicyPath type string default <empty>"
			print "option name ValuePath type string default <empty>"
			print "uciok"
		elif line == "isready":
			print "readyok"
		elif line == "ucinewgame":
			board = chess.Board()
			manager = engine.MCTSEngine()
		elif line.startswith("position fen"):
			_, _, fen = line.split(" ", 2)
			board = chess.Board(fen)
			manager.set_state(board)
			logging.debug(">>> Setting board: %s\n%s" % (board.fen(), color_board_string(board)))
		elif line.startswith("position startpos moves"):
			_, _, _, moves = line.split(" ", 3)
			board = chess.Board()
			for m in moves.split():
				board.push(chess.Move.from_uci(m))
			manager.set_state(board)
			logging.debug(">>> Setting board: %s\n%s" % (board.fen(), color_board_string(board)))
		elif line == "go" or line.startswith("go "):
			option_values = parse_options(go_options, line)
			our_time = float(option_values.get({True: "wtime", False: "btime"}[board.turn], default_ms_per_move)) * 1e-3
			our_inc  = float(option_values.get({True: "winc", False: "binc"}[board.turn], 0.0)) * 1e-3
			logging.debug(">>> Thinking with time=%.4f inc=%.4f." % (our_time, our_inc))
			move = manager.genmove_with_time_control(our_time, our_inc)
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
		elif line.startswith("setoption"):
			parsed = parse_options(option_options, line)
			all_options[parsed["name"]] = parsed["value"]
			print "info string Got option %r set to %r" % (parsed["name"], parsed["value"])
		else:
			try:
				move = chess.Move.from_uci(line)
				board.push(move)
			except ValueError:
				print "Unknown command."
				logging.debug(">>> Unknown command.")
		sys.stdout.flush()

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	os.chdir("/home/snp/proj/TinyMLChess")
	parser = argparse.ArgumentParser("Simple UCI Chess engine.")
	parser.add_argument("--policy-path", metavar="PATH", help="Path to policy network weights file.")
	parser.add_argument("--value-path", metavar="PATH", help="Path to policy network weights file.")
	parser.add_argument("--sp", action="store_true")
	args = parser.parse_args()

	LOG_PATH = "/tmp/chess.log"
	logging.basicConfig(
		format="[%(process)5d] %(message)s",
		filename=LOG_PATH,
		filemode="a+",
		level=logging.DEBUG,
	)
	logging.debug("Starting log.")

	# XXX: This is currently a completely inappropriate place to do this initialization.
	if args.policy_path != None:
		engine.policy_path = args.policy_path
	if args.value_path != None:
		engine.value_path = args.value_path
	engine.initialize_models()

	try:
		main()
	except Exception as e:
		logging.error(e, exc_info=True)

