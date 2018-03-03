#!/usr/bin/python

import chess
import chess.syzygy

tablebase = chess.syzygy.open_tablebases("/home/snp/Downloads/syzygy")

def score_dtz_value(move, dtz, halfmove_clock):
	# If we're winning then win as fast as possible.
	if dtz > 0:
		# A move that zeros the half-move clock is better than any other.
		if halfmove_clock == 0:
		 	return 20000
		return 10000 - dtz
	# If we're losing then make it take as long as possible.
	if dtz < 0:
		# A move that zeros the half-move clock is worse than any other.
		if halfmove_clock == 0:
			return -20000
		return -10000 - dtz
	# If it's a draw then just return the draw.
	return 0

def play_dtz(board):
	# If we have more than 6 pieces then no move will put us in our 5-man tablebase.
	if chess.popcount(board.occupied) > 6:
		return
	# Search legal moves for any that put us in our tablebase.
	candidates = []
	for move in board.legal_moves:
		board.push(move)
		if chess.popcount(board.occupied) <= 5:
			candidates.append((move, tablebase.probe_dtz(board), board.halfmove_clock))
		board.pop()
	if not candidates:
		return
	best_move, best_move_dtz, _ = max(candidates, key=lambda triple: -score_dtz_value(*triple))
	# If we have a guaranteed win move then take it.
	if best_move_dtz > 0:
		return best_move
	# Otherwise, only return the best move if we are unsure about our current standing.
	if chess.popcount(board.occupied) <= 5:
		return best_move

def score_position(board):
	if chess.popcount(board.occupied) > 5:
		return
	return tablebase.probe_dtz(board)

