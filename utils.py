#!/usr/bin/python

import chess
import numpy as np

FEATURE_LAYERS = (
	6 + # Friendly: pawns, knights, bishops, rooks, queens, king.
	6 + # Opponent: pawns, knights, bishops, rooks, queens, king.
	1)  # All ones for seeing the edge of the board in convolutions.

PIECE_NAMES = ["pawns", "knights", "bishops", "rooks", "queens", "kings"]

def extract_features(board):
	features = np.zeros((8, 8, FEATURE_LAYERS), dtype=np.int8)
	white_to_move = board.turn

	# Iterate over piece kinds, writing each kind into the feature map.
	for piece_index, name in enumerate(PIECE_NAMES):
		occupancy = getattr(board, name)
		# Iterate over possible board locations at which this piece could exist.
		for square_index in xrange(64):
			square_mask = 1 << square_index
			if occupancy & square_mask:
				# If a piece of kind `piece_index` does indeed exist at `square_index`, figure out it color and insert it.
				piece_is_white = bool(board.occupied_co[chess.WHITE] & square_mask)
				if not piece_is_white:
					assert board.occupied_co[chess.BLACK] & square_mask
				assert white_to_move in (False, True) and piece_is_white in (False, True)
				piece_is_friendly = white_to_move == piece_is_white
				features[
					square_index / 8,
					square_index % 8,
					piece_index + (1 - piece_is_friendly) * 6,
				] = 1

	# If we're encoding a move for black then we flip the board vertically.
	if not white_to_move:
		features = features[::-1,:,:]

	# Set the last feature map to be all ones.
	features[:,:,12] = 1

	return features

def encode_move(move):
	encoded = np.zeros((64, 2))
	encoded[move.from_square,0] = 1
	encoded[move.to_square,0]   = 1
	return encoded

if __name__ == "__main__":
	b = chess.Board()
	b.push_san("e4")
	f = extract_features(b)

