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
	encoded = np.zeros((2, 64))
	encoded[0, move.from_square] = 1
	encoded[1, move.to_square]   = 1
	return encoded

D = lambda x, y: x + y * 8
queen_moves = sum(
	([
		D(0, i), D(i, 0),
		D(0, -i), D(-i, 0),
		D(i,  i), D(-i,  i),
		D(i, -i), D(-i, -i),
	]
	for i in xrange(1, 8)),
	[]
)
knight_moves = [
	D( 1, 2), D( 1, -2),
	D(-1, 2), D(-1, -2),
	D( 2, 1), D( 2, -1),
	D(-2, 1), D(-2, -1),
]
all_layers = sorted(queen_moves + knight_moves)
assert len(all_layers) == 64
difference_to_layer_index = {diff: i for i, diff in enumerate(all_layers)}

def one_hot_to_large(move):
	assert move.shape == (2, 8, 8)
	pick_up, put_down = map(np.argmax, move)
	difference = put_down - pick_up
	result = np.zeros((len(all_layers), 64))
	result[difference_to_layer_index[difference], pick_up] = 1
	return result.reshape((len(all_layers), 8, 8))

def get_move_score(softmaxed_large_array, move):
	assert softmaxed_large_array.shape == (64, 8 * 8)
	# For now don't ever under-promote.
	if move.promotion not in (None, chess.QUEEN):
		return -1.0
	pick_up, put_down = move.from_square, move.to_square
	difference = put_down - pick_up
	layer = difference_to_layer_index[difference]
	posterior = softmaxed_large_array[layer, pick_up]
	return posterior

def features_to_board(feat):
	assert feat.shape == (8, 8, 13)
	feat = np.moveaxis(feat, -1, 0)
	board = chess.Board("8/8/8/8/8/8/8/8 w KQkq - 0 0")
	for i, layer in enumerate(feat[:12]):
		layer = layer.reshape((64,))
		color = (i / 6) == 0
		kind = getattr(chess, PIECE_NAMES[i % 6][:-1].upper())
		piece = chess.Piece(kind, color)
		for pos, v in enumerate(layer):
			if v == 1:
				board.set_piece_at(pos, piece)
	return board

if __name__ == "__main__":
	b = chess.Board()
	b.push_san("e4")
	f = extract_features(b)

