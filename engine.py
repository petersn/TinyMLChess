#!/usr/bin/python

import logging, time, sys, random
import numpy as np
import tensorflow as tf
import chess
import model_dual as model
import utils
import tb

RED  = "\x1b[91m"
ENDC = "\x1b[0m"

policy_path = "models/policy-20x128-model-024.npy"
value_path  = "models/value-20x128-fc64-model-012.npy"
#value_path  = "models/value-SP-20x128-model-012.npy"
#value_path  = "models/value-SP-20x128-model-008.npy"
if "--sp" in sys.argv:
	print >>sys.stderr, "Using other model."
	value_path  = "models/value-SP2-10x128-model-015.npy"

initialized = False

def initialize_models():
	global policy_net, value_net, sess, initialized
	if initialized:
		return
	initialized = True
	policy_net = model.ChessPolicyNet("policy/", build_training=False)
	value_net = model.ChessValueNet("value/", build_training=False)
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess
	#model.load_model(net, "models/joint-model-040.npy")
	#model.load_model(net, "models/value-12x96-fc64-model-001.npy")
	#model.load_model(policy_net, "models/joint-model-040.npy")
	model.load_model(policy_net, policy_path)
	model.load_model(value_net, value_path)

def softmax(logits):
	"""Somewhat numerically stable softmax routine."""
	e_x = np.exp(logits - np.max(logits))
	return e_x / e_x.sum()

def estimate_plies_remaining(plies_into_game):
	offset, a1, c1, a2, b2, c2 = 34.0, 46.1, 38.3, 19.5, 157.8, 15.3
	return offset + a1 * np.exp(-plies_into_game / c1) + a2 / (1 + np.exp(-(plies_into_game - b2) / c2))

"""
def policy_score_moves(board):
	result = tb.play_dtz(board)
	if result != None:
		logging.debug("Playing tablebase move.")
		return [(0.0, result)]

	features = utils.extract_features(board)
	posterior, = policy_net.final_output.eval(feed_dict={
		policy_net.input_ph: [features],
		policy_net.is_training_ph: False,
	})
	assert posterior.shape == (64, 8 * 8)
	posterior = softmax(posterior)
#	posterior = map(softmax, posterior)
	def score_move(m):
		# For now only promote to queens.
		if m.promotion not in (None, chess.QUEEN):
			return -1
		return posterior[0][m.from_square] * posterior[1][m.to_square]
	return sorted([(utils.get_move_score(posterior, m), m) for m in board.legal_moves], reverse=True)
#	return sorted([(score_move(m), m) for m in board.legal_moves], reverse=True)

def score_moves(board):
	result = tb.play_dtz(board)
	if result != None:
		logging.debug("Playing tablebase move.")
		return [(0.0, result)]

	features = []
	all_moves = []
	for move in board.legal_moves:
		board.push(move)
		all_moves.append(move)
		features.append(utils.extract_features(board))
		board.pop()
	scores = value_net.final_output.eval(feed_dict={
		value_net.input_ph: features,
		value_net.is_training_ph: False,
	}).reshape((-1,))
#	scores = -scores
	return sorted(zip(scores, all_moves))
"""

def compute_posterior(boards):
	boards = [b for b in boards if not hasattr(b, "ML_posterior")]
	if not boards:
		return
	features = map(utils.extract_features, boards)
	posteriors = policy_net.final_output.eval(feed_dict={
		policy_net.input_ph: features,
		policy_net.is_training_ph: False,
	})
	for board, raw_posterior in zip(boards, posteriors):
		logging.debug("Posterior on: %r" % (board,))
		raw_posterior = softmax(raw_posterior)
		board.ML_posterior = {}
		# If the board has a solid_outcome then it gets an empty posterior.
		# However, if we're the root node, then we build a posterior regardless.
		if hasattr(board, "ML_solid_outcome") and not hasattr(board, "is_root"):
			continue
		for m in board.legal_moves:
			board.ML_posterior[m] = utils.get_move_score(raw_posterior, m)

def compute_value(boards):
	boards = [b for b in boards if not hasattr(b, "ML_value")]
	# Try to score by normal rules.
	for b in boards:
		# NB: There used to be a comment here about setting claim_draw=False because 0000 moves increment the half-move clock.
		# This should be fixed, because now I don't issue new moves.
		result = b.result(claim_draw=True)
		if result != "*":
			outcome_for_white = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result]
			outcome_for_current_player = outcome_for_white * (1 if b.turn else -1)
			# Stash the outcome into the board for caching purposes.
			b.ML_solid_outcome = outcome_for_current_player
#			logging.debug("Scored position with normal rules as: %s" % (outcome_for_current_player,))
			b.ML_value = float(outcome_for_current_player)
	# Repetition here is super ugly.
	boards = [b for b in boards if not hasattr(b, "ML_value")]
	# Try to score boards by tablebase.
	# NB: We *MUST* do this after trying by normal rules or we get draws by repetition and 50 move rule wrong.
	# XXX: TODO: Properly take into account b.halfmove_clock to avoid scoring a position as winning/losing when it's a draw.
	for b in boards:
		try:
			score = tb.score_position(b)
		except chess.syzygy.MissingTableError, e:
			logging.debug("Current FEN: %r" % (b.fen(),))
			logging.debug("Bad Syzygy attempt: %r" % (e,))
			raise e
		if score != None:
#			logging.debug("Scored with TB as: %s" % (score,))
			b.ML_value = float(cmp(score, 0))
	# Repetition here is super ugly.
	boards = [b for b in boards if not hasattr(b, "ML_value")]
	if not boards:
		return

	features = map(utils.extract_features, boards)
	values = value_net.final_output.eval(feed_dict={
		value_net.input_ph: features,
		value_net.is_training_ph: False,
	})
	for board, value in zip(boards, values):
		board.ML_value, = value

class MCTSEdge:
	def __init__(self, move, child_node, parent_node=None):
		self.move = move
		self.child_node = child_node
		self.parent_node = parent_node
		self.edge_visits = 0
		self.edge_total_score = 0

	def get_edge_score(self):
		return self.edge_total_score / self.edge_visits

	def adjust_score(self, new_score):
		self.edge_visits += 1
		self.edge_total_score += new_score

	def __str__(self):
		return "<%s %4.1f%% v=%i s=%.5f c=%i>" % (
			str(self.move),
			100.0 * self.parent_node.board.ML_posterior[self.move],
			self.edge_visits,
			self.get_edge_score(),
			len(self.child_node.outgoing_edges),
		)

class MCTSNode:
	def __init__(self, board, parent=None):
		self.board = board
		self.parent = parent
		self.all_edge_visits = 0
		self.outgoing_edges = {}
		self.individual_value_score = 0.0
		self.graph_name_suffix = ""

	def total_action_score(self, move):
		if move in self.outgoing_edges:
			edge = self.outgoing_edges[move]
			u_score = MCTS.exploration_parameter * self.board.ML_posterior[move] * (1.0 + self.all_edge_visits)**0.5 / (1.0 + edge.edge_visits)
			Q_score = edge.get_edge_score() if edge.edge_visits > 0 else 0.0
		else:
			u_score = MCTS.exploration_parameter * self.board.ML_posterior[move] * (1.0 + self.all_edge_visits)**0.5
			Q_score = 0.0
		return Q_score + u_score

	def select_action(self):
		compute_posterior([self.board])
		# If we have no legal moves then return None.
		if not self.board.ML_posterior:
			return
			#logging.debug("Board state with no variations: %s" % (self.board.fen(),))
		return max(self.board.ML_posterior, key=self.total_action_score)

	def graph_name(self, name_cache):
		if self not in name_cache:
			name_cache[self] = "n%i%s" % (len(name_cache), "") #self.graph_name_suffix)
		return name_cache[self]

	def make_graph(self, name_cache):
		l = []
		for edge in self.outgoing_edges.itervalues():
			l.append("%s -> %s;" % (self.graph_name(name_cache), edge.child_node.graph_name(name_cache)))
		for edge in self.outgoing_edges.itervalues():
			# Quadratic time here from worst case for deep graphs.
			l.extend(edge.child_node.make_graph(name_cache))
		return l

class TopN:
	def __init__(self, N, key):
		self.N = N
		self.key = key
		self.entries = []

	def add(self, item):
		if item not in self.entries:
			self.entries += [item]
		self.entries = sorted(self.entries, key=self.key)[-self.N:]

	# Ugly, DRY this, but performance?
	def update(self, items):
		for i in items:
			self.add(i)
		#self.entries = sorted(self.entries + list(items), key=self.key)[-self.N:]

class MCTS:
	exploration_parameter = 1.5 * 2.0

	def __init__(self, root_board):
		self.root_node = MCTSNode(root_board)

	def select_principal_variation(self, best=False):
		node = self.root_node
		edges_on_path = []
		while True:
			if best:
				if not node.outgoing_edges:
					break
				move = max(node.outgoing_edges.itervalues(), key=lambda edge: edge.edge_visits).move
			else:
				move = node.select_action()
			if move not in node.outgoing_edges:
				break
			edge = node.outgoing_edges[move]
			edges_on_path.append(edge)
			node = edge.child_node
		return node, move, edges_on_path

	def step(self):
		def to_move_name(move):
			return "_%s" % (move,)
		# 1) Pick a child by repeatedly taking the best child.
		node, move, edges_on_path = self.select_principal_variation()
		# 2) If the move is non-null, expand once.
		if move != None:
			new_board = node.board.copy()
			new_board.push(move)
			new_node = MCTSNode(new_board, parent=node)
			new_node.graph_name_suffix = to_move_name(move)
			new_edge = node.outgoing_edges[move] = MCTSEdge(move, new_node, parent_node=node)
			edges_on_path.append(new_edge)
		else:
			# 2b) If the move is null, then we had no legal moves, and just propagate the score again.
			new_node = node
		# 3) Evaluate the new node.
		compute_value([new_node.board])
		# Convert the expected value result into a score.
		value_score = (new_node.board.ML_value + 1) / 2.0
		new_node.individual_value_score = value_score
		# 4) Backup.
		inverted = False
		for edge in reversed(edges_on_path):
			# Remember that each edge corresponds to an alternating player, so we have to reverse scores.
			inverted = not inverted
			value_score = 1 - value_score
			assert inverted == (edge.parent_node.board.turn != new_node.board.turn)
			edge.adjust_score(value_score)
			edge.parent_node.all_edge_visits += 1
		if not edges_on_path:
			self.write_graph()
			logging.debug("WARNING no edges on path!")
			logging.debug(`node`)
			logging.debug(`self.root_node`)
			logging.debug(`node is self.root_node`)
			logging.debug(`move`)
			logging.debug(`self.root_node.board`)
			logging.debug(`self.root_node.board.ML_posterior`)
			compute_posterior([self.root_node.board])
			logging.debug(`self.root_node.board.ML_posterior`)
			#logging.debug(`self.root_node.board.ML_solid_outcome`)
			logging.debug(`self.root_node.board.result(claim_draw=True)`)
			logging.debug(`getattr(self.root_node.board, "is_root", "hahabad")`)
			logging.debug(`list(self.root_node.board.legal_moves)`)
			logging.debug(`self.root_node.parent`)
			logging.debug("DONE. Crashing now.")
		# The final value of edge encodes the very first edge out of the root.
		return edge

	def play(self, player, move, print_variation_count=True):
		assert self.root_node.board.turn == player, "Bad play direction for MCTS!"
		if move not in self.root_node.outgoing_edges:
			if print_variation_count:
				logging.debug("Completely unexpected variation!")
			new_board = self.root_node.board.copy()
			new_board.push(move)
			self.root_node = MCTSNode(new_board)
			return
		edge = self.root_node.outgoing_edges[move]
		if print_variation_count:
			logging.debug("Traversing to variation with %i visits." % edge.edge_visits)
		self.root_node = edge.child_node
		self.root_node.parent = None

	def write_graph(self):
		name_cache = {}
		with open("/tmp/mcts.dot", "w") as f:
			f.write("digraph G {\n")
			f.write("\n".join(self.root_node.make_graph(name_cache)))
			f.write("\n}\n")
		return name_cache

class MCTSEngine:
#	VISITS = 1200 * 2
#	MAX_STEPS = 12000 * 2
#	VISITS = 5000
	VISITS    = 10000
	MAX_STEPS = 1000000
	TIME_SAFETY_MARGIN = 0.150
	IMPORTANCE_FACTOR = {
		1: 0.1, 2: 0.2,
		3: 0.3, 4: 0.4,
		5: 0.5, 6: 0.6,
		7: 0.7, 8: 0.8,
		9: 0.9,
	}
	MAX_STEPS_PER_SECOND = 100.0

	def __init__(self):
		self.state = chess.Board()
		self.mcts = MCTS(self.state)

	def set_state(self, new_board):
		# XXX: You can pick out the wrong board because board equality doesn't include history.
		# For example, you can't distinguish different kinds of promotions that are followed by a capture.
		# TODO: Evaluate if this can cause a really rare bug.

		# Check to see if this board is one of our children's children.
		for edge1 in self.mcts.root_node.outgoing_edges.itervalues():
			for edge2 in edge1.child_node.outgoing_edges.itervalues():
				if edge2.child_node.board == new_board:
					# We found a match! Reuse part of the tree.
					self.mcts.play(self.state.turn, edge1.move)
					self.mcts.play(not self.state.turn, edge2.move)
					self.state = new_board
					return
		logging.debug(RED + "Failed to match a subtree." + ENDC)
		self.state = new_board
		# XXX: Some UCI masters will ask us to make a move even when we can claim a threefold repetition.
		# If we can claim a threefold repetiton, clear out the ML_solid_result, so we make progress.

		self.mcts = MCTS(self.state)

	def genmove(self, time_to_think):
		# XXX: This is so ugly.
		self.mcts.root_node.board.is_root = True
		if hasattr(self.mcts.root_node.board, "ML_posterior"):
			del self.mcts.root_node.board.ML_posterior
			#compute_posterior([self.mcts.root_node.board])
		# XXX: End horrifically ugly.

		start_time = time.time()
		most_visited_edges = TopN(2, key=lambda edge: edge.edge_visits)
		most_visited_edges.update(self.mcts.root_node.outgoing_edges.itervalues())
#		most_visited_edge = None
#		if self.mcts.root_node.outgoing_edges:
#			most_visited_edge = max(self.mcts.root_node.outgoing_edges.itervalues(), key=lambda edge: edge.edge_visits)
		total_steps = 0
		for step_number in xrange(self.MAX_STEPS):
			now = time.time()
			# Compute remaining time we have left to think.
			remaining_time = time_to_think - (now - start_time) - self.TIME_SAFETY_MARGIN
			if remaining_time <= 0.0 and total_steps > 0:
				break
			# If we don't have enough time for the number two option to catch up, early out.
			if len(most_visited_edges.entries) == 2 and total_steps > 0:
				runner_up, top_choice = most_visited_edges.entries
				#logging.debug("Step number: %i runner_up: %r top_choice: %r %r" % (step_number, runner_up, top_choice, remaining_time))
				if runner_up.edge_visits + remaining_time * MCTSEngine.MAX_STEPS_PER_SECOND < top_choice.edge_visits:
					logging.debug("Early out; cannot catch up in %f seconds." % (remaining_time,))
					break
			total_steps += 1
			visited_edge = self.mcts.step()
			assert visited_edge.parent_node == self.mcts.root_node
#			if most_visited_edge is None or visited_edge.edge_visits > most_visited_edge.edge_visits:
#				most_visited_edge = visited_edge
			most_visited_edges.add(visited_edge)

			# We early out if we reach our POST value, and just visited the most visited edge.
			if visited_edge.edge_visits >= self.VISITS:
				break
			# We early out if we don't have enough time
			# Print debugging values.
			if step_number == 0 or (step_number + 1) % 250 == 0:
				logging.debug("Steps: %5i (C=%i/%i Top: %s This: %s)" % (
					step_number + 1,
					len(self.mcts.root_node.outgoing_edges),
					len(self.mcts.root_node.board.ML_posterior),
					(most_visited_edges.entries[-1] if most_visited_edges.entries else None),
					visited_edge,
				))
				if (step_number + 1) % 1000 == 0:
					self.print_principal_variation()
		logging.debug("Completed %i steps." % total_steps)
		logging.debug("Exploration histogram: %s" % (
			" ".join(
				str(edge)
				for edge in sorted(
					self.mcts.root_node.outgoing_edges.itervalues(),
					key=lambda edge: -edge.edge_visits,
				)
			),
		))
		self.print_principal_variation()
		return most_visited_edges.entries[-1].move

	def genmove_with_time_control(self, our_time, our_increment):
		# First, figure out how many plies we probably have remaining in the game.
		plies_into_game = self.state.fullmove_number - self.state.turn
		moves_remaining = estimate_plies_remaining(plies_into_game) / 2.0
		# Assume we will have to make this many additional moves.
		total_time = our_time + our_increment * moves_remaining
		time_budget = total_time / moves_remaining
		# Compute an importance factor.
		importance_factor = self.IMPORTANCE_FACTOR.get(self.state.fullmove_number, 1.3)
		importance_factor *= 1.5
		time_budget *= importance_factor
		# Never budget more than 50% of our remaining time.
		time_budget = min(time_budget, 0.5 * our_time)
		logging.debug("Budgeting %.2fms for this move." % (time_budget * 1e3,))
		return self.genmove(time_budget)

	def print_principal_variation(self):
		_, _, pv = self.mcts.select_principal_variation(best=True)
		logging.debug("PV [%2i]: %s" % (
			len(pv),
			" ".join(
				["%s", RED + "%s" + ENDC][i % 2] % (edge.move,)
				for i, edge in enumerate(pv)
			),
		))

if __name__ == "__main__":
	__import__("pprint").pprint(sorted(score_moves(chess.Board())))

