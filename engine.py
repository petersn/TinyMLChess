#!/usr/bin/python

import logging, time, sys, random, collections
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
	model.load_model(policy_net, policy_path)
	model.load_model(value_net, value_path)

def softmax(logits):
	"""Somewhat numerically stable softmax routine."""
	e_x = np.exp(logits - np.max(logits))
	return e_x / e_x.sum()

def estimate_plies_remaining(plies_into_game):
	offset, a1, c1, a2, b2, c2 = 34.0, 46.1, 38.3, 19.5, 157.8, 15.3
	return offset + a1 * np.exp(-plies_into_game / c1) + a2 / (1 + np.exp(-(plies_into_game - b2) / c2))

def BAD_compute_posterior(boards):
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

def BAD_compute_value(boards):
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

class NNEvaluator:
	ENSEMBLE_SIZE = 32
	QUEUE_DEPTH = 4096
	PROBABILITY_THRESHOLD = 0.09
	MAXIMUM_CACHE_ENTRIES = 200000

	class Entry:
		__slots__ = ["board", "value", "posterior", "game_over"]

		def __init__(self, board, value, posterior, game_over):
			self.board = board
			self.value = value
			self.posterior = posterior
			self.game_over = game_over

	def __init__(self):
		self.cache = {}
		self.board_queue = collections.deque(maxlen=NNEvaluator.QUEUE_DEPTH)
		self.ensemble_sizes = []

	def __repr__(self):
		return "<NNEvaluator cache=%i queue=%i>" % (len(self.cache), len(self.board_queue))

	@staticmethod
	def board_key(b):
		return (
			b.turn,
			b.pawns,
			b.knights,
			b.bishops,
			b.rooks,
			b.queens,
			b.kings,
			b.occupied_co[0],
		)

	def __contains__(self, board):
		return NNEvaluator.board_key(board) in self.cache

	def evaluate(self, input_board):
		# Build up an ensemble to evaluate together.
		ensemble = [input_board]
		while self.board_queue and len(ensemble) < NNEvaluator.ENSEMBLE_SIZE:
			queued_board = self.board_queue.popleft()
			# The board might have been evaluated since we queued it, in which case skip it.
			# TODO: Evaluate if this is worth it. How many transpositions do we get?
			if queued_board not in self:
				ensemble.append(queued_board)

		# Evaluate the boards together.
		self.ensemble_sizes.append(len(ensemble))
		features = map(utils.extract_features, ensemble)
		posteriors = policy_net.final_output.eval(feed_dict={
			policy_net.input_ph: features,
			policy_net.is_training_ph: False,
		})
		values = value_net.final_output.eval(feed_dict={
			value_net.input_ph: features,
			value_net.is_training_ph: False,
		})

		# Write an entry into our cache.
		for board, raw_posterior, (value,) in zip(ensemble, posteriors, values):
			raw_posterior = softmax(raw_posterior)
			posterior = {move: utils.get_move_score(raw_posterior, move) for move in board.legal_moves}
			# Renormalize the posterior. Add a small epsilon into the denominator to prevent divison by zero.
			denominator = sum(posterior.itervalues()) + 1e-6
			posterior = {move: prob / denominator for move, prob in posterior.iteritems()}
			entry = NNEvaluator.Entry(board=board, value=value, posterior=posterior, game_over=False)
			self.cache[NNEvaluator.board_key(board)] = entry

	def add_to_queue(self, board):
		if board not in self:
			self.board_queue.append(board)

	def populate(self, board):
		# XXX: This is ugly...
		if hasattr(board, "evaluations"):
			return

		# Get base value and posterior, independent of special value adjustments.
		if board not in self:
			self.evaluate(board)
		entry = self.cache[NNEvaluator.board_key(board)]
		# Evaluate special value adjustments.
		# Adjustment #1: The game is mate, stalemate, 50 move rule, or is a three-fold repetition.
		result = board.result(claim_draw=True)
		if result != "*":
			outcome_for_white = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result]
			outcome_for_current_player = outcome_for_white * (1 if board.turn else -1)
			entry.value = float(outcome_for_current_player)
			entry.game_over = True
		else:
			# NB: It is *critical* that we not let tablebase values override basic scoring,
			# or we get draws by repetition and 50 move rule wrong.
			# Adjustment #2: Score by tablebase.
			# XXX: TODO: Properly take into account board.halfmove_clock.
			score = tb.score_position(board)
			if score != None:
				entry.value = float(cmp(score, 0))
		board.evaluations = entry

		# If we exceed our cache size then empty our cache.
		if len(self.cache) > NNEvaluator.MAXIMUM_CACHE_ENTRIES:
			logging.debug("Emptying cache!")
			self.cache = {}

global_evaluator = NNEvaluator()

class MCTSEdge:
	def __init__(self, move, child_node, parent_node=None):
		self.move = move
		self.child_node = child_node
		self.parent_node = parent_node
		self.edge_visits = 0
		self.edge_total_score = 0

		self.total_weight = 0

	def get_edge_score(self):
#		return self.edge_total_score / (self.edge_visits * (self.edge_visits + 1.0) / 2.0)
		return self.edge_total_score / self.total_weight

	def adjust_score(self, new_score):
#		self.edge_visits += 1
#		self.edge_total_score += new_score * self.edge_visits
		self.edge_visits += 1
		weight = (2 + self.edge_visits) ** MCTS.VISIT_WEIGHT_EXPONENT
		self.edge_total_score += new_score * weight
		self.total_weight += weight

	def __str__(self):
		return "<%s %4.1f%% v=%i s=%.5f c=%i>" % (
			str(self.move),
			100.0 * self.parent_node.board.evaluations.posterior[self.move],
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
		self.graph_name_suffix = ""

	def total_action_score(self, move):
		if move in self.outgoing_edges:
			edge = self.outgoing_edges[move]
			u_score = MCTS.exploration_parameter * self.board.evaluations.posterior[move] * (1.0 + self.all_edge_visits)**0.5 / (1.0 + edge.edge_visits)
			Q_score = edge.get_edge_score() if edge.edge_visits > 0 else 0.0
		else:
			u_score = MCTS.exploration_parameter * self.board.evaluations.posterior[move] * (1.0 + self.all_edge_visits)**0.5
			Q_score = 0.0
		return Q_score + u_score

	def select_action(self, continue_even_if_game_over=False):
		global_evaluator.populate(self.board)
		# If we have no legal moves then return None.
		if not self.board.evaluations.posterior:
			return
			#logging.debug("Board state with no variations: %s" % (self.board.fen(),))
		# If the game is over and we're not supposed to continue then return None.
		if self.board.evaluations.game_over and not continue_even_if_game_over:
			return
		return max(self.board.evaluations.posterior, key=self.total_action_score)

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
	VISIT_WEIGHT_EXPONENT = 2.0

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
				# XXX: TODO: Document this logic here properly.
				# Basically, the gist is that sometimes UCI masters will ask to generate a move when the root is already finished (e.g., a draw can be claimed).
				# Rather than just reporting a totally random move we instead ignore that the game is over.
				continue_even_if_game_over = node == self.root_node
				move = node.select_action(continue_even_if_game_over=continue_even_if_game_over)
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
		# 3a) Evaluate the new node.
		global_evaluator.populate(new_node.board)
		# 3b) Queue up some children just for efficiency.
		for m, probability in new_node.board.evaluations.posterior.iteritems():
			if probability > NNEvaluator.PROBABILITY_THRESHOLD:
				new_board = new_node.board.copy(stack=False)
				new_board.push(m)
				global_evaluator.add_to_queue(new_board)
		# Convert the expected value result into a score.
		value_score = (new_node.board.evaluations.value + 1) / 2.0
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
			logging.debug(`self.root_node.board.evaluations.posterior`)
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
	VISITS    = 10000000
	MAX_STEPS = 10000000
	TIME_SAFETY_MARGIN = 0.150
	IMPORTANCE_FACTOR = {
		1: 0.1, 2: 0.2,
		3: 0.3, 4: 0.4,
		5: 0.5, 6: 0.6,
		7: 0.7, 8: 0.8,
		9: 0.9,
	}
	# XXX: This is awful. Switch this over to run-time benchmarking.
	MAX_STEPS_PER_SECOND = 375.0

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

	def genmove(self, time_to_think, early_out=True):
		tablebase_move = tb.play_dtz(self.state)
		if tablebase_move != None:
			logging.debug(RED + ("Playing tablebase move: %s" % (tablebase_move,)) + ENDC)
			return tablebase_move

		start_time = time.time()
		most_visited_edges = TopN(2, key=lambda edge: edge.edge_visits)
		most_visited_edges.update(self.mcts.root_node.outgoing_edges.itervalues())
		total_steps = 0
		for step_number in xrange(self.MAX_STEPS):
			now = time.time()
			# Compute remaining time we have left to think.
			remaining_time = time_to_think - (now - start_time) - self.TIME_SAFETY_MARGIN
			if remaining_time <= 0.0 and total_steps > 0:
				break
			# If we don't have enough time for the number two option to catch up, early out.
			if early_out and len(most_visited_edges.entries) == 2 and total_steps > 0:
				runner_up, top_choice = most_visited_edges.entries
				#logging.debug("Step number: %i runner_up: %r top_choice: %r %r" % (step_number, runner_up, top_choice, remaining_time))
				if runner_up.edge_visits + remaining_time * MCTSEngine.MAX_STEPS_PER_SECOND < top_choice.edge_visits:
					logging.debug("Early out; cannot catch up in %f seconds." % (remaining_time,))
					break
			total_steps += 1
			visited_edge = self.mcts.step()
			assert visited_edge.parent_node == self.mcts.root_node
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
					len(self.mcts.root_node.board.evaluations.posterior),
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
		logging.debug("Cache entries: %i" % (len(global_evaluator.cache),))
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
	#__import__("pprint").pprint(sorted(score_moves(chess.Board())))
	logging.basicConfig(
		format="[%(process)5d] %(message)s",
		level=logging.DEBUG,
	)
	initialize_models()
	engine = MCTSEngine()
	for _ in xrange(2):
		print "Doing warmup evaluation."
		start = time.time()
		engine.genmove(0.1)
		stop = time.time()
		print "Warmup took:", stop - start

	print "Starting performance section."
	measure_time = 10.0
	engine.genmove(measure_time, early_out=False)

	total_visits = engine.mcts.root_node.all_edge_visits
	print "Total visits:", total_visits
	print "Ensembles:", len(global_evaluator.ensemble_sizes)
	print "Average ensemble:", np.average(global_evaluator.ensemble_sizes)

	with open("speeds", "a+") as f:
		print >>f, "ES=%i  QD=%4i  PT=%.3f  (MT=%f)  append (renorm)  Speed: %f" % (
			NNEvaluator.ENSEMBLE_SIZE,
			NNEvaluator.QUEUE_DEPTH,
			NNEvaluator.PROBABILITY_THRESHOLD,
			measure_time,
			total_visits / measure_time,
		)

	print global_evaluator.ensemble_sizes

