"""
This module controls the general global optimization heuristic. It is not
specific to any domain, but provides the main Optimizer class to be subclassed
by specific global solvers, as well as generic functions to be used by them.
"""
import collections
import logging

import random

log = logging.getLogger('syphus.optimize')


def random_choice(group):
    """
    Randomly choose one element of `group`.
    """
    return random.choice(group)


def random_idx_with_new_value(group, legal_values):
    """
    Randomly pick an element of group and choose a new legal value.
    """
    if group:
        idcs = range(len(group))
        # Get random slot at `group`.
        idx = random.choice(idcs)
        # Get value of slot.
        current_value = group[idx]
        legal_idx_values = [
            value for value in legal_values[idx] if value != current_value
        ]
        if legal_idx_values:
            # If there are legal values for this idx, pick one.
            new_value = random.choice(legal_idx_values)
            return (idx, new_value)

    return (None, None)


def random_reassign(group, legal_values, **kwargs):
    idx, value = random_idx_with_new_value(group, legal_values, **kwargs)
    if idx is not None and value is not None:
        group[idx] = value


def persistent_reassign(group, legal_values):
    """
    Random reassignment for persistent data structures.
    """
    idx, value = random_idx_with_new_value(group, legal_values)
    if idx is not None and value is not None:
        group = group.set(idx, value)

    return idx, group


def random_swap(group):
    """
    Swap the order of two elements in a group.
    """
    if len(group) >= 2:
        i = random.choice(len(group) - 1)
        node_idx = group[i]
        swapped_idx = group[i + 1]
        group[i], group[i + 1] = group[i + 1], group[i]

        return (node_idx, swapped_idx)


class Tabu(object):

    __slots__ = ('value', 'criterion', 'lifetime')

    def __init__(self, value, criterion=None, lifetime=1):
        self.value = value
        self.criterion = criterion
        self.lifetime = lifetime


class Optimizer(object):
    """
    Parent class for implementing Tabu Search optimizers.

    Tabu Search works as a meta-heuristic around a hillclimbing solution-space
    search by maintaining a number of data structures to record notable points
    in the space as well as notable attributes of those solutions. This allows
    the search to backtrack or accept moves that are not strict improvements
    while still optimizing for solution quality.

    In particular, solution elements can be marked as *tabu*, indicating that
    they, or the solutions containing them, are probably undesirable, and
    imposing a greater cost in the form of score improvement to justify
    accepting them.
    """
    MAX_ITER = 150
    MAX_SINCE_MINIMUM = 30
    MAX_TABU_SIZE = None
    CRITERIA_REDUCTION_MULTIPLIER = 1.1

    def __init__(self):
        # A queue of solution components that are tabu-active.
        self.tabu = collections.deque()
        # Initialize counter for debugging purposes.
        self.global_minima_found = 0

    #
    # === Optimization Callbacks ===
    #
    # Override these methods to implement the tabu search strategies for a
    # subclass domain.
    #

    def get_best_score_move(self, X, min_score):
        """
        Generate a neighborhood of possible adjacent moves from the current
        state and find the best move to make.
        """
        # Get a neighborhood of minimally different states.
        neighborhood = self.get_neighborhood(X)
        score_moves = sorted((self.objective(
            move, current_state=X, current_score=min_score), move)
                             for move in neighborhood)
        best_score, best_move = (None, None)
        # Iterate through the sorted score moves, picking the lowest (first)
        # one available.
        for score, move in score_moves:
            if best_score is None and not self.is_tabu(move, test_score=score):
                best_score, best_move = (score, move)
            elif best_score is not None and score > min_score:
                self.mark_tabu(move, diff_with=best_move, cost=score)
            else:
                self.mark_good_move(move, diff_with=X)

        if best_score is None:
            # If no move is allowed, fall back to the 'least inadmissable'
            # move.
            best_score, best_move = score_moves[0]

        return best_score, best_move

    def handle_non_optimal_best(self, best_move, current_solution, best_score):
        """
        Handle the case where a move has been selected from the neighborhood
        but it's not better than the current optimum.
        """
        # If the best move is not favorable, mark it tabu.
        self.mark_tabu(best_move, diff_with=current_solution, cost=best_score)

    def initial_state(self):
        """
        Return the initial solution state.
        """
        raise NotImplementedError()

    def objective(self, X):
        raise NotImplementedError()

    def get_neighborhood(self, X):
        raise NotImplementedError()

    def mark_tabu(self, move, diff_with=None, cost=None):
        """
        Mark a move, or components of it, as tabu.
        """
        raise NotImplementedError()

    def is_tabu(self, move, test_score=None):
        raise NotImplementedError()

    def mark_good_move(self, move, diff_with=None):
        raise NotImplementedError()

    def restart(self):
        """
        Return state for restarting optimization.
        """
        return self.initial_state()

    #
    # === Optimization Logic ===
    #

    def manage_tabu_data(self):
        """
        Adjust tabu lifetimes and remove any aged elements from the tabu set.
        """
        to_remove = set()
        for element in self.tabu:
            if element.lifetime <= 0:
                to_remove.add(element)
            else:
                element.lifetime -= 1
                element.criterion *= self.CRITERIA_REDUCTION_MULTIPLIER

        for element in to_remove:
            self.tabu.remove(element)

        if self.MAX_TABU_SIZE:
            while len(self.tabu) > self.MAX_TABU_SIZE:
                self.tabu.popleft()

    def optimize(self):
        """
        Perform Tabu Search optimization.
        """
        # Initialize optimization variables.
        n_iter = iter_since_last_minimum = 0
        min_solution = current_solution = self.initial_state()
        min_score = self.objective(current_solution)

        self.local_minima_found = 0

        while n_iter < self.MAX_ITER:
            n_iter += 1
            iter_since_last_minimum += 1

            best_score, best_move = self.get_best_score_move(current_solution,
                                                             min_score)
            if best_score >= min_score:
                # The best score is potentially better than the current
                # solution, but not better than the known-best solution.
                self.handle_non_optimal_best(best_move, current_solution,
                                             best_score)
                if best_score == float('inf'):
                    self.handle_constraint_failure(best_score, best_move)

            else:
                # Otherwise if it's an improvement, mark the improved-upon
                # features tabu.
                self.mark_tabu(
                    current_solution, diff_with=best_move, cost=min_score)
                # Mark it as worth returning to.
                self.mark_good_move(best_move, diff_with=current_solution)
                iter_since_last_minimum = 0
                self.local_minima_found += 1
                # Record the current best solution.
                min_score = best_score
                min_solution = best_move

            if iter_since_last_minimum >= self.MAX_SINCE_MINIMUM:
                try:
                    current_solution = self.restart()
                    iter_since_last_minimum = 0
                except StopIteration:
                    break
            else:
                current_solution = best_move

            self.manage_tabu_data()

        log.debug("{} minima found on state of size {}".format(
            self.local_minima_found, len(current_solution)))
        return min_score, min_solution
