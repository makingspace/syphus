import collections

import numpy as np

import pytest
import syphus
from hypothesis import event, given, settings
from hypothesis.strategies import floats


class GenericOptimizer(syphus.Optimizer):
    def __init__(self, a, b):
        super(GenericOptimizer, self).__init__()
        self.a = a
        self.b = b

        self.x_scale = (self.x_max - self.x_min) / 6.
        self.y_scale = (self.y_max - self.y_min) / 6.
        self.MAX_TABU_SIZE = 18

        self.cell_memory = [collections.Counter(), collections.Counter()]

    def hash(self, xy):
        return xy

    def initial_state(self):
        return (self.a, self.b)

    def _step(self, xy):
        x, y = xy

        step_size = np.random.uniform(-self.x_scale, self.x_scale)
        step_size2 = np.random.uniform(-self.y_scale, self.y_scale)

        x += step_size
        y += step_size2

        if x < self.x_min or x > self.x_max:
            x = self.a
        if y < self.y_min or y > self.y_max:
            y = self.a

        return (x, y)

    def get_neighborhood(self, xy):
        return [self._step(xy) for _ in xrange(20)]

    def _decompose(self, xy):
        x, y = xy
        return ((0, round(x, 4)), (1, round(y, 4)))

    def _update_lifetimes(self, lifetimes, xy, diff_with):
        x_component, y_component = self._decompose(xy)

        if diff_with is not None:
            x, y = xy
            other_x, other_y = diff_with
            delta_x, delta_y = abs(other_x - x), abs(other_y - y)
            ratio_x, ratio_y = (delta_x / self.x_scale, delta_y / self.y_scale)
        else:
            ratio_x = .1
            ratio_y = .1

        def lifetime(ratio):
            return max(1, int(100 * ratio))

        lifetimes[x_component] += lifetime(ratio_x)
        lifetimes[y_component] += lifetime(ratio_y)

    def mark_tabu(self, xy, diff_with=None, cost=None):
        x_component, y_component = self._decompose(xy)

        self.tabu.append(x_component)
        self.tabu.append(y_component)

        self._update_lifetimes(self.tabu_lifetimes, xy, diff_with)
        # Set up aspiration criteria for these tabu components. To consider
        # them admissable, the resulting score would need to be lower than the
        # cost set here.
        if cost is not None:
            self.aspiration_criteria[x_component] = cost
            self.aspiration_criteria[y_component] = cost
            self._update_lifetimes(self.aspiration_criteria_lifetimes, xy,
                                   diff_with)

    def _is_tabu(self, move, test_score):
        if move in self.tabu:
            if test_score is None or move not in self.aspiration_criteria:
                return True
            else:
                return test_score < self.aspiration_criteria[move]
        else:
            return False

    def is_tabu(self, xy, test_score=None):
        x_component, y_component = self._decompose(xy)
        return self._is_tabu(x_component, test_score) or self._is_tabu(
            y_component, test_score)

    def mark_good_move(self, xy, diff_with=None, cost=None):
        x, y = xy
        self.cell_memory[0][round(x, 4)] += 1
        self.cell_memory[0][round(y, 4)] += 1

    def restart(self):
        xy = []
        for i in xrange(2):
            for value, _ in self.cell_memory[i].most_common(6):
                if (i, value) not in self.tabu:
                    xy.append(value)
                    break
            else:
                raise StopIteration()
        return xy

    def observe(self, best_move, best_score, current_move, current_score):
        pass


def run_optimizer(optimizer_class, a, b, opt_xy, opt):

    optimizer = optimizer_class(a, b)

    assert round(optimizer.objective(opt_xy), 4) == round(opt, 4)

    baseline = optimizer.objective((a, b))
    best_score, best_ab = optimizer.optimize()

    assert best_score == optimizer.objective(best_ab)
    assert best_score <= baseline

    return round(best_score, 4)


MCCORMICK_X_MIN, MCCORMICK_X_MAX = (-1.5, 4)
MCCORMICK_Y_MIN, MCCORMICK_Y_MAX = (-3, 4)
MCCORMICK_OPT_X, MCCORMICK_OPT_Y = (-.54719, -1.54719)
# The official optimum result is 1.9133, however np's `sin` function produces
# this with the above xy.
MCCORMICK_OPTIMUM = -1.9132


@pytest.mark.optimize
@given(
    floats(
        min_value=MCCORMICK_X_MIN / 2, max_value=MCCORMICK_X_MAX / 2),
    floats(
        min_value=MCCORMICK_Y_MIN / 2, max_value=MCCORMICK_Y_MAX / 2))
@settings(max_examples=500)
def test_optimize_mccormick(a, b):
    class McCormickOptimizer(GenericOptimizer):

        x_min, x_max = MCCORMICK_X_MIN, MCCORMICK_X_MAX
        y_min, y_max = MCCORMICK_Y_MIN, MCCORMICK_Y_MAX

        def objective(self, xy):
            """
            Global minimum = -1.9133, at (-.54719, -1.54719)
            """
            x, y = xy
            return np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1

    best_score = run_optimizer(McCormickOptimizer, a, b,
                               (MCCORMICK_OPT_X, MCCORMICK_OPT_Y),
                               MCCORMICK_OPTIMUM)

    delta = best_score - round(MCCORMICK_OPTIMUM, 4)

    if delta <= .0001:
        event("excellent")
    elif delta < .001:
        event("very good")
    elif delta < .01:
        event("good")
    elif delta < .1:
        event("fair")
    else:
        event("bad")


BUKIN_X_MIN, BUKIN_X_MAX = (-15, -5)
BUKIN_Y_MIN, BUKIN_Y_MAX = (-3, 3)
BUKIN_OPT_X, BUKIN_OPT_Y = (-10, 1)
BUKIN_OPTIMUM = 0


@pytest.mark.optimize
@given(
    floats(
        min_value=BUKIN_X_MIN / 2, max_value=BUKIN_X_MAX / 2),
    floats(
        min_value=BUKIN_Y_MIN / 2, max_value=BUKIN_Y_MAX / 2))
@settings(max_examples=500)
def test_optimize_bukin(a, b):
    class BukinOptimizer(GenericOptimizer):

        x_min, x_max = BUKIN_X_MIN, BUKIN_X_MAX
        y_min, y_max = BUKIN_Y_MIN, BUKIN_Y_MAX

        def objective(self, xy):
            x, y = xy
            return 100 * np.sqrt(abs(y - .01 * x**2)) + .01 * abs(x + 10)

    best_score = run_optimizer(BukinOptimizer, a, b,
                               (BUKIN_OPT_X, BUKIN_OPT_Y), BUKIN_OPTIMUM)

    if best_score <= .1:
        event("excellent")
    elif best_score < 1:
        event("very good")
    elif best_score < 5:
        event("good")
    elif best_score < 10:
        event("fair")
    else:
        event("bad")


EGGHOLDER_MIN, EGGHOLDER_MAX = (-512, 512)
EGGHOLDER_OPT_X, EGGHOLDER_OPT_Y = (512, 404.2319)
EGGHOLDER_OPTIMUM = -959.6407


@pytest.mark.optimize
@given(
    floats(
        min_value=EGGHOLDER_MIN / 2, max_value=EGGHOLDER_MAX / 2),
    floats(
        min_value=EGGHOLDER_MIN / 2, max_value=EGGHOLDER_MAX / 2))
@settings(max_examples=500)
def test_optimize_eggholder(a, b):
    class EggholderOptimizer(GenericOptimizer):

        y_min = x_min = EGGHOLDER_MIN
        y_max = x_max = EGGHOLDER_MAX

        def objective(self, xy):
            x, y = xy
            return (-(y + 47) * np.sin(np.sqrt(abs(y + (x / 2.) + 47))) - x *
                    np.sin(np.sqrt(abs(x - (y + 47)))))

    best_score = run_optimizer(EggholderOptimizer, a, b,
                               (EGGHOLDER_OPT_X, EGGHOLDER_OPT_Y),
                               EGGHOLDER_OPTIMUM)

    delta = best_score - round(EGGHOLDER_OPTIMUM, 4)
    if delta <= 25:
        event("excellent")
    elif delta < 100:
        event("very good")
    elif delta < 250:
        event("good")
    elif delta < 500:
        event("fair")
    else:
        event("bad")
