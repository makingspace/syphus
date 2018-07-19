import collections
import math
import random

import pytest
import syphus
from hypothesis import event, given, settings
from hypothesis.strategies import floats


class GenericOptimizer(syphus.Optimizer):
    def __init__(self, a, b):
        super(GenericOptimizer, self).__init__()
        self.a = a
        self.b = b

        self.x_scale = (self.x_max - self.x_min) / 7.
        self.y_scale = (self.y_max - self.y_min) / 7.
        self.MAX_TABU_SIZE = 18

        self.cell_memory = [collections.Counter(), collections.Counter()]

    def hash(self, xy):
        return xy

    def initial_state(self):
        return (self.a, self.b)

    def _step(self, xy):
        x, y = xy

        if random.choice([True, False]):
            step_size = random.uniform(-self.x_scale, self.x_scale)
            x += step_size
            if x < self.x_min or x > self.x_max:
                x = self.a
        else:
            step_size2 = random.uniform(-self.y_scale, self.y_scale)
            y += step_size2
            if y < self.y_min or y > self.y_max:
                y = self.a

        return (x, y)

    def get_neighborhood(self, xy):
        return [self._step(xy) for _ in range(20)]

    def _decompose(self, xy):
        x, y = xy
        return ((0, round(x, 3)), (1, round(y, 3)))

    def mark_tabu(self, xy, diff_with=None, cost=None):
        decomposed = self._decompose(xy)

        for i in range(1):
            value = xy[i]
            other_value = diff_with[i]
            delta = abs(other_value - value)
            ratio = delta / (self.x_scale, self.y_scale)[i]

            def lifetime(ratio):
                return max(1, int(100 * ratio))

            tabu = syphus.Tabu(
                value=decomposed[i], lifetime=lifetime(ratio), criterion=cost
            )
            self.tabu.append(tabu)

    def _is_tabu(self, move, test_score):
        for tabu in self.tabu:
            if tabu.value == move:
                return test_score > tabu.criterion
        return False

    def is_tabu(self, xy, test_score=None):
        x_component, y_component = self._decompose(xy)
        result = self._is_tabu(x_component, test_score) or self._is_tabu(
            y_component, test_score
        )
        return result

    def handle_non_optimal_best(self, best_move, current_solution, best_score):
        pass

    def mark_good_move(self, xy, diff_with=None, cost=None):
        decomposed_x, decomposed_y = self._decompose(xy)
        self.cell_memory[decomposed_x[0]][decomposed_x[1]] += 1
        self.cell_memory[decomposed_y[0]][decomposed_y[1]] += 1

    def restart(self):
        xy = []
        for i in range(2):
            for value, _ in self.cell_memory[i].most_common(6):
                if not self._is_tabu((i, value), float("inf")):
                    xy.append(value)
                    del self.cell_memory[i][value]
                    break
            else:
                raise StopIteration()
        return xy


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
# The official optimum result is 1.9133, however the `sin` function produces
# this with the above xy.
MCCORMICK_OPTIMUM = -1.9132


@pytest.mark.optimize
@given(
    floats(min_value=MCCORMICK_X_MIN / 2, max_value=MCCORMICK_X_MAX / 2),
    floats(min_value=MCCORMICK_Y_MIN / 2, max_value=MCCORMICK_Y_MAX / 2),
)
@settings(max_examples=500)
def test_optimize_mccormick(a, b):
    class McCormickOptimizer(GenericOptimizer):

        x_min, x_max = MCCORMICK_X_MIN, MCCORMICK_X_MAX
        y_min, y_max = MCCORMICK_Y_MIN, MCCORMICK_Y_MAX

        def objective(self, xy, **kwargs):
            """
            Global minimum = -1.9133, at (-.54719, -1.54719)
            """
            x, y = xy
            return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

    best_score = run_optimizer(
        McCormickOptimizer, a, b, (MCCORMICK_OPT_X, MCCORMICK_OPT_Y), MCCORMICK_OPTIMUM
    )

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
    floats(min_value=BUKIN_X_MIN / 2, max_value=BUKIN_X_MAX / 2),
    floats(min_value=BUKIN_Y_MIN / 2, max_value=BUKIN_Y_MAX / 2),
)
@settings(max_examples=500)
def test_optimize_bukin(a, b):
    class BukinOptimizer(GenericOptimizer):

        x_min, x_max = BUKIN_X_MIN, BUKIN_X_MAX
        y_min, y_max = BUKIN_Y_MIN, BUKIN_Y_MAX

        def objective(self, xy, **kwargs):
            x, y = xy
            return 100 * math.sqrt(abs(y - .01 * x ** 2)) + .01 * abs(x + 10)

    best_score = run_optimizer(
        BukinOptimizer, a, b, (BUKIN_OPT_X, BUKIN_OPT_Y), BUKIN_OPTIMUM
    )

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
    floats(min_value=EGGHOLDER_MIN / 2, max_value=EGGHOLDER_MAX / 2),
    floats(min_value=EGGHOLDER_MIN / 2, max_value=EGGHOLDER_MAX / 2),
)
@settings(max_examples=500)
def test_optimize_eggholder(a, b):
    class EggholderOptimizer(GenericOptimizer):

        y_min = x_min = EGGHOLDER_MIN
        y_max = x_max = EGGHOLDER_MAX

        def objective(self, xy, **kwargs):
            x, y = xy
            return -(y + 47) * math.sin(
                math.sqrt(abs(y + (x / 2.) + 47))
            ) - x * math.sin(math.sqrt(abs(x - (y + 47))))

    best_score = run_optimizer(
        EggholderOptimizer, a, b, (EGGHOLDER_OPT_X, EGGHOLDER_OPT_Y), EGGHOLDER_OPTIMUM
    )

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
