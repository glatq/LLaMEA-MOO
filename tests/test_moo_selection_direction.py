"""Regression guard for the MOO fitness-sign fix.

The MOO data layer stores ``best_y = -HV`` -- a per-point *loss* for the
convergence machinery, where lower is better. Population selection, however,
*maximizes* fitness: every ``ESPopulation``/``Population`` sort uses
``reverse=True`` and keeps the highest-fitness individuals, and failed runs are
pinned to ``-inf``. The run score handed to selection (``EvaluatorResult.score``)
must therefore be ``+HV`` so that selection breeds *toward* higher hypervolume.

Before the fix, ``score = mean(best_y) = -mean(HV)``, so selection kept the
*lowest*-HV algorithms -- evolution drifted toward worse solutions and looked
indistinguishable from random search. These tests pin the corrected interaction
end-to-end: the score convention is ``-best_y`` (also asserted against the real
evaluator in ``test_moo_evaluator_with_pymoo_real`` /
``test_moo_constraints_data_layer``), and selection keeps the higher-HV runs.
"""

import numpy as np

from llamevol.individual import Individual
from llamevol.population import ESPopulation


def _moo_fitness(hv: float) -> float:
    """Population fitness for a MOO run, mirroring the evaluator.

    The evaluator stores ``best_y = -hv`` and sets ``score = -mean(best_y)``, so
    for a single repeat the population fitness is simply ``+hv``.
    """
    best_y = -hv
    return -float(np.mean([best_y]))


def test_moo_fitness_increases_with_hypervolume():
    # Higher HV must map to higher (more positive) fitness, the quantity
    # selection maximizes.
    assert _moo_fitness(0.8) == 0.8
    assert _moo_fitness(0.1) == 0.1
    assert _moo_fitness(0.8) > _moo_fitness(0.1)


def _select(specs, n_parent):
    pop = ESPopulation(
        n_parent=n_parent,
        n_parent_per_offspring=1,
        n_offspring=max(1, n_parent),
        use_elitism=True,
    )
    for name, hv in specs:
        ind = Individual()
        ind.name = name
        ind.fitness = _moo_fitness(hv)
        pop.add_individual(ind, generation=0)
    pop.select_next_generation()
    return {pop.individuals[i].name for i in pop.selected_generations[-1]}


def test_elitist_selection_keeps_highest_hypervolume():
    specs = [
        ("hv0.9_best", 0.9),
        ("hv0.7", 0.7),
        ("hv0.2", 0.2),
        ("hv0.1_worst", 0.1),
    ]
    # 2 + 2 elitism keeps the two highest-HV algorithms, never the worst.
    survivors = _select(specs, n_parent=2)
    assert survivors == {"hv0.9_best", "hv0.7"}


def test_one_plus_one_keeps_higher_hypervolume():
    # The (1+1) case the bug hit hardest: parent vs offspring -> keep the better.
    survivors = _select([("parent_hv0.6", 0.6), ("offspring_hv0.8", 0.8)], n_parent=1)
    assert survivors == {"offspring_hv0.8"}
