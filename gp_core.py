import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from gp_tree import FUNCTION_SET, Individual, Node, collect_nodes_with_parents, generate_random_tree, generate_terminal


EPS = 1e-8


@dataclass
class GPConfig:
    population_size: int = 150
    generations: int = 60
    max_init_depth: int = 4
    max_tree_depth: int = 9
    tournament_size: int = 4
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    elitism: int = 1
    const_min: float = -5.0
    const_max: float = 5.0
    parsimony_lambda: float = 0.001
    structure_depth_penalty: float = 0.01
    preferred_min_depth: int = 3
    preferred_max_depth: int = 7
    mutation_max_subtree_depth: int = 4
    global_search_generations: int = 30
    no_change_window_generations: int = 10
    global_area_cutoff_depth: int = 9
    global_similarity_threshold: float = 0.75
    transferred_global_init_depth: int = 3


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.where(np.abs(y_true) < EPS, EPS, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / den)) * 100.0)


def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray, bound: float) -> float:
    hits = np.abs(y_true - y_pred) <= bound
    return float(np.mean(hits))


def depth_penalty(depth: int, preferred_min_depth: int, preferred_max_depth: int) -> int:
    if depth < preferred_min_depth:
        return preferred_min_depth - depth
    if depth > preferred_max_depth:
        return depth - preferred_max_depth
    return 0


def evaluate_fitness(ind: Individual, x_train: np.ndarray, y_train: np.ndarray, cfg: GPConfig) -> float:
    pred = ind.tree.evaluate(x_train)
    data_fit = rmse(y_train, pred)
    complexity = cfg.parsimony_lambda * ind.tree.size()
    structural = cfg.structure_depth_penalty * depth_penalty(
        ind.tree.depth(), cfg.preferred_min_depth, cfg.preferred_max_depth
    )
    return data_fit + complexity + structural


def validation_score(ind: Individual, x_val: np.ndarray, y_val: np.ndarray) -> float:
    if len(x_val) == 0:
        return ind.fitness
    pred = ind.tree.evaluate(x_val)
    return rmse(y_val, pred)


def _structure_counts(population: Sequence[Individual]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ind in population:
        sig = ind.tree.structure_signature()
        counts[sig] = counts.get(sig, 0) + 1
    return counts


def _structure_token_set(node: Node) -> set:
    if node.kind in {"var", "const"}:
        return {"T"}
    tokens = {f"{node.value}/{len(node.children)}"}
    for child in node.children:
        tokens.update(_structure_token_set(child))
    return tokens


def structural_similarity(a: Node, b: Node) -> float:
    tokens_a = _structure_token_set(a)
    tokens_b = _structure_token_set(b)
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(tokens_a & tokens_b) / len(union)


def population_similarity(
    population: Sequence[Individual],
    rng: random.Random,
    max_pairs: int = 30,
) -> float:
    if len(population) < 2:
        return 1.0

    sample_count = min(max_pairs, (len(population) * (len(population) - 1)) // 2)
    similarities: List[float] = []
    for _ in range(sample_count):
        i, j = rng.sample(range(len(population)), 2)
        similarities.append(structural_similarity(population[i].tree, population[j].tree))
    return float(np.mean(similarities)) if similarities else 1.0


def select_mate(
    population: Sequence[Individual],
    parent1: Individual,
    k: int,
    rng: random.Random,
    structure_counts: Dict[str, int],
    similarity_threshold: float,
    search_mode: str,
    attempts: int = 8,
) -> Individual:
    fallback = tournament_selection(population, k, rng, structure_counts)
    for _ in range(attempts):
        candidate = tournament_selection(population, k, rng, structure_counts)
        sim = structural_similarity(parent1.tree, candidate.tree)
        if search_mode == "global" and sim <= similarity_threshold:
            return candidate
        if search_mode == "local" and sim >= similarity_threshold:
            return candidate
        fallback = candidate
    return fallback


def tournament_selection(
    population: Sequence[Individual],
    k: int,
    rng: random.Random,
    structure_counts: Dict[str, int],
) -> Individual:
    candidates = rng.sample(list(population), k=min(k, len(population)))

    def rank_tuple(ind: Individual) -> Tuple[float, int, int]:
        sig = ind.tree.structure_signature()
        return (ind.fitness, structure_counts.get(sig, 0), ind.tree.size())

    return min(candidates, key=rank_tuple)


def subtree_crossover(p1: Node, p2: Node, rng: random.Random, max_depth: int) -> Tuple[Node, Node]:
    c1 = p1.clone()
    c2 = p2.clone()

    n1 = collect_nodes_with_parents(c1)
    n2 = collect_nodes_with_parents(c2)

    s1, parent1, idx1 = rng.choice(n1)
    compatible = [n for n in n2 if len(n[0].children) == len(s1.children)]
    if compatible:
        s2, parent2, idx2 = rng.choice(compatible)
    else:
        s2, parent2, idx2 = rng.choice(n2)

    swap1 = s1.clone()
    swap2 = s2.clone()

    if parent1 is None:
        c1 = swap2
    else:
        parent1.children[idx1] = swap2

    if parent2 is None:
        c2 = swap1
    else:
        parent2.children[idx2] = swap1

    if c1.depth() > max_depth:
        c1 = p1.clone()
    if c2.depth() > max_depth:
        c2 = p2.clone()

    return c1, c2


def _point_mutation(node: Node, rng: random.Random, const_min: float, const_max: float) -> None:
    if node.kind == "const":
        node.value = float(np.clip(node.value + rng.uniform(-1.0, 1.0), const_min, const_max))
        return
    if node.kind == "var":
        return

    arity = len(node.children)
    ops = [op for op, a in FUNCTION_SET.items() if a == arity and op != node.value]
    if ops:
        node.value = rng.choice(ops)


def subtree_mutation(
    tree: Node,
    num_features: int,
    rng: random.Random,
    cfg: GPConfig,
    subtree_max_depth: int = None,
) -> Node:
    mutated = tree.clone()
    all_nodes = collect_nodes_with_parents(mutated)

    if mutated.depth() >= cfg.max_tree_depth - 1 and rng.random() < 0.6:
        target, parent, idx = rng.choice(all_nodes)
        replacement = generate_terminal(num_features, rng, cfg.const_min, cfg.const_max)
        if parent is None:
            mutated = replacement
        else:
            parent.children[idx] = replacement
        return mutated

    if rng.random() < 0.4:
        target, _, _ = rng.choice(all_nodes)
        _point_mutation(target, rng, cfg.const_min, cfg.const_max)
        return mutated

    _, parent, idx = rng.choice(all_nodes)
    max_subtree_depth = cfg.mutation_max_subtree_depth if subtree_max_depth is None else subtree_max_depth
    depth_cap = max(2, min(max_subtree_depth, cfg.max_tree_depth - 1))
    new_subtree = generate_random_tree(
        num_features=num_features,
        max_depth=depth_cap,
        rng=rng,
        const_min=cfg.const_min,
        const_max=cfg.const_max,
        method="grow",
    )

    if parent is None:
        mutated = new_subtree
    else:
        parent.children[idx] = new_subtree

    if mutated.depth() > cfg.max_tree_depth:
        return tree.clone()

    return mutated


def initialize_population(num_features: int, cfg: GPConfig, rng: random.Random) -> List[Individual]:
    pop: List[Individual] = []
    depth_cycle = list(range(2, cfg.max_init_depth + 1))
    half = cfg.population_size // 2

    for i in range(cfg.population_size):
        depth = depth_cycle[i % len(depth_cycle)]
        method = "full" if i < half else "grow"
        tree = generate_random_tree(
            num_features=num_features,
            max_depth=depth,
            rng=rng,
            const_min=cfg.const_min,
            const_max=cfg.const_max,
            method=method,
        )
        pop.append(Individual(tree=tree))

    return pop


def inject_transferred_global_area(
    population: Sequence[Individual],
    best: Individual,
    num_features: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: GPConfig,
    rng: random.Random,
) -> List[Individual]:
    pop_sorted = sorted(population, key=lambda i: i.fitness)
    elites = pop_sorted[: cfg.elitism]
    transfer_count = max(1, int(0.2 * len(population)))

    new_population: List[Individual] = [Individual(tree=e.tree.clone(), fitness=e.fitness) for e in elites]
    for _ in range(transfer_count):
        tree = subtree_mutation(
            tree=best.tree,
            num_features=num_features,
            rng=rng,
            cfg=cfg,
            subtree_max_depth=max(2, cfg.transferred_global_init_depth),
        )
        child = Individual(tree=tree)
        child.fitness = evaluate_fitness(child, x_train, y_train, cfg)
        new_population.append(child)

    for ind in pop_sorted[cfg.elitism :]:
        if len(new_population) >= len(population):
            break
        new_population.append(Individual(tree=ind.tree.clone(), fitness=ind.fitness))

    return new_population[: len(population)]


def evolve_one_run(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: GPConfig,
    seed: int,
    hit_bound: float = 0.01,
) -> Tuple[Individual, dict, List[dict]]:
    rng = random.Random(seed)
    num_features = x_train.shape[1]

    population = initialize_population(num_features, cfg, rng)
    for ind in population:
        ind.fitness = evaluate_fitness(ind, x_train, y_train, cfg)

    best = min(population, key=lambda i: validation_score(i, x_val, y_val))
    generation_rows: List[dict] = []
    best_val_score = validation_score(best, x_val, y_val)
    no_improve_counter = 0

    search_mode = "global"
    mode_generation_counter = 0
    mode_switches = 0

    local_mutation_rate = min(0.95, cfg.mutation_rate + 0.10)
    local_crossover_rate = max(0.05, cfg.crossover_rate - 0.10)
    local_tournament_size = max(2, cfg.tournament_size - 1)

    global_cutoff_depth = min(cfg.global_area_cutoff_depth, cfg.max_tree_depth)
    local_cutoff_depth = cfg.max_tree_depth

    for gen_idx in range(cfg.generations):
        structure_counts = _structure_counts(population)
        mode_generation_counter += 1

        if search_mode == "global":
            active_mutation = cfg.mutation_rate
            active_crossover = cfg.crossover_rate
            active_tournament = cfg.tournament_size
            active_cutoff_depth = global_cutoff_depth
        else:
            active_mutation = local_mutation_rate
            active_crossover = local_crossover_rate
            active_tournament = local_tournament_size
            active_cutoff_depth = local_cutoff_depth

        new_pop: List[Individual] = []
        elites = sorted(population, key=lambda i: i.fitness)[: cfg.elitism]
        new_pop.extend(Individual(tree=e.tree.clone(), fitness=e.fitness) for e in elites)

        while len(new_pop) < cfg.population_size:
            parent1 = tournament_selection(population, active_tournament, rng, structure_counts)

            if rng.random() < active_crossover:
                parent2 = select_mate(
                    population=population,
                    parent1=parent1,
                    k=active_tournament,
                    rng=rng,
                    structure_counts=structure_counts,
                    similarity_threshold=cfg.global_similarity_threshold,
                    search_mode=search_mode,
                )
                child_trees = list(subtree_crossover(parent1.tree, parent2.tree, rng, active_cutoff_depth))
            else:
                child_trees = [parent1.tree.clone()]

            for tree in child_trees:
                if rng.random() < active_mutation:
                    tree = subtree_mutation(
                        tree=tree,
                        num_features=num_features,
                        rng=rng,
                        cfg=cfg,
                        subtree_max_depth=cfg.mutation_max_subtree_depth,
                    )

                if tree.depth() > active_cutoff_depth:
                    tree = parent1.tree.clone()

                child = Individual(tree=tree)
                child.fitness = evaluate_fitness(child, x_train, y_train, cfg)
                new_pop.append(child)
                if len(new_pop) >= cfg.population_size:
                    break

        population = new_pop
        gen_best = min(population, key=lambda i: validation_score(i, x_val, y_val))
        gen_best_val = validation_score(gen_best, x_val, y_val)
        if gen_best_val < best_val_score:
            best = Individual(tree=gen_best.tree.clone(), fitness=gen_best.fitness)
            best_val_score = gen_best_val
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        pop_similarity = population_similarity(population, rng)

        optimum_reached = (
            no_improve_counter >= cfg.no_change_window_generations
            and pop_similarity >= cfg.global_similarity_threshold
        )

        if search_mode == "global":
            if optimum_reached or mode_generation_counter >= cfg.global_search_generations:
                search_mode = "local"
                mode_generation_counter = 0
                no_improve_counter = 0
                mode_switches += 1
                population = inject_transferred_global_area(
                    population=population,
                    best=best,
                    num_features=num_features,
                    x_train=x_train,
                    y_train=y_train,
                    cfg=cfg,
                    rng=rng,
                )
        else:
            if optimum_reached or mode_generation_counter >= cfg.no_change_window_generations:
                search_mode = "global"
                mode_generation_counter = 0
                no_improve_counter = 0
                mode_switches += 1

        avg_standardized_fitness = statistics.fmean(ind.fitness for ind in population)
        avg_tree_size = statistics.fmean(ind.tree.size() for ind in population)
        unique_structure_count = len({ind.tree.structure_signature() for ind in population})
        variety_percentage = (unique_structure_count / len(population)) * 100.0 if population else 0.0

        best_test_pred = best.tree.evaluate(x_test)
        best_hit_ratio = hit_ratio(y_test, best_test_pred, hit_bound)

        generation_rows.append(
            {
                "generation": gen_idx + 1,
                "average_standardized_fitness": avg_standardized_fitness,
                "average_tree_size": avg_tree_size,
                "variety_percentage": variety_percentage,
                "average_hit_ratio": best_hit_ratio,
                "search_mode": search_mode,
                "population_similarity": pop_similarity,
            }
        )

    train_pred = best.tree.evaluate(x_train)
    val_pred = best.tree.evaluate(x_val) if len(x_val) > 0 else np.asarray([])
    test_pred = best.tree.evaluate(x_test)

    metrics = {
        "train_rmse": rmse(y_train, train_pred),
        "val_rmse": rmse(y_val, val_pred) if len(x_val) > 0 else 0.0,
        "test_rmse": rmse(y_test, test_pred),
        "test_mae": mae(y_test, test_pred),
        "test_mape": mape(y_test, test_pred),
        "test_hit_ratio": hit_ratio(y_test, test_pred, hit_bound),
        "hit_bound_used": hit_bound,
        "tree_size": best.tree.size(),
        "tree_depth": best.tree.depth(),
        "expression": best.tree.to_string(),
        "mode_switches": mode_switches,
    }

    return best, metrics, generation_rows


def summarize_runs(run_rows: List[dict]) -> dict:
    def values(col_name: str) -> List[float]:
        return [float(r[col_name]) for r in run_rows]

    test_rmses = values("test_rmse")
    test_maes = values("test_mae")
    test_mapes = values("test_mape")
    test_hit_ratios = values("test_hit_ratio")
    runtimes = values("runtime_seconds")

    best_idx = int(np.argmin(np.asarray(test_rmses)))

    return {
        "num_runs": len(run_rows),
        "test_rmse_mean": statistics.fmean(test_rmses),
        "test_rmse_std": statistics.pstdev(test_rmses) if len(test_rmses) > 1 else 0.0,
        "test_rmse_best": min(test_rmses),
        "test_mae_mean": statistics.fmean(test_maes),
        "test_mae_std": statistics.pstdev(test_maes) if len(test_maes) > 1 else 0.0,
        "test_mape_mean": statistics.fmean(test_mapes),
        "test_mape_std": statistics.pstdev(test_mapes) if len(test_mapes) > 1 else 0.0,
        "test_hit_ratio_mean": statistics.fmean(test_hit_ratios),
        "test_hit_ratio_std": statistics.pstdev(test_hit_ratios) if len(test_hit_ratios) > 1 else 0.0,
        "test_hit_ratio_best": max(test_hit_ratios),
        "hit_bound_used": run_rows[0]["hit_bound_used"],
        "runtime_mean_seconds": statistics.fmean(runtimes),
        "runtime_std_seconds": statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0,
        "runtime_total_seconds": float(np.sum(runtimes)),
        "best_run_index_1based": best_idx + 1,
        "best_expression": run_rows[best_idx]["expression"],
    }
