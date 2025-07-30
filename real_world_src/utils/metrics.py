import numpy as np

def brier_score(true_goal, p_dist):
    """
    Multi-class Brier score given a dict goal→prob.
    """
    score = 0.0
    for g, p in p_dist.items():
        y = 1.0 if g == true_goal else 0.0
        score += (p - y) ** 2
    return float(score)


def brier_along_path(path, true_goal, posteriors_by_step, goals):
    """
    Compute Brier scores at 0%, 10%, …, 100% of the path.
    
    Parameters
    ----------
    path : list
        The sequence of states (or state-action tuples).
    true_goal : hashable
        The ground-truth goal for this trajectory.
    posteriors_by_step : dict[int, dict] or list[dict]
        Either
          * a dict mapping step index 1..n → posterior dict, OR
          * a list of posterior dicts in step-order (len = n).
        Each posterior dict maps each goal → P(goal | trajectory up to that step).
    
    Returns
    -------
    dict[float, float]
        Mapping fraction → Brier score.
    """
    n = len(path) - 1
    if n < 1:
        raise ValueError("Path must have at least 2 states to define transitions")

    # uniform prior for t=0
    uniform = {g: 1.0/len(goals) for g in goals}

    results = []
    for frac in np.linspace(0, 1, 11):  # 0.0,0.1,...,1.0
        t = int(np.floor(frac * n))
        if t == 0:
            p_dist = uniform
        else:
            p_dist = posteriors_by_step[t-1]
        results.append(brier_score(true_goal, p_dist))

    return results

def accuracy_along_path(path, true_goal, posteriors_by_step, goals):
    """
    Compute “accuracy” (did we pick the MAP goal?) at 0%, 10%, …, 100% of the path.
    
    Parameters
    ----------
    path : list
        The sequence of states (or state–action tuples).
    true_goal : hashable
        The ground-truth goal for this trajectory.
    posteriors_by_step : dict[int, dict] or list[dict]
        Either
          * a dict mapping step index 1..n → posterior dict, OR
          * a list of posterior dicts in step-order (len = n).
        Each posterior dict maps each goal → P(goal | trajectory up to that step).
    goals : list
        The list of all possible goals (needed only for the t=0 “uniform” prior).
    
    Returns
    -------
    list of float
        A length-11 list of 0/1 values for fractions [0.0,0.1,…,1.0].
        At t=0 (no observations) we always return 0 (could equally choose 1/len(goals)).
    """
    n = len(path) - 1
    if n < 1:
        raise ValueError("Path must have ≥2 states")

    # Precompute a “uniform” posterior for t=0
    uniform = {g: 1.0/len(goals) for g in goals}

    accs = []
    for frac in np.linspace(0, 1, 11):
        t = int(np.floor(frac * n))
        if t == 0:
            # no info yet → we consider “incorrect”
            accs.append(0.0)
        else:
            # posteriors_by_step is 0‐indexed list or 1‐indexed dict:
            if isinstance(posteriors_by_step, dict):
                post = posteriors_by_step[t]
            else:
                post = posteriors_by_step[t-1]
            # find all goals achieving the max
            maxp = max(post.values())

            true_goal = str(true_goal) # Dict stores ids as strings

            # if true_goal is one of the maxima, count as correct
            accs.append(1.0 if post[true_goal] >= maxp else 0.0)

    return accs

def fooled_score(true_goal, p_dist, false_goal):
    """
    Fooled scores counts .
    """
    prob_true_goal = p_dist[true_goal]
    prob_false_goal = p_dist[false_goal]
    return prob_false_goal - prob_true_goal


def fooled_along_path(path, true_goal, false_goal, posteriors_by_step, goals):
    """
    Compute Brier scores at 0%, 10%, …, 100% of the path.
    
    Parameters
    ----------
    path : list
        The sequence of states (or state-action tuples).
    true_goal : hashable
        The ground-truth goal for this trajectory.
    posteriors_by_step : dict[int, dict] or list[dict]
        Either
          * a dict mapping step index 1..n → posterior dict, OR
          * a list of posterior dicts in step-order (len = n).
        Each posterior dict maps each goal → P(goal | trajectory up to that step).
    
    Returns
    -------
    dict[float, float]
        Mapping fraction → Brier score.
    """
    n = len(path) - 1
    if n < 1:
        raise ValueError("Path must have at least 2 states to define transitions")

    # uniform prior for t=0
    uniform = {g: 1.0/len(goals) for g in goals}

    results = []
    for frac in np.linspace(0, 1, 11):  # 0.0,0.1,...,1.0
        t = int(np.floor(frac * n))
        if t == 0:
            p_dist = uniform
        else:
            p_dist = posteriors_by_step[t-1]
        results.append(p_dist[false_goal])

    return results