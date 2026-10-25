"""Dense, metric-faithful training reward for the A3 scenarios.

`DefaultRewards` pays almost everything as lumps at the DONE transition or at
episode end; the corridor runs 796 steps and the shipped discount is
gamma=0.999, so such a lump survives at 0.999**796 ~= 0.45 of its size and the
gradient is dominated by which episode ended, not by what the train did. A
policy therefore has no local signal for dispatching, serving a stop, or
avoiding a deadlock until it is far too late.

This module does **not** re-derive the metric.  It defines a potential

    R_a(t) = the cost the metric would charge agent ``a`` if the episode ended
             at step ``t``

by calling the metric's own `_agent_done_or_max_episode_steps_reward`, and
emits the per-step difference

    r_t = sum_a [ R_a(t) - R_a(t-1) ].

Because ``R`` is evaluated by the same code path that scores the episode, every
condition of the metric is inherited rather than re-written: the
``is_on_map_state()`` gate on TARGET_NOT_REACHED, the ``STOPPED``-required
service test, the minimum-penalty window over visits, the retroactive early
departure, and the cancellation lump.  Summed over an episode the reward is
exactly the metric total (``sum(r_t) = R(T) - R(0)``), so the only thing left
to tune is how long the policy trains.

Notes
-----
* ``R(0)`` is 0 by definition (all agents off-map, no arrivals recorded), so
  ``sum(r_t) = R(T)``.
* ``end_of_episode_reward`` returns 0: the potential already contains it.
* ``step_reward`` is a no-op until `begin_episode` is called, so the harness'
  reference rollout used to calibrate deadlines stays cheap.
* With gamma < 1 this is a (1 - gamma)-biased potential (Ng et al. 1999); at
  gamma=0.999 and integer ``R`` the residual is below one metric unit per
  episode, i.e. below the resolution of a single late step.
"""

from __future__ import annotations

from typing import Dict

from flatland.envs.rewards import BaseDefaultRewards


class PotentialRewards(BaseDefaultRewards):
    """Metric potential shaping with per-term logging (see module docstring)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._primed = False
        self._prev_terms: Dict[int, Dict[str, float]] = {}
        self._wp_cache: Dict[int, list] = {}
        self.last_delta: Dict[int, Dict[str, float]] = {}
        self.totals: Dict[int, Dict[str, float]] = {}

    # ------------------------------------------------------------------ setup
    def begin_episode(self, env) -> None:
        """Reset per-episode bookkeeping.  Call immediately after ``env.reset()``."""
        self.arrivals.clear()
        self.departures.clear()
        self.states.clear()
        self._wp_cache.clear()
        self._prev_terms.clear()
        self.last_delta.clear()
        self.totals.clear()
        zero = dict.fromkeys(self._cached_default_penalty_values, 0.0)
        for agent in env.agents:
            self._prev_terms[agent.handle] = dict(zero)
            self.totals[agent.handle] = dict(zero)
        self._primed = True

    # waypoint sanitisation is pure but rebuilt on every call; cache per episode
    def _sanitize_waypoints(self, agent_waypoints):  # type: ignore[override]
        key = id(agent_waypoints)
        cached = self._wp_cache.get(key)
        if cached is None:
            cached = BaseDefaultRewards._sanitize_waypoints(agent_waypoints)
            self._wp_cache[key] = cached
        return cached

    # ---------------------------------------------------------------- rewards
    def potential(self, agent, distance_map, elapsed_steps: int) -> Dict[str, float]:
        """``R_a(t)`` as a per-term dict, without touching the running state."""
        terms = self.empty()
        self._agent_done_or_max_episode_steps_reward(
            agent, distance_map, elapsed_steps, terms
        )
        return terms

    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps):  # type: ignore[override]
        if not self._primed:
            return 0.0
        # same booking as the metric (arrivals / departures / states)
        super().step_reward(agent, agent_transition_data, distance_map, elapsed_steps)
        terms = self.potential(agent, distance_map, elapsed_steps)
        prev = self._prev_terms[agent.handle]
        delta = {k: v - prev.get(k, 0.0) for k, v in terms.items()}
        self._prev_terms[agent.handle] = terms
        self.last_delta[agent.handle] = delta
        totals = self.totals[agent.handle]
        for k, v in delta.items():
            totals[k] = totals.get(k, 0.0) + v
        return float(sum(delta.values()))

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps):  # type: ignore[override]
        # already accounted for by the potential at the final step
        return 0.0

    def cumulate(self, *rewards):  # type: ignore[override]
        """Sum whatever the env hands us: the per-term dict from ``empty()`` on
        the first step, floats afterwards."""
        total = 0.0
        for reward in rewards:
            if reward is None:
                continue
            if isinstance(reward, dict):
                total += float(sum(reward.values()))
            else:
                total += float(reward)
        return total

    # --------------------------------------------------------------- analysis
    def term_table(self) -> Dict[str, float]:
        """Per-term totals across all agents (for the identity test / logging)."""
        out: Dict[str, float] = {}
        for handle_terms in self.totals.values():
            for k, v in handle_terms.items():
                out[k] = out.get(k, 0.0) + v
        return out


class ShapedMetricRewards(PotentialRewards):
    """Metric potential + the two shaping terms every Flatland winner used.

    From the 2020 competition: all winning RL entries paid a **target bonus** and
    a **deadlock penalty**, and the documented failure of weaker entries was
    "over-penalising entering the grid -> agents never enter".  Our metric has
    exactly that pathology built in (cancelling costs -shortest_path_time, while
    dispatching into a jam costs -current_delay, an order of magnitude more), so
    the arrival bonus is not a nicety: it is what makes dispatching worth trying.

    ``r_t = metric_potential_delta + arrival_bonus*[reached DONE]
                                    - stuck_penalty*[on map, tried to move, did not]``

    Broken-down trains are excluded from the stuck term (that is the environment,
    not the decision), and stops served by the env's force-stop reflex are not
    "stuck" either - only a MOVING state with an unchanged position counts.
    """

    def __init__(self, arrival_bonus: float = 50.0, stuck_penalty: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.arrival_bonus = float(arrival_bonus)
        self.stuck_penalty = float(stuck_penalty)
        self._last_position = {}
        self._last_state = {}

    def begin_episode(self, env) -> None:  # type: ignore[override]
        super().begin_episode(env)
        self._last_position.clear()
        self._last_state.clear()

    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps):  # type: ignore[override]
        from flatland.envs.step_utils.states import TrainState

        reward = super().step_reward(agent, agent_transition_data, distance_map, elapsed_steps)
        if not self._primed:
            return reward

        configuration = agent.current_configuration
        position = (
            None
            if configuration is None or configuration[0] is None
            else (int(configuration[0][0]), int(configuration[0][1]))
        )
        previous_position = self._last_position.get(agent.handle)
        previous_state = self._last_state.get(agent.handle)

        if agent.state == TrainState.DONE and previous_state not in (None, TrainState.DONE):
            reward += self.arrival_bonus
        elif (
            position is not None
            and previous_position is not None
            and position == previous_position
            and agent.state == TrainState.MOVING
        ):
            reward -= self.stuck_penalty

        self._last_position[agent.handle] = position
        self._last_state[agent.handle] = agent.state
        return reward
