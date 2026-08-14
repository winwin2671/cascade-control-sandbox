"""Residual RL wrappers — xinji's model-based residual policy approach.

The RL agent outputs a 2D residual (V-33 drain valve + heater correction) on top
of a model-based feedforward baseline. The feedforward inverts the plant model to
find the steady-state action for the reference target. An inline PID provides
level-tracking feedback for the upstream actuators (pump, V-12). The RL only
adjusts the 2 hardest actuators (V-33 + heater) — the ones the model+PID struggle
with (Tank3 level + the under-actuated temperature cascade).

This is a Gymnasium Wrapper that sits between the RL agent and the existing
CascadeBridgeEnv / AIOGymNativeEnv. The agent sees:
  - action: 2D residual [-1, 1] (V-33 + heater)
  - observation: 17D normalized [0, 1] (output + reference + prev_action)
  - reward: regulation (tracking + 0.1 × feedforward)

Adapted from xinji's AIO-Gym v0.2 control.py + definition.py.
"""
from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---- Constants (from xinji's TANK3_INTERNAL_CONTROL + definition.py) ----
_PUMP_KP = 0.08           # pump PID gain (Tank1 level tracking)
_V12_KP = 0.08            # V-12 PID gain (Tank2 level tracking)
_LEVEL_TOL = 0.01         # level tolerance for PID scaling, m
_MAX_CORRECTION = 0.25    # max PID correction per actuator
_FEEDFORWARD_WEIGHT = 0.1  # weight on the feedforward penalty in the reward
_OUTPUT_SCALES = np.asarray((0.5, 0.5, 0.5, 100.0, 100.0, 100.0))   # obs normalization
_ERROR_SCALES = np.asarray((0.4, 0.4, 0.4, 10.0, 10.0, 10.0))      # reward normalization


def regulation_reward(levels, temps, reference, action, model_optimal_action):
    """Xinji's regulation reward: tracking + feedforward penalty.

    tracking = mean(((output − reference) / error_scales)²)
    feedforward = 0.1 × sum((action − model_optimal)²)
    reward = -(tracking + feedforward)
    """
    output = np.concatenate([np.asarray(levels[:3]), np.asarray(temps[:3])])
    tracking = float(np.mean(((output - np.asarray(reference)) / _ERROR_SCALES) ** 2))
    feedforward = _FEEDFORWARD_WEIGHT * float(np.sum(
        (np.asarray(action) - np.asarray(model_optimal_action)) ** 2))
    return -(tracking + feedforward)


class RegulationRewardWrapper(gym.Wrapper):
    """Override any env's reward with xinji's regulation reward (tracking + feedforward).
    Keeps the existing action space (5D direct). This eliminates the reward-drift
    between the numpy track (economic) and the modbus track (tracking) by giving both
    the same regulation reward.

    The two tracks expose levels/temps at DIFFERENT obs indices, so pass them in:
      numpy  (grouped     [h1,h2,h3,T1,T2,T3,...]): level_idx=(0,1,2), temp_idx=(3,4,5)
      modbus (interleaved [h1,T1,h2,T2,h3,T3,...]): level_idx=(0,2,4), temp_idx=(1,3,5)
    """

    def __init__(self, env, model, reference, level_idx=(0, 2, 4), temp_idx=(1, 3, 5)):
        super().__init__(env)
        self.model = model
        self.reference = np.asarray(reference, dtype=float)
        self.level_idx = list(level_idx)
        self.temp_idx = list(temp_idx)
        self._model_optimal = tracking_steady_state_action(model, reference)
        self._last_action = None

    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        self._last_action = np.asarray(action, dtype=float)
        raw_obs, _, terminated, truncated, info = self.env.step(action)
        levels = np.asarray(raw_obs[self.level_idx], dtype=float)
        temps = np.asarray(raw_obs[self.temp_idx], dtype=float)
        # use the 5D applied action for the feedforward penalty
        applied = self._last_action[:5] if len(self._last_action) >= 5 else self._last_action
        reward = regulation_reward(levels, temps, self.reference, applied, self._model_optimal)
        return raw_obs, reward, terminated, truncated, info


def tracking_steady_state_action(model, y_sp):
    """Invert the plant model: given a 6D target (3 levels + 3 temps), find the
    5-actuator command [pump, V-12, V-23, V-33, heater] in [0, 1].

    Steady-state assumptions: q_pump = q_12 = q_23 = q_3r = q (mass balance).
    Thermal: RHO_CP × q × (T_upstream − T_downstream) = UA × (T_downstream − T_amb).
    Pump curve: q = q_max × √((shutoff×u² − static)/(shutoff − static)).
    Valve flow: q = cv × u × √(level + gravity_drop).
    """
    levels = [float(v) for v in y_sp[:3]]
    temps = [float(v) for v in y_sp[3:]]
    rho_cp = model.rho * model.cp
    t_amb = model.t_ambient
    t_res = model.t_supply

    flow_candidates = []
    for i in (1, 2):
        upstream, downstream = temps[i - 1], temps[i]
        numerator = model.ua * (downstream - t_amb)
        denominator = rho_cp * (upstream - downstream)
        if abs(denominator) > 1e-12:
            q = numerator / denominator
            if math.isfinite(q) and q > 0.0:
                flow_candidates.append(q)
    q = float(np.mean(flow_candidates)) if flow_candidates else model.q_max * 0.2

    pump = math.sqrt(
        (model.pump_static_head
         + (model.pump_shutoff_head - model.pump_static_head) * (q / model.q_max) ** 2)
        / model.pump_shutoff_head
    )

    cvs = [model.c_v12, model.c_v23, model.c_v33]
    valves = []
    for i in range(3):
        denom = cvs[i] * math.sqrt(levels[i] + model.gravity_drop)
        valves.append(min(1.0, q / denom) if denom > 1e-12 else 0.0)

    liquid_heat = rho_cp * q * (temps[0] - t_res) + model.ua * (temps[0] - t_amb)
    heater = liquid_heat / model.q_heat_max

    action = [pump, valves[0], valves[1], valves[2], heater]
    return np.asarray([max(0.0, min(1.0, a)) for a in action], dtype=np.float32)


class ResidualEnvWrapper(gym.Wrapper):
    """Combined residual RL wrapper: 2D action → 5D physical, 17D normalized obs,
    regulation reward. The RL agent adjusts only V-33 (drain) + heater on top of
    a model feedforward + inline PID.

    Layout-agnostic — the wrapped env's obs/action order is passed in so the same
    wrapper runs on either track:
      numpy  obs grouped [h1,h2,h3,T1,T2,T3,...], action canonical:  level_idx=(0,1,2), temp_idx=(3,4,5), act_idx=(0,1,2,3,4)
      modbus obs interleaved [h1,T1,h2,T2,h3,T3,...], action [V12,V23,E101,V33,VFD]: level_idx=(0,2,4), temp_idx=(1,3,5), act_idx=(4,0,1,3,2)
    canonical physical action = [pump, V-12, V-23, V-33, heater]."""

    def __init__(self, env, model, reference, level_idx=(0, 2, 4), temp_idx=(1, 3, 5),
                 act_idx=(0, 1, 2, 3, 4)):
        super().__init__(env)
        self.model = model
        self.reference = np.asarray(reference, dtype=float)  # 6D: 3 levels + 3 temps
        self.level_idx = list(level_idx)
        self.temp_idx = list(temp_idx)
        # act_idx[k] = where canonical slot k lands in the wrapped env's action vector
        self.act_idx = list(act_idx)
        self._n_act = int(env.action_space.shape[0])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(17,), dtype=np.float32)
        self._prev_physical = np.zeros(5, dtype=np.float32)
        self._model_optimal = np.zeros(5, dtype=np.float32)
        self._last_raw_obs = None

    def _resolve_action(self, residual):
        """2D residual → canonical 5D physical [pump, V-12, V-23, V-33, heater]
        (feedforward + PID + residual on V-33 + heater). Caller maps to env order."""
        residual = np.clip(np.asarray(residual, dtype=float), -1.0, 1.0)
        # Layer 1: model feedforward
        baseline = tracking_steady_state_action(self.model, self.reference)
        self._model_optimal = baseline.copy()
        physical = baseline.copy()

        # Layer 2: inline PID (pump → Tank1 level, V-12 → Tank2 level)
        if self._last_raw_obs is not None and len(self._last_raw_obs) > max(self.level_idx):
            h1 = float(self._last_raw_obs[self.level_idx[0]])
            h2 = float(self._last_raw_obs[self.level_idx[1]])
            pump_corr = np.clip(
                _PUMP_KP * (self.reference[0] - h1) / _LEVEL_TOL, -_MAX_CORRECTION, _MAX_CORRECTION)
            v12_corr = np.clip(
                _V12_KP * (self.reference[1] - h2) / _LEVEL_TOL, -_MAX_CORRECTION, _MAX_CORRECTION)
            physical[0] += pump_corr
            physical[1] += v12_corr

        # Layer 3: RL residual (V-33 [canonical index 3] + heater [canonical index 4])
        for idx, val in zip((3, 4), residual):
            base = float(physical[idx])
            physical[idx] = base + val * (1.0 - base) if val >= 0 else base + val * base

        return np.clip(physical, 0.0, 1.0).astype(np.float32)

    def _to_env_action(self, physical):
        """canonical [pump,V12,V23,V33,heater] → wrapped env's action order."""
        env_action = np.zeros(self._n_act, dtype=np.float32)
        for canon, eidx in enumerate(self.act_idx):
            env_action[eidx] = physical[canon]
        return env_action

    def _make_obs(self, raw_obs):
        """Raw sensor obs → 17D normalized (output + reference + prev_action)."""
        self._last_raw_obs = np.asarray(raw_obs, dtype=float)
        levels = self._last_raw_obs[self.level_idx]
        temps = self._last_raw_obs[self.temp_idx]
        output = np.concatenate([levels, temps])
        norm_output = np.clip(output / _OUTPUT_SCALES, 0, 1)
        norm_ref = np.clip(self.reference / _OUTPUT_SCALES, 0, 1)
        norm_action = np.clip(self._prev_physical, 0, 1)
        return np.concatenate([norm_output, norm_ref, norm_action]).astype(np.float32)

    def _regulation_reward(self, raw_obs, physical_action):
        """Regulation reward: tracking + feedforward penalty (canonical action)."""
        levels = np.asarray(raw_obs[self.level_idx], dtype=float)
        temps = np.asarray(raw_obs[self.temp_idx], dtype=float)
        output = np.concatenate([levels, temps])
        tracking = float(np.mean(((output - self.reference) / _ERROR_SCALES) ** 2))
        feedforward = _FEEDFORWARD_WEIGHT * float(np.sum((physical_action - self._model_optimal) ** 2))
        return -(tracking + feedforward)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self._prev_physical = np.zeros(5, dtype=np.float32)
        self._model_optimal = tracking_steady_state_action(self.model, self.reference)
        return self._make_obs(raw_obs), info

    def step(self, action):
        physical = self._resolve_action(action)           # canonical [pump,V12,V23,V33,heater]
        self._prev_physical = physical
        env_action = self._to_env_action(physical)         # map to wrapped env's action order
        raw_obs, _, terminated, truncated, info = self.env.step(env_action)
        reward = self._regulation_reward(raw_obs, physical)
        return self._make_obs(raw_obs), reward, terminated, truncated, info


__all__ = ["tracking_steady_state_action", "regulation_reward",
           "RegulationRewardWrapper", "ResidualEnvWrapper"]
