#!/usr/bin/env python3
"""Exact toy example with an environment latent (Section 7 of the paper).

Environment e = (rho, nu), two uniform bits drawn once per environment.
  rho : viability-relevant rule bit.  The correct action is a* = z XOR rho.
  nu  : viability-irrelevant bias bit. The nuisance observation n ~ Bernoulli(p_nu),
        p_0 = 0.2, p_1 = 0.8.  nu is real, learnable environmental structure; it just
        does not matter for viability.
Each episode: z ~ Bernoulli(1/2), n ~ Bernoulli(p_nu); the agent observes (z, n), acts,
and receives r = 1{a = a*}.  Every agent starts from the same fixed memory state M_pre and
trains for K episodes, so  Delta C_u = I(M_post; rho | M_pre) = I(M_post; rho).

Five agents (all deterministic functions of the K episodes):
  rule learner        M = rho-hat                         (1 bit)
  rule+bias learner   M = (rho-hat, nu-hat)               (2 bits; learns everything about e)
  lookup table        M = table (z,n) -> a for seen pairs (unseen entries default to 0)
  noise memoriser     M = majority vote of n              (1 bit about nu, nothing about rho)
  reflex              no memory; hard-wired a = z         (correct in exactly the rho = 0 worlds)

Memory is the agent's canonical summary state: for the rule+bias learner the pair (rho-hat, nu-hat) and for the
noise memoriser the majority bit, not the raw counts from which they are computed; for the lookup table it is the
table itself including 'unseen' markers.  H(M) is the entropy of that summary.

Quantities (all exact, by enumerating the 4^K episode sequences x 4 environments):
  H(M)                       stored bits (rate)
  I(M; rho, nu)              structure about the environment (Takahashi-Hayashi epiplexity)
  C_u = I(M; rho)            viability-relevant structure (this paper)
  V   = E[acc | M used] - E[acc | M scrambled]        value form (Kolchinsky-Wolpert intervention)
  A   = E_e[acc]  and  P_e[acc >= theta]                adaptive reach, expectation and threshold form
  W_floor = k_B T ln2 * H(M)                            closed-cycle Landauer floor for resetting M
  eta*_floor = C_u / H(M)                               acquisition efficiency at the floor (<= 1)
"""
from __future__ import annotations

import itertools
import json
import numpy as np

K = 8
P_N = {0: 0.2, 1: 0.8}
k_B, T = 1.380649e-23, 300.0
LANDAUER = k_B * T * np.log(2)


def H(p):
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def MI(joint):
    """I(X;Y) from a 2-D joint table."""
    joint = np.asarray(joint, dtype=float)
    return H(joint.sum(1)) + H(joint.sum(0)) - H(joint)


# ----------------------------------------------------------------------------- agents
# Each agent: update(M, z, n, r) -> M ; act(M, z, n) -> a ; M0 = initial state.

class RuleLearner:
    name = "Rule learner"
    M0 = None                                   # unknown rho
    def update(self, M, z, n, a, r):            # one rewarded episode pins rho
        return self._infer(M, z, a, r)
    @staticmethod
    def _infer(M, z, a, r):
        if M is not None:
            return M
        a_star = a ^ (1 - r)                     # binary reward reveals the correct action
        return a_star ^ z                        # and a* = z XOR rho
    def act(self, M, z, n):
        return z if M is None else z ^ M

class RuleBiasLearner(RuleLearner):
    name = "Rule + bias learner"
    M0 = (None, 0, 0)                            # (rho-hat, count n=1, count episodes)
    def update(self, M, z, n, a, r):
        rho, c1, c = M
        rho = RuleLearner._infer(rho, z, a, r)
        return (rho, c1 + n, c + 1)
    def act(self, M, z, n):
        rho = M[0]
        return z if rho is None else z ^ rho
    @staticmethod
    def canonical(M):                            # nu-hat = majority of n
        rho, c1, c = M
        return (rho, int(2 * c1 > c))

class LookupTable:
    name = "Lookup table"
    M0 = (None, None, None, None)                # entry for (z,n) -> a*, None = unseen
    def update(self, M, z, n, a, r):
        M = list(M)
        M[2 * z + n] = a ^ (1 - r)               # store the correct action for this (z, n)
        return tuple(M)
    def act(self, M, z, n):
        v = M[2 * z + n]
        return 0 if v is None else v             # unseen entry: fixed default guess

class NoiseMemoriser:
    name = "Noise memoriser"
    M0 = (0, 0)                                  # (count n=1, count episodes)
    def update(self, M, z, n, a, r):
        return (M[0] + n, M[1] + 1)
    def act(self, M, z, n):                      # acts on the majority nuisance value
        return M if isinstance(M, int) else int(2 * M[0] > M[1])
    @staticmethod
    def canonical(M):
        return int(2 * M[0] > M[1])

class KeyedLearner(RuleLearner):
    """Stores rho XOR k together with a key k = the cue z of the first episode (exactly fair and independent of
    rho and nu), but its policy reads the second bit as if it were rho and ignores the key.  The memory carries one
    full Shannon bit about rho (the pair determines rho), yet the agent's own readout can extract none of it:
    C_u in the Shannon sense is 1, C_u for the agent's readout is exactly 0, and V is exactly 0."""
    name = "Keyed learner"
    M0 = (None, None)                             # (rho-hat, key)
    def update(self, M, z, n, a, r):
        rho, k = M
        if k is None:
            k = z
        return (RuleLearner._infer(rho, z, a, r), k)
    def act(self, M, z, n):
        first, second = M
        if first is None:                         # untrained: (None, key or None)
            return z
        return z ^ second                         # reads the stored bit, ignores the first slot
    @staticmethod
    def canonical(M):
        rho, k = M
        return (k, rho ^ k)                       # what is physically stored: (key, rho XOR key); act reads slot 2


class Reflex:
    name = "Reflex (a = z)"
    M0 = 0
    def update(self, M, z, n, a, r):
        return 0
    def act(self, M, z, n):
        return z


def train(agent, rho, nu, episodes):
    M = agent.M0
    for z, n in episodes:
        # during training the agent acts with its current policy; reward computed from the truth
        a = agent.act(M, z, n)
        r = int(a == (z ^ rho))
        M = agent.update(M, z, n, a, r)
    return M


def test_accuracy(agent, M, rho, nu):
    """Exact accuracy of policy act(M, .) in environment (rho, nu)."""
    acc = 0.0
    for z in (0, 1):
        for n in (0, 1):
            p = 0.5 * (P_N[nu] if n == 1 else 1 - P_N[nu])
            acc += p * (agent.act(M, z, n) == (z ^ rho))
    return acc


def analyse(agent, theta=0.8):
    canon = getattr(agent, "canonical", lambda M: M)
    joint: dict[tuple, float] = {}                 # (M, rho, nu) -> prob
    acc: dict[tuple, float] = {}                   # (M, rho, nu) -> accuracy
    for rho in (0, 1):
        for nu in (0, 1):
            for seq in itertools.product([(z, n) for z in (0, 1) for n in (0, 1)], repeat=K):
                p = 0.25
                for z, n in seq:
                    p *= 0.5 * (P_N[nu] if n == 1 else 1 - P_N[nu])
                M = canon(train(agent, rho, nu, seq))
                key = (M, rho, nu)
                joint[key] = joint.get(key, 0.0) + p
                if key not in acc:
                    acc[key] = test_accuracy(agent, M, rho, nu)
    Ms = sorted({k[0] for k in joint}, key=repr)
    idx = {m: i for i, m in enumerate(Ms)}
    J = np.zeros((len(Ms), 2, 2))
    for (M, rho, nu), p in joint.items():
        J[idx[M], rho, nu] += p
    P_M = J.sum((1, 2))
    HM = H(P_M)
    I_M_rho = MI(J.sum(2))
    I_M_nu = MI(J.sum(1))
    I_M_e = MI(J.reshape(len(Ms), 4))
    # usable bits for the agent's OWN readout: the rule estimate implied by its action, rho_eff = a XOR z,
    # and its mutual information with rho over (M, e, z, n).  By data processing this is <= I(M; rho).
    J_eff = np.zeros((2, 2))
    for (M, rho, nu), p in joint.items():
        for z in (0, 1):
            for n in (0, 1):
                pzn = 0.5 * (P_N[nu] if n == 1 else 1 - P_N[nu])
                J_eff[agent.act(M, z, n) ^ z, rho] += p * pzn
    C_u_readout = MI(J_eff)
    # value: expected accuracy with M as learned vs M scrambled (drawn from its marginal, independent of e)
    E_used = sum(p * acc[(M, rho, nu)] for (M, rho, nu), p in joint.items())
    E_scr = 0.0
    for M in Ms:
        for rho in (0, 1):
            for nu in (0, 1):
                E_scr += P_M[idx[M]] * 0.25 * test_accuracy(agent, M, rho, nu)
    V = E_used - E_scr
    # per-environment value V_e and the per-environment divergence bound |V_e| <= sqrt(D(P_{M|e} || P_M) / 2) (nats)
    per_env_V, per_env_bound = {}, {}
    for rho in (0, 1):
        for nu in (0, 1):
            pe = sum(p for (M, r_, n_), p in joint.items() if (r_, n_) == (rho, nu))
            used = sum(p / pe * acc[(M, rho, nu)] for (M, r_, n_), p in joint.items() if (r_, n_) == (rho, nu))
            scr = sum(P_M[idx[M]] * test_accuracy(agent, M, rho, nu) for M in Ms)
            D = sum((p / pe) * np.log((p / pe) / P_M[idx[M]]) for (M, r_, n_), p in joint.items() if (r_, n_) == (rho, nu) and p > 0)
            per_env_V[f"rho={rho},nu={nu}"] = round(float(used - scr), 4)
            per_env_bound[f"rho={rho},nu={nu}"] = round(float(np.sqrt(max(D, 0.0) / 2)), 4)
    # adaptive reach: per-environment expected accuracy after training
    per_env = {}
    for (M, rho, nu), p in joint.items():
        per_env[(rho, nu)] = per_env.get((rho, nu), 0.0) + 4 * p * acc[(M, rho, nu)]
    A_mean = float(np.mean(list(per_env.values())))
    A_thr = float(np.mean([v >= theta for v in per_env.values()]))
    eta_floor = I_M_rho / HM if HM > 1e-12 else float("nan")
    return dict(agent=agent.name, H_M=HM, I_M_env=I_M_e, C_u=I_M_rho, C_u_readout=C_u_readout, I_M_nu=I_M_nu, V=V,
                A_mean=A_mean, A_thr=A_thr, W_floor_kTln2=HM, eta_floor=eta_floor, n_states=len(Ms),
                per_env_V=per_env_V, per_env_bound=per_env_bound,
                per_env_acc={f"rho={r},nu={n}": round(v, 4) for (r, n), v in sorted(per_env.items())})


if __name__ == "__main__":
    agents = [RuleLearner(), RuleBiasLearner(), LookupTable(), NoiseMemoriser(), KeyedLearner(), Reflex()]
    rows = [analyse(a) for a in agents]
    print(f"K = {K} training episodes, p(n=1 | nu) = {P_N}, 1 k_B T ln2 = {LANDAUER*1e21:.2f} zJ at {T:.0f} K\n")
    hdr = f"{'agent':<22}{'H(M)':>7}{'I(M;e)':>8}{'C_u':>7}{'C_u^V':>7}{'I(M;nu)':>9}{'V':>8}{'A':>7}{'A>=.8':>7}{'eta*':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        eta = "  --" if np.isnan(r["eta_floor"]) else f"{r['eta_floor']:.3f}"
        print(f"{r['agent']:<22}{r['H_M']:>7.3f}{r['I_M_env']:>8.3f}{r['C_u']:>7.3f}{r['C_u_readout']:>7.3f}{r['I_M_nu']:>9.3f}"
              f"{r['V']:>8.3f}{r['A_mean']:>7.3f}{r['A_thr']:>7.2f}{eta:>7}")
    print("\nper-environment accuracy after training:")
    for r in rows:
        print(f"  {r['agent']:<22} {r['per_env_acc']}")
    print("\nper-environment value V_e and divergence bound sqrt(D(P_M|e || P_M)/2):")
    for r in rows:
        ok = all(abs(r["per_env_V"][k]) <= r["per_env_bound"][k] + 1e-9 for k in r["per_env_V"])
        print(f"  {r['agent']:<22} V_e={r['per_env_V']}  bound={r['per_env_bound']}  holds={ok}")
    with open("toy_environment_latent.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\nsaved toy_environment_latent.json")
