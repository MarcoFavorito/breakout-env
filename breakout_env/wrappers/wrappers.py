import copy
from gym import Wrapper
from pythogic.base.Symbol import Symbol
from pythogic.base.Alphabet import Alphabet
from pythogic.base.Formula import AtomicFormula, PathExpressionEventually, PathExpressionSequence, And, Not, \
    LogicalTrue, PathExpressionStar
from pythogic.base.utils import _to_pythomata_dfa
from pythogic.ldlf_empty_traces.LDLf_EmptyTraces import LDLf_EmptyTraces
import numpy as np
from pythomata.base.Simulator import Simulator
from pythomata.base.utils import Sink


class BreakoutRABUWrapper(Wrapper):
    """Env wrapper for bottom-up rows deletion"""
    def __init__(self, env):
        super().__init__(env)
        self.row_symbols = [Symbol(r) for r in ["r0", "r1", "r2"]]
        self.dfa = self._build_automata()
        self.goal_reward = 1000
        self.transition_reward = 100
        self.simulator = Simulator(self.dfa)
        self.last_status = None


    def reset(self):
        self.env.reset()
        self.simulator.reset()

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        if done:
            # when we lose a life
            return obs, reward, done, _

        # overwrite old reward
        # reward = 0

        f = self.state2propositional_formula()

        old_state = self.simulator.cur_state
        self.simulator.make_transition(f)
        new_state = self.simulator.cur_state
        if new_state==Sink():
            done = True
            reward = -1000
        elif new_state in self.dfa.accepting_states:
            reward = 1000
        elif old_state!=new_state:
            reward = self.transition_reward

        return obs, reward, done or self.env.unwrapped.state.terminal, _



    def state2propositional_formula(self):
        e = self.unwrapped
        matrix = e.state.bricks.bricks_status_matrix
        row_status = np.all(matrix==0.0, axis=1)
        result = set()
        for rs, sym in zip(row_status, reversed(self.row_symbols)):
            if rs:
                result.add(sym)

        return frozenset(result)


    def _build_automata(self):
        rows = self.row_symbols
        atoms = [AtomicFormula(r) for r in rows]
        alphabet = Alphabet(set(rows))
        ldlf = LDLf_EmptyTraces(alphabet)
        f = PathExpressionEventually(
            PathExpressionSequence.chain([
                PathExpressionStar(And.chain([Not(atoms[0]), Not(atoms[1]), Not(atoms[2])])),
                PathExpressionStar(And.chain([atoms[0], Not(atoms[1]), Not(atoms[2])])),
                # Not(atoms[3]), Not(atoms[4]), Not(atoms[5])]),
                PathExpressionStar(And.chain([atoms[0], atoms[1], Not(atoms[2])])),
                # Not(atoms[3]), Not(atoms[4]), Not(atoms[5])]),
                # And.chain([atoms[0],      atoms[1],      atoms[2]]),  # Not(atoms[3]), Not(atoms[4]), Not(atoms[5])]),
                # And.chain([atoms[0],     atoms[1],      atoms[2],      atoms[3],  Not(atoms[4]), Not(atoms[5])]),
                # And.chain([atoms[0],     atoms[1],      atoms[2],      atoms[3],      atoms[4],  Not(atoms[5])]),
                # And.chain([atoms[0],     atoms[1],      atoms[2],      atoms[3],      atoms[4],      atoms[5] ])
            ]),
            And.chain([atoms[0], atoms[1], atoms[2]])
        )
        nfa = ldlf.to_nfa(f)
        dfa = _to_pythomata_dfa(nfa)

        return dfa





