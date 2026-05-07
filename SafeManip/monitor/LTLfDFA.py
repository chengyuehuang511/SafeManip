import copy
import os
import shutil
import sys
import tempfile
import time
import inspect

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import pydot
import networkx as nx
import ltlf2dfa.ltlf2dfa as ltlf2dfa_runtime
from ltlf2dfa.parser.ltlf import LTLfParser
import matplotlib.pyplot as plt
import matplotlib.animation

from PIL import Image
from io import BytesIO

from typing import List, Dict, Tuple


def _from_pydot_compat(dfa_pydot):
    """
    Bridge the networkx / pydot API mismatch introduced by newer pydot versions.

    networkx expects `get_strict(None)` while pydot 4 exposes `get_strict(self)`.
    """
    get_strict = getattr(dfa_pydot, "get_strict", None)
    if get_strict is not None:
        try:
            param_count = len(inspect.signature(get_strict).parameters)
        except (TypeError, ValueError):
            param_count = None
        if param_count == 0:
            original = get_strict

            def _wrapped_get_strict(*args, **kwargs):
                return original()

            dfa_pydot.get_strict = _wrapped_get_strict
    return nx.nx_pydot.from_pydot(dfa_pydot)


def _ensure_ltlf2dfa_writable_runtime():
    """
    Redirect ltlf2dfa's generated MONA files into a writable temp directory.

    The upstream package writes `automa.mona` inside its installed package
    directory, which is not guaranteed to be writable in shared environments.
    """
    runtime_dir = tempfile.mkdtemp(prefix="ltlf2dfa_runtime_")
    ltlf2dfa_runtime.PACKAGE_DIR = runtime_dir
    return runtime_dir


def _ensure_mona_on_path():
    """
    Make sure the MONA binary is discoverable by ltlf2dfa.

    ltlf2dfa invokes `mona` via `shell=True`, so it depends on PATH rather than
    Python import state. In our RoboCasa setup MONA is often installed inside
    the active conda environment but not exported globally.
    """
    if shutil.which("mona"):
        return shutil.which("mona")

    candidate_dirs = []
    executable_dir = os.path.dirname(sys.executable)
    if executable_dir:
        candidate_dirs.append(executable_dir)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate_dirs.append(os.path.join(conda_prefix, "bin"))

    for directory in candidate_dirs:
        candidate = os.path.join(directory, "mona")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            current_path = os.environ.get("PATH") or ""
            os.environ["PATH"] = directory if not current_path else f"{directory}:{current_path}"
            return candidate
    return None


def parse_mtlf_to_ltlf(ltlf_formula, connector='&', add_eventually=True):
    while '$' in ltlf_formula:
        index = ltlf_formula.find('$')
        next_left_bracket = index + ltlf_formula[index:].find('[')
        next_right_bracket = index + ltlf_formula[index:].find(']')
        end = next_right_bracket
        duration = int(ltlf_formula[next_left_bracket+1:next_right_bracket])
        post_duration = ltlf_formula[next_right_bracket:]
        next_left_bracket = post_duration.find('[')
        next_right_bracket = post_duration[next_left_bracket:].find(']')
        end += next_right_bracket + next_left_bracket
        subpredicate = post_duration[next_left_bracket+1:next_right_bracket+1]
        subpredicate = f'({subpredicate})'
        if '$' in subpredicate:
            raise ValueError("Cannot nest $ operators")
        replacement = ''
        for i in range(duration):
            substr = subpredicate
            for _ in range(i):
                substr = f'X({substr})'
            replacement += substr
            if i != duration - 1:
                replacement += f' {connector} '
        if add_eventually:
            replacement = f'F({replacement})'
        else:
            replacement = f'({replacement})'
        ltlf_formula = ltlf_formula[:index] + \
            replacement + ltlf_formula[end+1:]
    return ltlf_formula


def ltlf_to_python(ltl_predicate):
    ltl_predicate = ltl_predicate.replace("&", " and ")
    ltl_predicate = ltl_predicate.replace("|", " or ")
    ltl_predicate = ltl_predicate.replace("~", " not ")
    ltl_predicate = ltl_predicate.replace("true", "True")
    return ltl_predicate


def get_pydot_image(dfa, cur_node=None, color=True, svg=False):
    graph_copy = copy.deepcopy(dfa)
    graph_copy.graph['size'] = (100, 100)
    for node in graph_copy.nodes:
        node_props = graph_copy.nodes[node]
        node_props['height'] = 2.5
        node_props['fontsize'] = 50
        node_props['fontname'] = 'times bold'
        if color:
            if node == 'init' or node == '\\n':
                node_color = 'invis'
            else:
                node_color = 'green' if graph_copy.nodes[node]['accepting'] else 'red'
            node_props['shape'] = 'circle'
            node_props['color'] = 'black' if cur_node == node else node_color
            node_props['penwidth'] = 20 if cur_node == node else 3
            node_props['fillcolor'] = node_color
            node_props['style'] = 'filled'
        else:
            node_props['shape'] = 'doublecircle' if graph_copy.nodes[node]['accepting'] else 'circle'
            if node == 'init':
                node_props['fillcolor'] = 'invis'
                node_props['style'] = 'invis'
            else:
                node_props['fillcolor'] = 'grey80' if node == cur_node else 'white'
                node_props['style'] = 'filled'
    if svg:
        return nx.nx_pydot.to_pydot(graph_copy).create_svg()
    sg_img = nx.nx_pydot.to_pydot(graph_copy).create_png()
    return Image.open(BytesIO(sg_img))


class LTLfDFA:
    ACCEPTING_PREFIX = " node [shape = doublecircle];"

    def __init__(self, ltlf_formula):
        ltlf_formula = parse_mtlf_to_ltlf(ltlf_formula, add_eventually=False)
        self._formula = ltlf_formula
        self._runtime_dir = _ensure_ltlf2dfa_writable_runtime()
        self._mona_path = _ensure_mona_on_path()
        parser = LTLfParser()
        formula = parser(self._formula)
        self.symbols = formula.find_labels()
        self._pydot_str = formula.to_dfa()
        dfa_pydot = pydot.graph_from_dot_data(self._pydot_str)[0]
        self._dfa = _from_pydot_compat(dfa_pydot)
        if '0.0' in self._dfa:
            raise ValueError(
                "ltlf2dfa returned a degenerate DFA. MONA may be missing or unreachable, "
                f"or DFA generation failed for this formula. mona_path={self._mona_path!r}"
            )
        self._init_state = next(iter(self._dfa.out_edges('init')))[-1]
        self._current_state = self._init_state
        lines = self._pydot_str.split("\n")
        for line in lines:
            if not line.startswith(LTLfDFA.ACCEPTING_PREFIX):
                continue
            substr = line[len(LTLfDFA.ACCEPTING_PREFIX):]
            accepting_nodes = substr.split(";")
            for node in accepting_nodes:
                node = node.strip()
                if len(node) > 0:
                    self._dfa.nodes[node.strip()]['accepting'] = True
        for node in self._dfa.nodes:
            if 'accepting' not in self._dfa.nodes[node]:
                self._dfa.nodes[node]['accepting'] = False
        for u, v, a in self._dfa.edges(data=True):
            if 'label' in a:
                a['label'] = a['label'][1:-1]
                a['orig_label'] = a['label']
                a['label'] = ltlf_to_python(a['label'])
                label_symbols = a['label']
                a['symbols'] = [symbol for symbol in self.symbols if symbol in label_symbols]
        self._trap_states = []
        for node in self._dfa.nodes:
            if all([src == dst for src, dst in self._dfa.out_edges(node)]):
                self._trap_states.append(node)
        if '\\n' in self._dfa:
            self._dfa.remove_node('\\n')
        self._terminal_reject_state = "__terminal_reject__"
        if self._terminal_reject_state not in self._dfa:
            self._dfa.add_node(self._terminal_reject_state, accepting=False)
            self._dfa.add_edge(self._terminal_reject_state, self._terminal_reject_state, label="True", orig_label="true", symbols=[])
        self.q0 = self._init_state
        self.F = {node for node in self._dfa.nodes if self.is_accepting(node)}
        self.rejecting_states = {
            node for node in self._trap_states
            if node in self._dfa.nodes and not self.is_accepting(node)
        }
        self.rejecting_states.add(self._terminal_reject_state)

    def step(self, data, return_state=False):
        self._current_state = self.delta(self._current_state, data)
        if return_state:
            return self.is_accepting(self._current_state), self._current_state
        return self.is_accepting(self._current_state)

    def delta(self, current_state, data_dict):
        return self._compute_next_state(current_state, data_dict)

    def terminal_delta(self, current_state):
        """
        Resolve the final DFA outcome at episode end.

        Pending non-accepting states collapse into an explicit rejecting trap so
        finite-trace failures can be inspected the same way as ordinary trap
        states.
        """
        if self.is_accepting(current_state):
            return current_state
        return self._terminal_reject_state

    def _compute_next_state(self, current_state, data_dict):
        valid_states = []
        for u, v, a in self._dfa.out_edges(current_state, data=True):
            if eval(a['label'], dict(data_dict)):
                valid_states.append(v)
        if len(valid_states) != 1:
            raise ValueError(
                f"Unable to find state transition from {u} with {data_dict}, aborting.")
        return valid_states[0]

    def get_init_state(self):
        return self._init_state

    def from_init(self, data: Dict[str, List[Tuple[int, bool]]], return_state=False):
        current_state = self._init_state
        if data is None:
            return [(self.is_accepting(self._init_state), self._init_state)]
        data_key = next(iter(data))
        ret_val = []
        time_steps = len(data[data_key])
        for i in range(time_steps):
            data_dict = {var: data[var][i][-1] for var in data}
            current_state = self._compute_next_state(current_state, data_dict)
            if return_state:
                ret_val.append((self.is_accepting(current_state), current_state))
            else:
                ret_val.append(self.is_accepting(current_state))
        return ret_val

    def is_accepting(self, state):
        return self._dfa.nodes[state]['accepting']

    def set_state(self, state):
        assert state in self._dfa.nodes, f"Given state not in DFA"
        self._current_state = state

    def save_image(self, file_name, cur_node=None, color=False):
        if file_name.endswith('svg'):
            sg_img = self.get_pydot_image(cur_node=cur_node, color=color, svg=True)
            with open(file_name, 'wb') as f:
                f.write(sg_img)
        else:
            sg_img = self.get_pydot_image(cur_node=cur_node, color=color)
            sg_img.save(file_name)

    def get_pydot_image(self, cur_node=None, color=True, svg=False):
        return get_pydot_image(self._dfa, cur_node, color, svg)

    def is_trap_state(self, state):
        return state in self._trap_states or state == self._terminal_reject_state

    def animate(self, steps, mp4_file, fps=20):
        fig, ax = plt.subplots()
        ax.axis('off')

        def handle_current_state(index):
            cur_node = steps[index]
            ax.clear()
            sg_img = self.get_pydot_image(cur_node, color=False)
            ax.imshow(sg_img)
            ax.set_title(f"Time: {index} State: {cur_node}, Holds: {self._dfa.nodes[cur_node]['accepting']}")
            ax.axis('off')

        ani = matplotlib.animation.FuncAnimation(fig, handle_current_state, frames=len(steps), repeat=False)
        writer = matplotlib.animation.FFMpegFileWriter(fps=fps, codec="mpeg4")
        ani.save(mp4_file, writer=writer)

    def reset(self):
        self._current_state = self._init_state


class DFAView:
    def __init__(self, ltlfdfa: LTLfDFA, current_state=None):
        self.ltlfdfa = ltlfdfa
        self.current_state = current_state if current_state is not None else ltlfdfa.get_init_state()

    def is_trap(self):
        return self.ltlfdfa.is_trap_state(self.current_state)

    def is_accepting(self):
        return self.ltlfdfa.is_accepting(self.current_state)
