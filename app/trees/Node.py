from collections import Counter
from dataclasses import field, dataclass
from itertools import count
from math import log
from statistics import median
from typing import Any

import networkx as nx


def cnt_entropy(c: Counter, base: int = 2) -> float:
    return sum(- (p_i := no_reps / sum(c.values())) * log(p_i, base) for no_reps in c.values() if no_reps != 0)


@dataclass(slots=True)
class Node:
    train_set: list = field(default_factory=list, repr=False)
    parent: Any = field(default=None, repr=False)
    depth: int = field(default=0, kw_only=True)

    identifier: int = field(init=False, default_factory=count().__next__)
    description: str = field(init=False, default='')
    predicate: callable = field(init=False, default=None, repr=False)
    left: Any = field(init=False, default=None, repr=False)
    right: Any = field(init=False, default=None, repr=False)

    entropy: float = field(init=False)
    info_gain: float = field(init=False, default=None)
    counter: Counter = field(init=False)

    def __post_init__(self):
        self.counter = Counter(map(lambda x: x[1], self.train_set))
        self.entropy = cnt_entropy(self.counter)
        # # average weighted in node
        # if self.parent is not None:
        #     self.entropy *= len(self.train_set) / len(self.parent.train_set)

    def __call__(self, entry, *args, **kwargs):
        if self.predicate is None:
            # median
            return int(median(self.counter.elements()))
            # average
            # return round(sum(self.counter.elements()) / self.counter.total())
            # most common
            # return self.counter.most_common(1)[0][0]
        elif self.predicate(entry):
            return self.left(entry)
        else:
            return self.right(entry)

    def set_predicate(self, description: str, predicate: callable, info_gain: float, left, right):
        self.description, self.predicate = description, predicate
        self.left, self.right = left, right
        self.info_gain = info_gain

    def plot(self, g: nx.Graph, labels: dict):
        g.add_node(self.identifier)
        if self.description != '':
            labels[self.identifier] = f'{self.description}\n' \
                                      f'{self.counter.most_common()}\n' \
                                      f'{self.entropy}'
        else:
            labels[self.identifier] = f'{self.counter.most_common(1)[0][0]}\n' \
                                      f'{self.counter.most_common()}\n' \
                                      f'{self.entropy}'
        if self.left is not None:
            self.left.plot(g, labels)
            g.add_edge(self.identifier, self.left.identifier)
        if self.right is not None:
            self.right.plot(g, labels)
            g.add_edge(self.identifier, self.right.identifier)
