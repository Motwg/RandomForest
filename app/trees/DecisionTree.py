from abc import abstractmethod
from collections import Counter
from dataclasses import field, dataclass
from random import getrandbits
from typing import Iterator, Iterable, Any, Generic, TypeVar

import networkx as nx
from matplotlib import pyplot as plt

from Node import Node
from layout import hierarchy_pos

T = TypeVar('T')
Predicate = tuple[str, callable]


def join_predicates(p1: Predicate, p2: Predicate, mode: str) -> Predicate:
    if mode == 'or':
        return p1[0] + ' or ' + p2[0], p1[1] or p2[1]
    elif mode == 'and':
        return p1[0] + ' and ' + p2[0], p1[1] and p2[1]
    else:
        raise KeyError(f'Mode {mode} for joining predicates is invalid')


@dataclass(slots=True)
class DecisionTree:
    mapper: dict[int, Generic[T]]

    trees: list[Node] = field(init=False, default_factory=list)

    stats_n: int = field(init=False, default=0)
    stats_correct: list = field(init=False, default_factory=lambda: [0, 0, 0, 0, 0, 0])

    @abstractmethod
    def predicate_generator(self, node) -> Iterator[Predicate]:
        pass

    def complex_predicate_generator(self, node, *, k) -> Iterator[Predicate]:
        predicates = iter(self.predicate_generator(node))
        for _ in range(k):
            if (rand := getrandbits(2)) > 1:  # 50%
                yield next(predicates)
            elif rand == 1:
                yield join_predicates(next(predicates), next(predicates), 'and')  # 25%
            else:
                yield join_predicates(next(predicates), next(predicates), 'or')  # 25%

    def _generate_children(self, node, *, k=10):
        candidates = []
        for desc, predicate in self.complex_predicate_generator(node, k=k):
            lefts, rights = [], []
            for train_id, target in node.train_set:
                if predicate(self.mapper[train_id]):
                    lefts.append((train_id, target))
                else:
                    rights.append((train_id, target))
            left = Node(lefts, node, depth=node.depth + 1)
            right = Node(rights, node, depth=node.depth + 1)
            info_gain = node.entropy \
                - left.entropy * len(left.train_set) / len(node.train_set) \
                - right.entropy * len(right.train_set) / len(node.train_set)

            candidates.append((desc, predicate, info_gain, left, right))
        # if node.depth < 3:
        #     print(f'D: {node.depth} - {list(map(lambda c: (c[0], c[2]), sorted(candidates, key=lambda c: c[2], reverse=True)))}')
        return candidates

    def _select_children(self, node, *, k=10, max_depth=3, entropy_th=0.1, ig_th=0.1, k_div=None):
        depth = node.depth + 1
        if k_div:
            k = round(k / k_div)
        if depth > max_depth:
            # print(f'Max depth reached, that is something new')
            return
        if node.entropy < entropy_th:
            # print(f'Node entropy skip {entropy_th} > {node}')
            return
        candidates = self._generate_children(node, k=k)
        best = max(candidates, key=lambda candidate: candidate[2])
        if best[2] < ig_th:
            # print(f'Your features are one big joke: Best IG = {best[2]} for {node}')
            return
        if len(best[3].train_set) == 0 or len(best[4].train_set) == 0:
            return
        node.set_predicate(*best)
        self._select_children(node.left, k=k, max_depth=max_depth, entropy_th=entropy_th, ig_th=ig_th, k_div=k_div)
        self._select_children(node.right, k=k, max_depth=max_depth, entropy_th=entropy_th, ig_th=ig_th, k_div=k_div)

    def _generate_trees(self, train_set, no_trees: int = 1, **kwargs):
        self.trees.clear()
        for _ in range(no_trees):
            tree = Node(train_set)
            self._select_children(tree, **kwargs)
            self.trees.append(tree)

    def plot(self):
        g, labels = nx.DiGraph(), {}
        tree = self.trees[0]
        tree.plot(g, labels)
        color_map = ['red' if node == tree.identifier else 'tab:green' for node in g]
        pos = hierarchy_pos(g, tree.identifier)

        nx.draw(g, pos, labels=labels, with_labels=True, node_color=color_map, arrowstyle='-|>', node_size=250)
        plt.show()

    def fit(self, train_ids: Iterator[int], targets: Iterator[Any], no_validate: int = 15, **kwargs) -> None:
        val_ids = [next(train_ids) for _ in range(no_validate)]
        val_targets = [next(targets) for _ in range(no_validate)]

        self._generate_trees([(train_id, target) for train_id, target in zip(train_ids, targets)], **kwargs)
        # validate
        for val_got, val_target in zip(self.predict(val_ids), val_targets):
            self.stats_n += 1
            self.stats_correct[abs(val_got - val_target)] += 1

    def predict(self, test_ids: Iterable[int]) -> Iterable[Any]:
        for test_id in test_ids:
            votes = [tree(self.mapper[test_id]) for tree in self.trees]
            yield Counter(votes).most_common(1)[0][0]
            # yield int(median(votes))
            # yield round(sum(votes) / len(votes))
