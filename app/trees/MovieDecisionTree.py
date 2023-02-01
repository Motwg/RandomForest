from dataclasses import dataclass
from random import choice
from typing import Iterator

from DecisionTree import DecisionTree, Predicate


@dataclass(slots=True)
class MovieDecisionTree(DecisionTree):
    def predicate_generator(self, node) -> Iterator[Predicate]:
        while True:
            point = self.mapper[choice(node.train_set)[0]]
            predicates = [
                ('Budget > {}'.format(bg := point.budget),
                 lambda x: x.budget > bg),
                ('Revenue > {}'.format(rv := point.revenue),
                 lambda x: x.revenue >= rv),
                ('Release after {} year'.format(year := int(point.release_date[:4])),
                 lambda x: int(x.release_date[:4]) > year),
                ('Any genre = ' + (genre := choice(point.genres)),
                 lambda x: genre in x.genres),
                ('Vote count > {}'.format(vc := point.vote_count),
                 lambda x: x.vote_count > vc),
                ('Vote avg > {}'.format(va := point.vote_average),
                 lambda x: x.vote_average > va),
                ('Popularity > {}'.format(pop := point.popularity),
                 lambda x: x.popularity > pop),
                ('Language = {}'.format(lang := point.original_language),
                 lambda x: x.original_language == lang),
                ('Any prod company = {}'.format(pc := choice(point.production_companies)),
                 lambda x: pc in x.production_companies),
                ('Collection = {}'.format(collection := point.belongs_to_collection),
                 lambda x: x.belongs_to_collection == collection),
            ]
            yield choice(predicates)
