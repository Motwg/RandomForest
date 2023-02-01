from collections import namedtuple
from itertools import groupby, accumulate

from app.data_utils import read_csv, convert_fields, write_csv
from app.trees.MovieDecisionTree import MovieDecisionTree


def main():
    movies = read_csv('movies.csv',
                      ['movie_id', 'tmdb_id', 'title', 'popularity', 'genres', 'overview', 'vote_average',
                       'vote_count', 'release_date', 'revenue', 'budget',
                       'belongs_to_collection', 'original_language', 'production_companies'])
    test = read_csv('task.csv', ['idx', 'user_id', 'movie_id'])
    train = read_csv('train.csv', ['idx', 'user_id', 'movie_id', 'rate'])

    movies, test, train = (list(convert_fields(
        csv,
        idx=int,
        user_id=int,
        movie_id=int,
        popularity=float,
        genres=eval,
        vote_average=float,
        vote_count=float,
        rate=eval,
        budget=int,
        revenue=int,
        production_companies=eval
    )) for csv in [movies, test, train])
    print(movies[:10])
    movie_class = namedtuple('Movie', movies[0].keys())
    movies = {movie['movie_id']: movie_class(**movie) for movie in movies}

    # group by user id
    user_train = {k: list(gr) for k, gr in groupby(train, key=lambda item: item['user_id'])}
    user_test = {k: list(gr) for k, gr in groupby(test, key=lambda item: item['user_id'])}
    model = MovieDecisionTree(movies)
    options = {
        'no_trees': 101,
        'k': 70,
        'k_div': 1.6,
        'max_depth': 3,
        'entropy_th': 0.2,
        'ig_th': 0.02,
        'no_validate': 10
    }

    for i, (user_id, tests) in enumerate(user_test.items()):
        print(i, user_id)
        train = user_train[user_id]

        model.fit(map(lambda tr: tr['movie_id'], train), map(lambda tr: tr['rate'], train), **options)
        for idx_test, pred in enumerate(model.predict(map(lambda t: t['movie_id'], tests))):
            tests[idx_test]['rate'] = pred
        # break

    # show results
    if options['no_validate'] > 0:
        print(model.stats_correct)
        for i, diff in enumerate(map(lambda x: x / sum(model.stats_correct), accumulate(model.stats_correct))):
            print(f'diff={i}:  {100 * diff}%')

    model.plot()
    write_csv(test, 'submission.csv', ['idx', 'user_id', 'movie_id', 'rate'])


if __name__ == '__main__':
    main()
