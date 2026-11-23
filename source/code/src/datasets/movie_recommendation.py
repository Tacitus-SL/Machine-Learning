"""MovieLens ratings and the preference-pair dataset for ranking models."""
import csv
import random
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import torch

MOVIELENS_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
CACHE_DIR = Path.home() / '.cache' / 'midl' / 'ml-latest-small'
SMALL_GENRES = ('Action', 'Comedy', 'Drama')


def load_movielens(small=False, seed=0, max_users=610, max_movies=1000,
                   data_dir=None):
    """Load observed ratings; hold out about a fifth of every user's ratings.

    A full run reads MovieLens (downloading ``ml-latest-small`` to
    ``~/.cache/midl`` once, unless ``data_dir`` holds ``ratings.csv`` and
    ``movies.csv``): the first ``max_users`` users and their
    ``max_movies`` most-rated movies. A small run generates 12 users with
    three tastes and 24 movies of three genres instead, without a download.
    The held-out ratings belong to the same users and movies; they test
    whether the model orders ratings it was not trained on.

    Parameters
    ----------
    small : bool
        Generate the small offline sample instead of reading MovieLens.
    seed : int
        Seed of the small sample and of the held-out choice.
    max_users, max_movies : int
        Size limits of a full run.
    data_dir : str or Path, optional
        Folder with ``ratings.csv`` and ``movies.csv`` (e.g. MovieLens 32M).

    Returns
    -------
    dict
        ``'train_ratings'``, ``'heldout_ratings'``: lists of
        ``(user_id, movie_id, rating)`` with ids counted from 0 and ratings
        from 0.5 to 5 stars; ``'movies'``: a list whose item ``movie_id`` is
        ``{'title': ..., 'genres': ...}``; ``'num_users'``, ``'num_movies'``.
    """
    if small:
        ratings, movies, num_users = _small_sample(seed)
    else:
        ratings, movies, num_users = _read_movielens(
            Path(data_dir) if data_dir else _download_movielens(),
            max_users=max_users, max_movies=max_movies)
    train_ratings, heldout_ratings = _hold_out_per_user(ratings, seed)
    return {'train_ratings': train_ratings, 'heldout_ratings': heldout_ratings,
            'movies': movies, 'num_users': num_users, 'num_movies': len(movies)}


def _small_sample(seed):
    # Users like one genre and dislike the others, plus noise; each user
    # rates about two thirds of the movies.
    generator = torch.Generator().manual_seed(seed)
    num_users, num_movies = 12, 24
    user_taste = torch.arange(num_users) % len(SMALL_GENRES)
    movie_genre = torch.arange(num_movies) % len(SMALL_GENRES)
    match = (user_taste[:, None] == movie_genre[None, :]).float()
    noise = torch.randn(num_users, num_movies, generator=generator) * 0.5
    stars = (1.5 + 3.0 * match + noise).clamp(0.5, 5.0)
    stars = (stars * 2).round() / 2
    observed = torch.rand(num_users, num_movies, generator=generator) < 0.67
    ratings = [(user, movie, float(stars[user, movie]))
               for user in range(num_users) for movie in range(num_movies)
               if observed[user, movie]]
    movies = [{'title': f'Movie {movie} ({SMALL_GENRES[movie_genre[movie]]})',
               'genres': SMALL_GENRES[movie_genre[movie]]}
              for movie in range(num_movies)]
    return ratings, movies, num_users


def _download_movielens():
    if not (CACHE_DIR / 'ratings.csv').exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urlopen(MOVIELENS_URL, timeout=60) as response, \
                ZipFile(BytesIO(response.read())) as archive:
            for filename in ('ratings.csv', 'movies.csv'):
                (CACHE_DIR / filename).write_bytes(
                    archive.read('ml-latest-small/' + filename))
    return CACHE_DIR


def _read_movielens(folder, max_users, max_movies):
    for filename in ('ratings.csv', 'movies.csv'):
        if not (folder / filename).exists():
            raise FileNotFoundError(f'{folder} has no {filename}')

    rows, users = [], set()
    with (folder / 'ratings.csv').open(newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            user = int(row['userId'])
            if user not in users:
                if len(users) >= max_users:
                    break  # MovieLens files are ordered by userId.
                users.add(user)
            rows.append((user, int(row['movieId']), float(row['rating'])))

    popular = {movie for movie, _ in
               Counter(movie for _, movie, _ in rows).most_common(max_movies)}
    rows = [row for row in rows if row[1] in popular]
    user_index = {user: i for i, user in enumerate(sorted({r[0] for r in rows}))}
    movie_index = {movie: i for i, movie in enumerate(sorted(popular))}
    ratings = [(user_index[u], movie_index[m], rating) for u, m, rating in rows]

    movies = [None] * len(movie_index)
    with (folder / 'movies.csv').open(newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            movie = int(row['movieId'])
            if movie in movie_index:
                movies[movie_index[movie]] = {'title': row['title'],
                                              'genres': row['genres']}
    return ratings, movies, len(user_index)


def _hold_out_per_user(ratings, seed):
    by_user = defaultdict(list)
    for row in ratings:
        by_user[row[0]].append(row)
    rng = random.Random(seed)
    train, heldout = [], []
    for user in sorted(by_user):
        rows = by_user[user]
        rng.shuffle(rows)
        heldout_count = max(2, len(rows) // 5) if len(rows) >= 5 else 0
        heldout.extend(rows[:heldout_count])
        train.extend(rows[heldout_count:])
    return train, heldout


def make_heldout_dataset(train_ratings, heldout_ratings, margin=0.75):
    """Pairs of one user's ratings in which at least one rating is held out.

    Built with ``PairwiseMovieDataset`` from all ratings, without a cap, so
    every held-out rating is compared with every other rating of its user.
    """
    heldout = {(user_id, movie_id) for user_id, movie_id, _ in heldout_ratings}
    dataset = PairwiseMovieDataset(train_ratings + heldout_ratings, margin=margin)
    dataset.samples = [
        (user_id, positive_id, negative_id)
        for user_id, positive_id, negative_id in dataset.samples
        if (user_id, positive_id) in heldout or (user_id, negative_id) in heldout
    ]
    return dataset


def group_by_user(rating_rows):
    """Return ``{user_id: [(movie_id, rating), ...]}`` from rating triples."""
    by_user = defaultdict(list)
    for user_id, movie_id, rating in rating_rows:
        by_user[int(user_id)].append((int(movie_id), float(rating)))
    return dict(by_user)


class PairwiseMovieDataset(torch.utils.data.Dataset):
    """Preference pairs from the ratings of one user at a time.

    A pair ``(user_id, positive_movie_id, negative_movie_id)`` means: this
    user rated the positive movie at least ``margin`` stars higher than the
    negative one. Only observed ratings are compared: a movie a user did
    not rate is never treated as disliked. At most ``max_pairs_per_user``
    pairs, chosen at random with ``seed``, are kept per user.
    """

    def __init__(self, rating_rows, margin=0.75, max_pairs_per_user=None, seed=0):
        ratings_by_user = group_by_user(rating_rows)
        rng = random.Random(seed)
        self.samples = []
        for user_id, user_ratings in sorted(ratings_by_user.items()):
            user_pairs = []
            ## Append (user_id, positive_movie_id, negative_movie_id) to user_pairs
            ## for every two movies of this user whose ratings differ by at
            ## least margin
            for pos_movie, pos_rating in user_ratings:
                for neg_movie, neg_rating in user_ratings:
                    if pos_rating >= neg_rating + margin:
                        user_pairs.append((user_id, pos_movie, neg_movie))
            if max_pairs_per_user is not None and len(user_pairs) > max_pairs_per_user:
                user_pairs = rng.sample(user_pairs, max_pairs_per_user)
            self.samples.extend(user_pairs)

    def __len__(self):
        ## Return the number of pairs
        res = len(self.samples)
        return res

    def __getitem__(self, index):
        ## Return the pair with this index as a dict of torch.long tensors
        user_id, positive_movie_id, negative_movie_id = self.samples[index]
        sample = {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'positive_movie_id': torch.tensor(positive_movie_id, dtype=torch.long),
            'negative_movie_id': torch.tensor(negative_movie_id, dtype=torch.long),
        }
        return sample
