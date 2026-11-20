"""Models that score how much a user should like a movie."""
import torch


class MovieRecommender(torch.nn.Module):
    """``score(u, m) = quality[m] + user_embedding(u) . movie_embedding(m)``.

    ``quality`` is one number per movie, the same for every user; the dot
    product of the two vectors is the user's taste for the movie. Train the
    quality first (an optimizer over ``self.quality`` only), then the
    vectors (an optimizer over the two embeddings only). The movie vectors
    start at zero, so while only the quality is trained the dot product is
    exactly 0; the user vectors start small and random, so the movie
    vectors still get a gradient once they are trained.

    ``forward`` follows the course batch protocol: it reads
    ``batch['data']['user_id']``, ``['positive_movie_id']`` and
    ``['negative_movie_id']`` and writes ``batch['signals']['positive_score']``
    and ``['negative_score']``.
    """

    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        self.quality = torch.nn.Parameter(torch.zeros(num_movies))
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = torch.nn.Embedding(num_movies, embedding_dim)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.zeros_(self.movie_embedding.weight)

    def score(self, user_id, movie_id):
        """Return one score per ``(user_id, movie_id)`` pair, shape ``(batch,)``."""
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)
        score = (user_vector * 0).sum(dim=-1) + self.quality[movie_id] * 0
        ## Return the quality of the movie plus the dot product of the two vectors
        ## YOUR CODE HERE
        # -- start --
        score = self.quality[movie_id] + (user_vector * movie_vector).sum(dim=-1)
        # -- end --
        return score

    def forward(self, batch):
        data = batch['data']
        batch['signals'] = {
            'positive_score': self.score(data['user_id'], data['positive_movie_id']),
            'negative_score': self.score(data['user_id'], data['negative_movie_id']),
        }
        return batch


@torch.no_grad()
def evaluate_pairs(model, dataset, loss, metric):
    """Order accuracy and the loss averaged per movie on a dataset of pairs.

    ``metric`` gives the share of pairs in the right order. For the loss,
    every movie gets the mean ``loss`` of the pairs it appears in (on
    either side), and these per-movie losses are averaged, so a movie with
    thousands of pairs counts as much as a movie with ten.

    Returns
    -------
    dict
        ``{'accuracy': float, 'loss_per_movie': float}``; both are 0 for a
        dataset without pairs.
    """
    if not dataset.samples:
        return {'accuracy': 0.0, 'loss_per_movie': 0.0}
    pairs = torch.tensor(dataset.samples, dtype=torch.long)
    batch = {'data': {'user_id': pairs[:, 0], 'positive_movie_id': pairs[:, 1],
                      'negative_movie_id': pairs[:, 2]}}
    model(batch)
    correct, total = metric(batch)
    movie_losses = []
    for movie_id in torch.unique(pairs[:, 1:]).tolist():
        in_pair = (pairs[:, 1] == movie_id) | (pairs[:, 2] == movie_id)
        movie_batch = {'signals': {name: value[in_pair]
                                   for name, value in batch['signals'].items()}}
        movie_losses.append(float(loss(movie_batch)))
    return {'accuracy': float(correct) / float(total),
            'loss_per_movie': sum(movie_losses) / len(movie_losses)}


@torch.no_grad()
def recommend(model, user_id, seen_movie_ids, k=5, use_quality=True):
    """Return up to ``k`` movies with the highest score that the user has not seen.

    With ``use_quality=False``, the movie quality is left out of the score,
    so only the taste part ranks the movies.
    """
    seen = set(seen_movie_ids)
    movie_ids = torch.tensor([movie_id for movie_id in range(len(model.quality))
                              if movie_id not in seen], dtype=torch.long)
    scores = model.score(torch.full_like(movie_ids, user_id), movie_ids)
    if not use_quality:
        scores = scores - model.quality[movie_ids]
    order = scores.argsort(descending=True)[:k]
    return movie_ids[order].tolist()


@torch.no_grad()
def most_similar(vectors, index, k=5, least=False):
    """Return the ``k`` rows of ``vectors`` closest to row ``index`` (cosine).

    With ``least=True``, return the ``k`` farthest rows instead.
    """
    normalized = torch.nn.functional.normalize(vectors, dim=-1)
    similarity = normalized @ normalized[index]
    similarity[index] = float('inf') if least else float('-inf')
    order = similarity.argsort(descending=not least)
    return order[:k].tolist()
