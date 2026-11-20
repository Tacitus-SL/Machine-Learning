# Word2Vec With Hierarchical Softmax

Word2Vec learns a vector for each word by asking a simple question again and again: given one word, which words tend to appear nearby?

In this task, we will implement a skip-gram Word2Vec model. The encoder is intentionally simple: one embedding matrix. The decoder is hierarchical softmax: for each target word, the model predicts a sequence of left/right decisions in a deterministic binary-index tree. Each decision is a sigmoid binary classifier attached to one internal tree node.

The result is still a fully connected model in spirit, but the output layer is much cheaper than a full vocabulary softmax when the vocabulary becomes large.
