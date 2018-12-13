def get_minibatch(doc_stream, size):
    """Return a mini batch of doc, label pairs."""
    docs, labels = [], []
    try:
        for _ in range(size):
            doc, label = next(doc_stream)
            docs.append(doc)
            labels.append(label)
    except StopIteration:
        return None, None
    return docs, labels