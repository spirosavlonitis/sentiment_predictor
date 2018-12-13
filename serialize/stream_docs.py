def stream_docs(path):
    "Yield a doc, label pair at a time."
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            doc, label = line[:-3], int(line[-2])
            yield doc, label