import chromadb
import pandas as pd
import sys

print('python version:', sys.version)
print('chromadb version:', getattr(chromadb, '__version__', 'unknown'))

try:
    client = chromadb.PersistentClient(path="./chroma_store_GONECASE2")
    collection = client.get_collection("legal_judgments_collection")
except Exception as e:
    print('ERROR connecting/getting collection:', repr(e))
    raise

try:
    data = collection.get(include=['embeddings', 'documents', 'metadatas'])
except Exception as e:
    print('ERROR calling collection.get():', repr(e))
    raise

print('\nDATA KEYS:', list(data.keys()))
ids = data.get('ids', [])
docs = data.get('documents', [])
embs = data.get('embeddings', [])
metas = data.get('metadatas', [])
print('counts -> ids:', len(ids), 'documents:', len(docs), 'embeddings:', len(embs), 'metadatas:', len(metas))

if len(embs) > 0:
    e0 = embs[0]
    print('type(embedding[0]) ->', type(e0))
    try:
        print('len(embedding[0]) ->', len(e0))
    except Exception as ee:
        print('error getting len(embedding[0]):', repr(ee))
    try:
        from itertools import islice
        it = iter(e0) if hasattr(e0, '__iter__') else None
        sample = list(islice(it, 10)) if it is not None else e0
        print('embedding[0] sample (first 10):', sample)
    except Exception as ee:
        print('error sampling embedding[0]:', repr(ee))

# Build dataframe similar to viz.py and show preview
try:
    df = pd.DataFrame({
        'id': ids,
        'document': docs,
        'embedding': embs
    })
    meta_df = pd.DataFrame(metas)
    if not meta_df.empty:
        df = pd.concat([df, meta_df], axis=1)
    print('\nDataFrame preview:')
    print(df.head(10).to_string())
    print('\nAll embeddings list/tuple?:', df['embedding'].apply(lambda x: isinstance(x, (list, tuple))).all())
except Exception as e:
    print('ERROR building/printing DataFrame:', repr(e))
    raise

print('\nCompleted diagnostics')
