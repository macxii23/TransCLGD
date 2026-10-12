import os, json, gzip #torch-free graph storage: graphs.jsonl(.gz) or {id}/graph.json

_JSONL_NAMES = ('graphs.jsonl.gz', 'graphs.jsonl') #packed release filenames (gz preferred)
_JSONL_PLAIN, _JSONL_IDX = 'graphs.jsonl', 'graphs.jsonl.idx' #lazy expand + byte-offset index
_GID_KEYS = ('id', 'gid') #accepted id field names in a jsonl record

def jsonl_path(graph_data_dir): #packed file under dir if any
    for name in _JSONL_NAMES:
        p = os.path.join(graph_data_dir, name)
        if os.path.isfile(p): return p
    return None

def _open_text(path, mode='rt'): #utf-8 text; gzip when .gz
    return gzip.open(path, mode, encoding='utf-8') if path.endswith('.gz') else open(path, mode, encoding='utf-8')

def _record_gid(g, fallback=None): #string id from record
    for k in _GID_KEYS:
        if k in g and g[k] is not None: return str(g[k])
    return None if fallback is None else str(fallback)

def _iter_dir_graphs(graph_data_dir): #yield from {id}/graph.json
    ids = sorted([d for d in os.listdir(graph_data_dir) if os.path.isfile(os.path.join(graph_data_dir, d, 'graph.json'))],
                 key=lambda x: int(x) if x.isdigit() else x)
    for gid in ids:
        with open(os.path.join(graph_data_dir, gid, 'graph.json'), 'r', encoding='utf-8') as f: yield gid, json.load(f)

def _iter_jsonl_graphs(path): #stream jsonl/jsonl.gz
    with _open_text(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            g = json.loads(line); yield _record_gid(g, i), g

def iter_graphs(graph_data_dir): #prefer jsonl(.gz), else folders
    jp = jsonl_path(graph_data_dir)
    return _iter_jsonl_graphs(jp) if jp else _iter_dir_graphs(graph_data_dir)

def list_graph_ids(graph_data_dir): #sorted ids
    return sorted((gid for gid, _ in iter_graphs(graph_data_dir)), key=lambda x: int(x) if str(x).isdigit() else str(x))

def ensure_jsonl_index(graph_data_dir): #expand gz once + build seek index; return (plain, index)
    jp = jsonl_path(graph_data_dir)
    if not jp: return None, None
    plain, idxp = (jp, os.path.join(graph_data_dir, _JSONL_IDX)) if not jp.endswith('.gz') else (os.path.join(graph_data_dir, _JSONL_PLAIN), os.path.join(graph_data_dir, _JSONL_IDX))
    if jp.endswith('.gz') and (not os.path.isfile(plain) or os.path.getmtime(plain) < os.path.getmtime(jp)):
        with _open_text(jp) as fin, open(plain, 'w', encoding='utf-8') as fout:
            for line in fin: fout.write(line if line.endswith('\n') else line + '\n')
    if os.path.isfile(idxp) and os.path.getmtime(idxp) >= os.path.getmtime(plain):
        with open(idxp, 'r', encoding='utf-8') as f: return plain, json.load(f)
    index = {}
    with open(plain, 'rb') as f:
        while True:
            off = f.tell(); line = f.readline()
            if not line: break
            g = json.loads(line); index[_record_gid(g, len(index))] = off
    with open(idxp, 'w', encoding='utf-8') as f: json.dump(index, f, ensure_ascii=False)
    return plain, index

def load_graph(graph_data_dir, gid): #same source priority as iter_graphs (jsonl > folder)
    gid = str(gid)
    if jsonl_path(graph_data_dir):
        plain, index = ensure_jsonl_index(graph_data_dir)
        if plain and index and gid in index:
            with open(plain, 'rb') as f:
                f.seek(index[gid]); return json.loads(f.readline())
        for _gid, g in iter_graphs(graph_data_dir):
            if str(_gid) == gid: return g
        raise FileNotFoundError(f'graph id={gid} not found in jsonl under {graph_data_dir}')
    p = os.path.join(graph_data_dir, gid, 'graph.json')
    if os.path.isfile(p):
        with open(p, 'r', encoding='utf-8') as f: return json.load(f)
    raise FileNotFoundError(f'graph id={gid} not found under {graph_data_dir}')
