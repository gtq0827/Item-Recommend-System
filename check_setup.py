"""Check dependencies and inputs without starting full training."""
import argparse
import importlib
import sys
from project_paths import DATA_DIR, check_required_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-only', action='store_true', help='Skip dataset/cache checks')
    parser.add_argument('--smoke', action='store_true', help='Test graph vectors and a tiny ranking model')
    parser.add_argument('--stage', choices=('solution', 'graph', 'rank'), default='solution')
    args = parser.parse_args()
    print('Python:', sys.version.split()[0])
    print('Interpreter:', sys.executable)
    for name in ('tensorflow', 'numpy', 'pandas', 'scipy', 'sklearn', 'gensim', 'networkx', 'joblib', 'tqdm', 'psutil', 'h5py'):
        module = importlib.import_module(name)
        print(f'{name}: {module.__version__}')
    if args.smoke:
        import numpy as np
        from gensim.models import KeyedVectors
        from final_rank import DenseFeat, Wide_Deep, DNN
        import tensorflow as tf
        import tempfile
        from pathlib import Path
        import build_graph_model
        import recall_function
        for name in ('deepwalk_offline.bin', 'node2vec_offline.bin'):
            vectors = KeyedVectors.load_word2vec_format(str(DATA_DIR / name), binary=True)
            vectors.most_similar(vectors.index_to_key[0], topn=3)
            print(f'{name}: {len(vectors)} vectors, OK')
        import networkx as nx
        graph = nx.Graph()
        graph.add_edge('a', 'b', weight=1.0)
        graph.add_edge('b', 'c', weight=1.0)
        for kind in (build_graph_model.DeepWalk, build_graph_model.Node2Vec):
            embedding = kind(graph, walk_length=3, num_walks=2, workers=1)
            embedding.train(embed_size=4, window_size=2, workers=1, iters=1)
            assert embedding.w2v_model.wv.vector_size == 4
            print(f'{kind.__name__} tiny training: OK')
        columns = [DenseFeat('a'), DenseFeat('b')]
        model = Wide_Deep(columns, columns, dnn_hidden_units=(8, 4))
        model.compile('adam', 'binary_crossentropy')
        inputs = {'a': np.array([0., 1., 0., 1.]), 'b': np.array([1., 0., 1., 0.])}
        model.train_on_batch(inputs, np.array([0., 1., 0., 1.]))
        expected = model(inputs, training=False).numpy()
        assert np.isfinite(expected).all()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'smoke.h5')
            model.save(path)
            restored = tf.keras.models.load_model(path, custom_objects={'DNN': DNN})
            np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected)
        print('Ranking model training / prediction / H5 round-trip: OK')
    if not args.environment_only:
        check_required_files(args.stage)
    print('检查通过。')


if __name__ == '__main__':
    main()
