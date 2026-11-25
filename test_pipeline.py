"""Regression checks for recall, feature consistency and official data layouts."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from gensim.models import KeyedVectors

from project_paths import find_data_file
from run_pipeline import Recaller, build_associations


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.model = KeyedVectors(vector_size=8)
        self.model.add_vectors([str(i) for i in range(90)],
                               np.random.default_rng(7).normal(size=(90, 8)).astype(np.float32))
        self.model.fill_norms()
        self.histories = {'u1': ['0', '1', '2'], 'u2': ['1', '3']}
        associations, counts = build_associations(self.histories)
        self.recall = Recaller([self.model], associations, counts, 60, 10)

    def test_vector_batch_matches_gensim(self):
        self.recall.prepare_neighbors(self.histories.values())
        expected = self.model.most_similar('1', topn=10)
        actual = self.recall.nearest(0, '1')
        self.assertEqual([item for item, _ in actual], [item for item, _ in expected])
        np.testing.assert_allclose([score for _, score in actual], [score for _, score in expected], atol=1e-6)

    def test_unique_candidates_exclude_seen_and_features_are_consistent(self):
        history = self.histories['u1']
        candidates, channels = self.recall.recall_for(history)
        self.assertEqual(len(set(candidates)), 60)
        self.assertFalse(set(candidates) & set(history))
        features = self.recall.features(history, candidates, channels)
        self.assertTrue(np.isfinite(features).all())
        # Features for sampled negatives must equal the same rows at prediction.
        selected = candidates[::4]
        np.testing.assert_allclose(self.recall.features(history, selected, channels), features[::4], atol=1e-6)
        for position in range(3):
            offset = len(channels) + 2 + position * 3
            item = history[-1-position]
            expected = [self.model.similarity(item, candidate) for candidate in candidates]
            np.testing.assert_allclose(features[:, offset], expected, atol=1e-6)

    def test_cold_user_receives_unique_top50(self):
        candidates, channels = self.recall.recall_for([])
        self.assertGreaterEqual(len(set(candidates)), 50)
        self.assertTrue(np.isfinite(self.recall.features([], candidates, channels)).all())

    def test_candidates_respect_official_catalog(self):
        allowed = set(str(i) for i in range(20, 90))
        recall = Recaller([self.model], self.recall.associations, self.recall.counts, 60, 10,
                          allowed_items=allowed)
        candidates, _ = recall.recall_for(['0', '1'])
        self.assertEqual(len(set(candidates)), 60)
        self.assertTrue(set(candidates) <= allowed)

    def test_flat_and_official_nested_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'underexpose_test' / 'underexpose_test_click-0'
            folder.mkdir(parents=True)
            nested = folder / 'underexpose_test_qtime-0.csv'
            compressed = nested.with_suffix('.csv.gz')
            compressed.touch()
            self.assertEqual(find_data_file(root, 'test', nested.name), compressed)
            nested.touch()
            self.assertEqual(find_data_file(root, 'test', nested.name), nested)
            direct = root / 'underexpose_test' / nested.name
            compressed_direct = direct.with_suffix('.csv.gz')
            compressed_direct.touch()
            self.assertEqual(find_data_file(root, 'test', nested.name), compressed_direct)
            direct.touch()
            self.assertEqual(find_data_file(root, 'test', nested.name), direct)


if __name__ == '__main__':
    unittest.main()
