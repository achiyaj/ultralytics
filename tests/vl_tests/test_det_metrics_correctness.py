# import numpy as np
# import pytest
from ultralytics.utils.metrics import calc_total_det_stats


# @pytest.fixture
# def test_data():
#     return [
#         # Test case 1: Perfect detection
#         (np.array([True, True]),
#          np.array([0.9, 0.8]),
#          np.array([0, 1]),
#          np.array([0, 1]),
#          1.0, 1.0, 1.0),

#         # Test case 2: Some false positives and false negatives
#         (np.array([True, False, True, False]),
#          np.array([0.7, 0.6, 0.9, 0.8]),
#          np.array([0, 1, 0, 2]),
#          np.array([0, -1, 0, 2]),
#          2 / 3, 2 / 3, 0.7777777777777778),

#         # Test case 3: No detections
#         (np.array([]),
#          np.array([]),
#          np.array([]),
#          np.array([]),
#          0.0, 0.0, 0.0),
#     ]

# def test_precision_recall_ap(test_data):
#     for tp, conf, pred_cls, target_cls, exp_precision, exp_recall, exp_ap in test_data:
#         precision, recall, ap = calc_total_det_stats(tp, conf, pred_cls, target_cls)
#         assert precision == pytest.approx(exp_precision)
#         assert recall == pytest.approx(exp_recall)
#         assert ap == pytest.approx(exp_ap)
import unittest 
import numpy as np


class TestObjectDetector(unittest.TestCase):
    def test_precision_recall_ap(self):
        all_confs = np.linspace(0, 1, 1000)

        # Test case 1: Perfect detection
        tp = np.array([[True], [True]])
        conf = np.array([0.9, 0.8])
        pred_cls = np.array([0, 1])
        target_cls = np.array([0, 1])

        total_recall_per_conf, total_precision_per_conf, total_ap = calc_total_det_stats(tp, conf, pred_cls, target_cls)
        self.assertEqual(all_confs.size, total_recall_per_conf.shape[0])
        self.assertTrue(np.all(total_precision_per_conf == 1.0))
        self.assertTrue(np.all(total_recall_per_conf[:800] == 1.0))
        self.assertTrue(np.all(total_recall_per_conf[900:] == 1.0))
        self.assertTrue(np.allclose(total_ap, 1.0, atol=0.01))
        self.assertTrue(np.allclose(total_recall_per_conf[899], 0.5, atol=0.01))

if __name__ == '__main__':
    unittest.main()
