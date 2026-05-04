from training.core.experiment_tracking import REWARD_TERM_KEYS
from training.core.metrics_registry import METRIC_INDEX


def test_all_logged_reward_terms_exist_in_metrics_registry():
    missing_terms = [term for term in REWARD_TERM_KEYS if term not in METRIC_INDEX]
    assert missing_terms == []
