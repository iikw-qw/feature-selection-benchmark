import pytest

from src import core, data


@pytest.fixture(scope="module")
def dataset():
    return data.load_with_noise(n_noise=30, n_correlated=5, seed=1)


def test_dataset_layout(dataset):
    x, y = dataset
    assert x.shape == (569, 30 + 30 + 5)
    assert sum(data.is_real(c) for c in x.columns) == 30


def test_jaccard():
    assert core.jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert core.jaccard([1, 2], [3, 4]) == 0.0
    assert core.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)


def test_selector_returns_requested_count(dataset):
    x, y = dataset
    names = core.selected_names(x, y, core.selectors(8)["anova_f"])
    assert len(names) == 8


def test_filter_methods_avoid_noise(dataset):
    x, y = dataset
    names = core.selected_names(x, y, core.selectors(8)["anova_f"])
    assert sum(c.startswith("noise_") for c in names) == 0


def test_benchmark_table_has_all_methods(dataset):
    x, y = dataset
    table = core.benchmark(x, y, k=8)
    assert set(table.index) == set(core.selectors(8))
    assert table["auc_mean"].min() > 0.9
