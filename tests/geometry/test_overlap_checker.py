import numpy as np

np.random.seed(0.0)
test_ranges = []
for i in range(5):
    example_ranges = np.random.rand(10).reshape(2, 5)
    example_ranges.sort(axis=0)
    test_ranges.append(example_ranges)


@pytest.mark.parametrize(
    ("min_", "max_"),
    [tuple(example_ranges) for example_ranges in test_ranges],
)
def test_exclusivity_matrix_is_symmetric(min_, max_):
    matrix = is_mutually_exclusive(min_, max_)
    np.testing.assert_equal(matrix, matrix.T)


def test_mutual_exclusivity():
    min_ = []
    max_ = []
    matrix = is_mutually_exclusive(min_, max_)
    np.testing.assert_equal(
        matrix,
        np.array([
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]),
    )
