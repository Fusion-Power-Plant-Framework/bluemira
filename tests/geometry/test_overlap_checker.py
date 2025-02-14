import numpy as np

from bluemira.geometry.overlap_checker import get_overlaps, is_mutually_exclusive

np.random.seed(0.0)
test_ranges = []
for _ in range(5):
    example_ranges = np.random.rand(10).reshape(2, 5)
    example_ranges.sort(axis=0)
    test_ranges.append(example_ranges)


@pytest.mark.parametrize(
    ("min_", "max_"),
    [tuple(example_ranges) for example_ranges in test_ranges],
)
def test_exclusivity_matrix_is_symmetric(min_, max_):
    exclusivity_matrix = is_mutually_exclusive(min_, max_)
    np.testing.assert_equal(exclusivity_matrix, exclusivity_matrix.T)


def test_mutual_exclusivity():
    box1 = [0, 1]
    box2 = [1, 2]
    box3 = [1.9, 3]
    box4 = [1.9, 4]
    box5 = [4.5, 5]
    expected_collision_indices = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    min_, max_ = np.array([box1, box2, box3, box4, box5]).T
    exclusivity_matrix = is_mutually_exclusive(min_, max_)
    overlap_matrix = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_equal(overlap_matrix, ~exclusivity_matrix)

    np.testing.assert_equal(get_overlaps(exclusivity_matrix), expected_collision_indices)
