from src.fuzzy_controller import sigmoid_dimming


def test_sigmoid_bounds():
    limit = 50.0
    for l in [5, 25, 50, 70, 120]:
        val = sigmoid_dimming(l, limit)
        assert isinstance(val, float)
        assert val <= limit * 1.05
