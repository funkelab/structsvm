import structsvm as ssvm


def test_quadratic():

    # f(x) = (x - 1)**2
    def value_gradient(x):
        return (x[0] - 1.0)**2, 2*(x - 1)

    bundle_method = ssvm.BundleMethod(
        value_gradient,
        dims=1,
        regularizer_weight=0.0001,
        eps=1e-5)

    w = bundle_method.optimize(max_iterations=100)
    assert round(w[0], 5) == 0.99897
