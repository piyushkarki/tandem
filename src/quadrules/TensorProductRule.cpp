#include "TensorProductRule.h"
#include "GaussJacobi.h"

namespace tndm {

template <> SimplexQuadratureRule<1u> tensorProductRule<1u>(unsigned n) {
    SimplexQuadratureRule<1u> rule(n);

    auto gj = GaussJacobi(n, 0, 0);
    gj.changeInterval(0.0, 1.0);

    for (std::size_t i = 0; i < n; ++i) {
        rule.points()[i][0] = gj.points()[i];
        rule.weights()[i] = gj.weights()[i];
    }

    return rule;
}

template <> SimplexQuadratureRule<2u> tensorProductRule<2u>(unsigned n) {
    SimplexQuadratureRule<2u> rule(n * n);

    auto gj0 = GaussJacobi(n, 0, 0);
    auto gj1 = GaussJacobi(n, 1, 0);

    gj0.changeInterval(0.0, 1.0);
    gj1.changeInterval(0.0, 1.0);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            rule.points()[idx][0] = gj1.points()[i];
            rule.points()[idx][1] = gj0.points()[j] * (1.0 - gj1.points()[i]);
            rule.weights()[idx] = 0.5 * gj1.weights()[i] * gj0.weights()[j];
            ++idx;
        }
    }

    return rule;
}

template <> SimplexQuadratureRule<3u> tensorProductRule<3u>(unsigned n) {
    SimplexQuadratureRule<3u> rule(n * n * n);

    auto gj0 = GaussJacobi(n, 0, 0);
    auto gj1 = GaussJacobi(n, 1, 0);
    auto gj2 = GaussJacobi(n, 2, 0);

    gj0.changeInterval(0.0, 1.0);
    gj1.changeInterval(0.0, 1.0);
    gj2.changeInterval(0.0, 1.0);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                rule.points()[idx][0] = gj2.points()[i];
                rule.points()[idx][1] = gj1.points()[j] * (1 - gj2.points()[i]);
                rule.points()[idx][2] =
                    gj0.points()[k] * (1 - gj1.points()[j]) * (1 - gj2.points()[i]);
                rule.weights()[idx] =
                    0.125 * gj2.weights()[i] * gj1.weights()[j] * gj0.weights()[k];
                ++idx;
            }
        }
    }

    return rule;
}

} // namespace tndm
