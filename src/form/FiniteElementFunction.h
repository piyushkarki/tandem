#ifndef FINITEELEMENTFUNCTION_20200630_H
#define FINITEELEMENTFUNCTION_20200630_H

#include "form/RefElement.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace tndm {

template <std::size_t D> class FiniteElementFunction {
public:
    FiniteElementFunction(std::unique_ptr<RefElement<D>> refElement, double const* data,
                          std::size_t numberOfBasisFunctions, std::size_t numberOfQuantities,
                          std::size_t numberOfElements)
        : refElement_(std::move(refElement)),
          data_(numberOfBasisFunctions, numberOfQuantities, numberOfElements) {
        std::copy(data, data + data_.size(), data_.data());
    }

    Managed<Matrix<double>>
    evaluationMatrix(std::vector<std::array<double, D>> const& points) const;

    TensorBase<Matrix<double>> mapResultInfo(std::size_t numPoints) const;
    void map(std::size_t eleNo, Matrix<double> const& evalMatrix, Tensor<double, 2u>& result) const;

    RefElement<D> const& refElement() const { return *refElement_; }
    std::size_t numBasisFunctions() const { return data_.shape(0); }
    std::size_t numQuantities() const { return data_.shape(1); }
    std::size_t numElements() const { return data_.shape(2); }

private:
    std::unique_ptr<RefElement<D>> refElement_;
    Managed<Tensor<double, 3u>> data_;
};

} // namespace tndm

#endif // FINITEELEMENTFUNCTION_20200630_H