#ifndef RATEANDSTATEBASE_20201026_H
#define RATEANDSTATEBASE_20201026_H

#include "config.h"

#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "quadrules/AutoRule.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"

#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class RateAndStateBase {
public:
    static auto Space() -> NodalRefElement<DomainDimension - 1u>;

    static constexpr std::size_t TangentialComponents = DomainDimension - 1u;
    // D-1 slip components + 1 state component
    constexpr static std::size_t NumQuantities = TangentialComponents + 1u;

    RateAndStateBase(std::shared_ptr<Curvilinear<DomainDimension>> cl);

    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }
    std::size_t slip_block_size() const {
        return space_.numBasisFunctions() * TangentialComponents;
    }
    std::size_t scratch_mem_size() const { return 1; }

    void begin_preparation(std::size_t numFaultFaces);
    void prepare(std::size_t faultNo, FacetInfo const& info, LinearAllocator<double>&);
    void end_preparation() {}

    auto const& space() const { return space_; }
    std::vector<std::vector<double>> allWeightDetProducts;

protected:
    std::shared_ptr<Curvilinear<DomainDimension>> cl_;
    std::size_t nq;
    // Basis
    NodalRefElement<DomainDimension - 1u> space_;
    std::vector<Managed<Matrix<double>>> geoE_q;

    std::vector<double> wgts;
    std::vector<std::array<double, 2>> pts;
    Managed<Matrix<double>>
        interpolate_matrix_basis_to_quad; // = space_.evaluateBasisAt(space_.refNodes(),{1,0});
    //
    std::vector<Managed<Tensor<double, 3u>>> geoDxi_q;
    std::vector<std::vector<double>> allDeterminantJ;

    struct Coords {
        using type = std::array<double, DomainDimension>;
    };
    struct FaultBasis {
        using type = std::array<double, DomainDimension * DomainDimension>;
    };
    struct SignFlipped {
        using type = bool;
    };
    struct Normal {
        using type = std::array<double, DomainDimension>;
    };
    struct NormalLength {
        using type = double;
    };
    std::array<double, DomainDimension> ref_normal_={0,-1,0};

    using fault_t = mneme::MultiStorage<mneme::DataLayout::SoA, Coords, FaultBasis, SignFlipped,
                                        Normal, NormalLength>;
    mneme::StridedView<fault_t> fault_;
    std::array<double, DomainDimension> up_ = {0,0,1};
    Tensor<double,3> fault_basis_q;
};

} // namespace tndm

#endif // RATEANDSTATEBASE_20201026_H
