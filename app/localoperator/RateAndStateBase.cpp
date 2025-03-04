#include "RateAndStateBase.h"

#include "basis/WarpAndBlend.h"
#include "tensor/Tensor.h"
#include <algorithm>
#include <array> // If you're iterating over an array
#include <iostream>
#include <memory>
#include <vector> // If you're iterating over a vector

namespace tndm {

auto RateAndStateBase::Space() -> NodalRefElement<DomainDimension - 1u> {
    return NodalRefElement<DomainDimension - 1u>(
        PolynomialDegree, WarpAndBlendFactory<DomainDimension - 1u>(), ALIGNMENT);
}

RateAndStateBase::RateAndStateBase(std::shared_ptr<Curvilinear<DomainDimension>> cl)
    : cl_(std::move(cl)), space_(Space()) {
    auto rule = simplexQuadratureRule<DomainDimension - 1u>(7);
    wgts = rule.weights();
    pts = rule.points();
    interpolate_matrix_basis_to_quad = space_.evaluateBasisAt(pts, {1, 0});
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl_->facetParam(f, space_.refNodes());
        geoE_q.emplace_back(cl_->evaluateBasisAt(facetParam));
    }
}

void RateAndStateBase::begin_preparation(std::size_t numFaultFaces) {
    auto nbf = space_.numBasisFunctions();
    fault_.setStorage(std::make_shared<fault_t>(numFaultFaces * nbf), 0u, numFaultFaces, nbf);
}

void RateAndStateBase::prepare(std::size_t faultNo, FacetInfo const& info,
                               LinearAllocator<double>&) {
    auto nbf = space_.numBasisFunctions();
    auto coords =
        Tensor(fault_[faultNo].template get<Coords>().data()->data(), cl_->mapResultInfo(nbf));
    cl_->map(info.up[0], geoE_q[info.localNo[0]], coords);
    
    // find centroid of triangle
    auto x_avg = coords(0, 0) + coords(1, 0) + coords(2, 0);
    auto y_avg = coords(0, 1) + coords(1, 1) + coords(2, 1);
    
    auto distance = std::sqrt(x_avg * x_avg + y_avg * y_avg);
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto facetParam = cl_->facetParam(f, pts);
        geoDxi_q.emplace_back(cl_->evaluateGradientAt(facetParam));
    }
    auto J = Managed(cl_->jacobianResultInfo(pts.size()));
    auto determinantJ = Managed(cl_->detJResultInfo(pts.size()));
    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, determinantJ);
    std::vector<double> detElem;
    for (int i=0; i<determinantJ.shape(0); i++){
        detElem.push_back(determinantJ(i));
    }
    allDeterminantJ.push_back(detElem);
    std::vector<double> weightDetProduct;
    auto rvw = 0.2;
    for (std::size_t i = 0; i < determinantJ.shape(0); ++i) {
        if (distance<=rvw){
            weightDetProduct.push_back(wgts[i] * std::abs(determinantJ(i)));
        }
        else{
            weightDetProduct.push_back(0.0);
        }
    }
    // Store the product for this fault face
    allWeightDetProducts.push_back(weightDetProduct);
}


} // namespace tndm
