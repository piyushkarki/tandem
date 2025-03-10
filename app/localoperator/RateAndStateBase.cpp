#include "RateAndStateBase.h"

#include "basis/WarpAndBlend.h"
#include "tensor/Tensor.h"
#include <algorithm>
#include <array> // If you're iterating over an array
#include <iostream>
#include <memory>
#include <vector> // If you're iterating over a vector
#include "geometry/Vector.h"

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
    nq = rule.size();
    interpolate_matrix_basis_to_quad = space_.evaluateBasisAt(pts);
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
    // auto J = Managed(cl_->jacobianResultInfo(pts.size()));
    // auto determinantJ = Managed(cl_->detJResultInfo(pts.size()));
    // cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    // cl_->detJ(info.up[0], J, determinantJ);


    auto J = Managed(cl_->jacobianResultInfo(nq));
    auto JInv = Managed(cl_->jacobianResultInfo(nq));
    auto determinantJ =Managed(cl_->detJResultInfo(nq));
    auto normal =
        Tensor(fault_[faultNo].template get<Normal>().data()->data(), cl_->normalResultInfo(nq));
    auto& nl = fault_[faultNo].template get<NormalLength>();
    fault_basis_q = Tensor(fault_[faultNo].template get<FaultBasis>().data()->data(),
                                cl_->facetBasisResultInfo(nq));
    cl_->jacobian(info.up[0], geoDxi_q[info.localNo[0]], J);
    cl_->detJ(info.up[0], J, determinantJ);
    cl_->jacobianInv(J, JInv);
    cl_->normal(info.localNo[0], determinantJ, JInv, normal);
    for (std::size_t i = 0; i < nq; ++i) {
        auto& sign_flipped = fault_[faultNo].template get<SignFlipped>()[i];
        auto& normal_i = fault_[faultNo].template get<Normal>()[i];
        nl[i] = norm(normal_i);

        auto n_ref_dot_n = dot(ref_normal_, normal_i);
        if (std::fabs(n_ref_dot_n) < 10000.0 * std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("Normal and reference normal are almost perpendicular.");
        }
        sign_flipped = n_ref_dot_n < 0;
        if (sign_flipped) {
            normal_i = -1.0 * normal_i;
        }
    }
    cl_->facetBasis(up_, normal, fault_basis_q);
    for (std::size_t q = 0; q < nq; ++q) {
        if (fault_[faultNo].template get<SignFlipped>()[q]) {
            for (std::size_t i = 0; i < fault_basis_q.shape(1); ++i) {
                for (std::size_t j = 0; j < fault_basis_q.shape(0); ++j) {
                    fault_basis_q(i, j, q) *= -1.0;
                }
            }
        }
    }




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
