#include "Poisson.h"

#include "basis/WarpAndBlend.h"
#include "config.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "kernels/init.h"
#include "kernels/kernel.h"
#include "kernels/tensor.h"
#include "parallel/MPITraits.h"
#include "tensor/EigenMap.h"

#include <limits>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

namespace tndm {

Poisson::Poisson(LocalSimplexMesh<DomainDimension> const& mesh, Curvilinear<DomainDimension>& cl,
                 std::unique_ptr<RefElement<DomainDimension>> refElement, unsigned minQuadOrder,
                 functional_t kFun)
    : DG(mesh, cl, std::move(refElement), minQuadOrder),
      nodalRefElement_(PolynomialDegree, WarpAndBlendFactory<DomainDimension>()) {

    userVol.setStorage(
        std::make_shared<user_vol_t>(numElements() * nodalRefElement_.numBasisFunctions()), 0u,
        numElements(), nodalRefElement_.numBasisFunctions());

    auto Minv = nodalRefElement_.inverseMassMatrix();
    auto E = nodalRefElement_.evaluateBasisAt(volRule.points());
    Eigen::MatrixXd P = EigenMap(Minv) * EigenMap(E);
#pragma omp parallel
    {
        auto rhs = Eigen::VectorXd(volRule.size());
#pragma omp for
        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            auto K = Vector<double>(userVol[elNo].data(), nodalRefElement_.numBasisFunctions());

            auto coords = vol[elNo].get<Coords>();
            for (std::size_t q = 0; q < volRule.size(); ++q) {
                rhs(q) = kFun(coords[q]) * volRule.weights()[q];
            }
            EigenMap(K) = P * rhs;
        }
    }

    Em = nodalRefElement_.evaluateBasisAt(volRule.points(), {1, 0});
    for (std::size_t f = 0; f < DomainDimension + 1u; ++f) {
        auto points = cl.facetParam(f, fctRule.points());
        em.emplace_back(nodalRefElement_.evaluateBasisAt(points, {1, 0}));
    }
}

Mat Poisson::assemble() {
    Mat mat;
    PetscInt blockSize = tensor::A::Shape[0];
    PetscInt localRows = numElements() * tensor::A::Shape[0];
    PetscInt localCols = numElements() * tensor::A::Shape[1];
    MatCreateBAIJ(PETSC_COMM_WORLD, blockSize, localRows, localCols, PETSC_DETERMINE,
                  PETSC_DETERMINE, 5, nullptr, 5, nullptr, &mat);
    MatSetOption(mat, MAT_ROW_ORIENTED, PETSC_FALSE);

    double D_x[tensor::D_x::size()];
    double A[tensor::A::size()];
    auto Aview = init::A::view::create(A);

    assert(volRule.size() == tensor::W::Shape[0]);
    assert(D_xi.shape(0) == tensor::D_xi::Shape[0]);
    assert(D_xi.shape(1) == tensor::D_xi::Shape[1]);
    assert(D_xi.shape(2) == tensor::D_xi::Shape[2]);

    for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
        kernel::assembleVolume krnl;
        krnl.A = A;
        krnl.D_x = D_x;
        krnl.D_xi = D_xi.data();
        krnl.G = vol[elNo].template get<JInv>().data()->data();
        krnl.K = userVol[elNo].data();
        krnl.Em = Em.data();
        krnl.J = vol[elNo].template get<AbsDetJ>().data();
        krnl.W = volRule.weights().data();
        krnl.execute();
        PetscInt ib = volInfo[elNo].cGID;
        PetscInt jb = volInfo[elNo].cGID;
        MatSetValuesBlocked(mat, 1, &ib, 1, &jb, A, ADD_VALUES);
    }

    double a00[tensor::a::size(0, 0)];
    double a01[tensor::a::size(0, 1)];
    double a10[tensor::a::size(1, 0)];
    double a11[tensor::a::size(1, 1)];
    double d_x0[tensor::d_x::size(0)];
    double d_x1[tensor::d_x::size(1)];

    assert(fctRule.size() == tensor::w::Shape[0]);
    assert(e[0].shape(0) == tensor::e::Shape[0][0]);
    assert(e[0].shape(1) == tensor::e::Shape[0][1]);
    assert(d_xi[0].shape(0) == tensor::d_xi::Shape[0][0]);
    assert(d_xi[0].shape(1) == tensor::d_xi::Shape[0][1]);
    assert(d_xi[0].shape(2) == tensor::d_xi::Shape[0][2]);

    for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        kernel::assembleFacetLocal local;
        double half = (info.up[0] != info.up[1]) ? 0.5 : 1.0;
        local.c00 = -half;
        local.c10 = epsilon * half;
        local.c20 = penalty(info);
        local.a(0, 0) = a00;
        local.d_x(0) = d_x0;
        local.d_xi(0) = d_xi[info.localNo[0]].data();
        local.e(0) = e[info.localNo[0]].data();
        local.em(0) = em[info.localNo[0]].data();
        local.g = fct[fctNo].template get<JInv>().data()->data();
        local.K = userVol[info.up[0]].data();
        local.n = fct[fctNo].template get<Normal>().data()->data();
        local.nl = fct[fctNo].template get<NormalLength>().data();
        local.w = fctRule.weights().data();
        local.execute();

        auto push = [&info, &mat](auto x, auto y, double* a) {
            PetscInt ib = info.g_up[x()];
            PetscInt jb = info.g_up[y()];
            MatSetValuesBlocked(mat, 1, &ib, 1, &jb, a, ADD_VALUES);
        };

        push(std::integral_constant<int, 0>(), std::integral_constant<int, 0>(), a00);

        if (info.up[0] != info.up[1]) {
            kernel::assembleFacetNeighbour neighbour;
            neighbour.c00 = local.c00;
            neighbour.c01 = -local.c00;
            neighbour.c10 = local.c10;
            neighbour.c11 = -local.c10;
            neighbour.c20 = local.c20;
            neighbour.c21 = -local.c20;
            neighbour.a(0, 1) = a01;
            neighbour.a(1, 0) = a10;
            neighbour.a(1, 1) = a11;
            neighbour.d_x(0) = d_x0;
            neighbour.d_x(1) = d_x1;
            neighbour.d_xi(1) = d_xi[info.localNo[1]].data();
            neighbour.e(0) = local.e(0);
            neighbour.em(0) = local.em(0);
            neighbour.e(1) = e[info.localNo[1]].data();
            neighbour.em(1) = em[info.localNo[1]].data();
            neighbour.g = fct[fctNo].template get<JInvOther>().data()->data();
            neighbour.K = userVol[info.up[1]].data();
            neighbour.n = local.n;
            neighbour.nl = local.nl;
            neighbour.w = local.w;
            neighbour.execute();

            push(std::integral_constant<int, 0>(), std::integral_constant<int, 1>(), a01);
            push(std::integral_constant<int, 1>(), std::integral_constant<int, 0>(), a10);
            push(std::integral_constant<int, 1>(), std::integral_constant<int, 1>(), a11);
        }
    }

    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    MatSetOption(mat, MAT_SYMMETRIC, PETSC_TRUE);
    return mat;
}

Vec Poisson::rhs(functional_t forceFun, functional_t dirichletFun) {
    Vec B;
    PetscInt localRows = numElements() * tensor::A::Shape[0];
    VecCreateMPI(PETSC_COMM_WORLD, localRows, PETSC_DETERMINE, &B);
    VecSetBlockSize(B, tensor::b::Shape[0]);

    double b[tensor::b::size()];

    assert(tensor::b::Shape[0] == tensor::A::Shape[0]);

    double F[tensor::F::size()];
    assert(tensor::F::size() == volRule.size());
    for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
        auto coords = vol[elNo].template get<Coords>();
        for (unsigned q = 0; q < tensor::F::size(); ++q) {
            F[q] = forceFun(coords[q]);
        }

        kernel::rhsVolume rhs;
        rhs.E = E.data();
        rhs.F = F;
        rhs.J = vol[elNo].template get<AbsDetJ>().data();
        rhs.W = volRule.weights().data();
        rhs.b = b;
        rhs.execute();

        PetscInt ib = volInfo[elNo].cGID;
        VecSetValuesBlocked(B, 1, &ib, b, ADD_VALUES);
    }

    double f[tensor::f::size()];
    assert(tensor::f::size() == fctRule.size());

    for (std::size_t fctNo = 0; fctNo < numFacets(); ++fctNo) {
        auto const& info = fctInfo[fctNo];
        if (info.up[0] == info.up[1]) {
            auto coords = fct[fctNo].template get<Coords>();
            for (unsigned q = 0; q < tensor::f::size(); ++q) {
                f[q] = dirichletFun(coords[q]);
            }

            kernel::rhsFacet rhs;
            rhs.c10 = epsilon;
            rhs.c20 = penalty(info);
            rhs.b = b;
            rhs.d_xi(0) = d_xi[info.localNo[0]].data();
            rhs.e(0) = e[info.localNo[0]].data();
            rhs.em(0) = em[info.localNo[0]].data();
            rhs.f = f;
            rhs.g = fct[fctNo].template get<JInv>().data()->data();
            rhs.K = userVol[info.up[0]].data();
            rhs.n = fct[fctNo].template get<Normal>().data()->data();
            rhs.nl = fct[fctNo].template get<NormalLength>().data();
            rhs.w = fctRule.weights().data();
            rhs.execute();

            PetscInt ib = info.g_up[0];
            VecSetValuesBlocked(B, 1, &ib, b, ADD_VALUES);
        }
    }
    VecAssemblyBegin(B);
    VecAssemblyEnd(B);
    return B;
}

FiniteElementFunction<DomainDimension> Poisson::finiteElementFunction(Vec x) const {
    // assert(numeric.rows() == numElements() * tensor::b::Shape[0]);
    PetscScalar const* values;
    VecGetArrayRead(x, &values);
    return FiniteElementFunction<DomainDimension>(refElement_->clone(), values, tensor::b::Shape[0],
                                                  1, numElements());
    VecRestoreArrayRead(x, &values);
}

} // namespace tndm