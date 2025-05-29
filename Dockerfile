#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#STAGE 1 : Pull a base image and install the first layer of dependencies
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

#Add a non - root user to avoid permission issues
RUN groupadd -r tandem && useradd -r -g tandem -m -s /bin/bash tandem

#Install system level dependencies
RUN apt-get update && apt-get install -y \
    # compilers and build tools
    gcc-13 g++-13 clang-18 clang++-18 cmake \
    make build-essential ca-certificates \
    # parallel computing libraries
    libopenmpi-dev openmpi-bin libomp-dev libgomp1 \
    # Meshing and partitioning library
    gmsh libmetis-dev libparmetis-dev \
    # Math libraries
    libeigen3-dev python3-numpy \
    # Linear algebra library
    libopenblas-dev \
    # Lua library for tandem parameter script support
    liblua5.3-dev \
    # Python and virtual environment tools
    python3 python3-pip python3-venv \
    # Download and git tools
    wget git jq curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and activate Python virtual environment
RUN python3 -m venv $VENV_PATH && \
    . $VENV_PATH/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    echo "source $VENV_PATH/bin/activate" >> /etc/bash.bashrc

RUN echo "Base image and system dependencies installed successfully."

# -----------------------------------------------------------------------
# STAGE 2: Build dependencies specific to Tandem from source
# -----------------------------------------------------------------------
FROM base AS tandem_dependencies

ARG PETSC_VERSION=3.22.0
ARG CC=mpicc
ARG CXX=mpic++

ENV CC=${CC}
ENV CXX=${CXX}
ENV CFLAGS="-O2 -Wall -DNDEBUG"
ENV CXXFLAGS="-O2 -Wall -DNDEBUG"
ENV PETSC_INSTALL_DIR=/opt/petsc
ENV PATH="/opt/venv/bin:$PATH"

# Create install directory
RUN mkdir -p ${PETSC_INSTALL_DIR}

# Install libxsmm for kernel generation
RUN git clone --depth 1 --branch 1.17 https://github.com/libxsmm/libxsmm.git && \
    cd libxsmm && \
    make generator CC=${CC} CXX=${CXX} CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" -j$(nproc) && \
    cp bin/libxsmm_gemm_generator /usr/bin && \
    cd .. && rm -rf libxsmm

# Install PETSc from source
RUN echo "Using PETSc version $PETSC_VERSION" && \
    wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz && \
    tar -xf petsc-${PETSC_VERSION}.tar.gz && rm -rf petsc-${PETSC_VERSION}.tar.gz && cd petsc-${PETSC_VERSION} && \
    PETSC_DIR=$(pwd) && ./configure --with-fortran-bindings=0 --with-debugging=0 \
    --with-memalign=32 --with-64-bit-indices \
    --with-cc="$CC" --with-cxx="$CXX" --with-fc=0 --prefix=$PETSC_INSTALL_DIR \
    --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3" --with-mpi-dir=/usr/lib/x86_64-linux-gnu/openmpi && \
    make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt -j$(nproc) && \
    make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install && \
    rm -rf petsc-${PETSC_VERSION}.tar.gz petsc-${PETSC_VERSION}

# Save PETSc version for reference
RUN echo "${PETSC_VERSION}" > ${PETSC_INSTALL_DIR}/version.txt

# Install common Python packages for development and CI
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir \
    numpy scipy matplotlib \
    pytest pytest-cov \
    black flake8 mypy \
    requests PyYAML

RUN echo "Dependencies Installed Successfully. Installing Tandem source code."

WORKDIR /app

RUN ls -lah /app && chown -R tandem:tandem /app

# Switch to non-root user
USER tandem

# Activate virtual environment by default
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]