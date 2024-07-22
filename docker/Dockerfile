FROM julia:1.10-bullseye

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Amsterdam

ENV JULIA_DEPOT_PATH=/opt

RUN apt-get update && \
    apt-get -y install gcc mono-mcs vim procps wget lsb-release software-properties-common gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# Import the project

COPY . /opt/Simulations

WORKDIR /opt/Simulations

# Precompile and Test project
RUN julia --project -e'using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()'

# Build Sysimage
RUN julia --project --startup-file=no deps/build_sysimage.jl

# Check script is callable

RUN julia --project --startup-file=no -JSimulations.so targene-simulation.jl --help