# ---------- base image ----------
FROM condaforge/mambaforge:latest

ENV DEBIAN_FRONTEND=noninteractive

# ---- System packages ----
# OpenMPI 4.1.x is needed by ORCA 6.1 parallel modules (libmpi.so.40).
# ssh is required by OpenMPI for spawning processes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenmpi-dev \
        openmpi-bin \
        openssh-client \
        ca-certificates \
        curl \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ---- Python environment (RDKit + deps via conda-forge) ----
RUN mamba install -y -c conda-forge \
        python=3.11 \
        rdkit \
        numpy \
        scipy \
    && mamba clean -afy

# ---- ORCA installation ----
# ORCA is proprietary — extract the tarball into an "orca/" directory next to
# this Dockerfile before building.  Never push a built image to a public registry.
COPY orca/ /opt/orca/
RUN chmod +x /opt/orca/orca
ENV PATH="/opt/orca:${PATH}"
# ORCA ships its own liborca_tools + libstdc++ under orca/lib/
ENV LD_LIBRARY_PATH="/opt/orca/lib:/opt/orca:${LD_LIBRARY_PATH}"

# ---- xtb → otool_xtb ----
# If otool_xtb is not already in the ORCA directory, download it automatically.
ARG XTB_VERSION=6.7.1
RUN if [ ! -f /opt/orca/otool_xtb ]; then \
        curl -fSL "https://github.com/grimme-lab/xtb/releases/download/v${XTB_VERSION}/xtb-${XTB_VERSION}-linux-x86_64.tar.xz" \
            -o /tmp/xtb.tar.xz && \
        tar xf /tmp/xtb.tar.xz -C /tmp && \
        cp /tmp/xtb-*/bin/xtb /opt/orca/otool_xtb && \
        chmod +x /opt/orca/otool_xtb && \
        rm -rf /tmp/xtb*; \
    fi

# Allow OpenMPI to run as root inside the container
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# ---- Application code ----
WORKDIR /app
COPY ConformerGenerator.py input_parsers.py cpcm_radii.inp ./

# Default working directory for calculations (mount your data here)
WORKDIR /data

ENTRYPOINT ["python", "/app/ConformerGenerator.py"]
CMD ["--structures_file", "structures.inp", "--cpcm_radii_file", "/app/cpcm_radii.inp", "--n_cores", "2"]
