# openCOSMO-RS_conformer_pipeline

Important for the execution are following requirements:
- Working installation of ORCA 6.1+ (please make sure to download and extract all available compressed files for your OS)
- The folder containing the ORCA installation has to be added to the path environment variable.
- Install the xtb executable from the [official repo releases](https://github.com/grimme-lab/xtb/releases). To do this copy the main binary from the 'bin' folder inside the compressed file into the ORCA installation directory and rename it to 'otool_xtb' or 'otool_xtb.exe' depending on your OS.
- In case you want to use balloon to generate conformers instead of rdkit, you can get balloon [here](http://users.abo.fi/mivainio/balloon/download.php).
- If you want to run parallelized calculations please check the ORCA manual to see how to setup this properly (installing MPI, setting paths, etc.)

The idea is that you can run the conformer generator as follows:
```python
python ConformerGenerator.py --structures_file file.inp --cpcm_radii cpcm_radii.inp --n_cores 2
```

with file.inp beeing a TAB separated file similar to the following:

name&nbsp;_[TAB]_&nbsp;SMILES&nbsp;_[TAB]_&nbsp;optional xyz/sdf file&nbsp;_[TAB]_&nbsp;charge&nbsp;_[TAB]_&nbsp;optional geometry optimization

The structure file column accepts `.xyz`, `.sdf`, or `.mol` files. SDF/MOL files are parsed directly for bond connectivity, so no bond-order guessing is needed.

The optional geometry optimization column assumes [0, False, no] to be False and everything else to be True.

Examples:
```
methane[TAB]C[TAB]methane.xyz[TAB]0[TAB]True
ethanol[TAB]CCO[TAB]ethanol.sdf[TAB]0
propanediol[TAB][TAB]propanediol.sdf[TAB]0
```

## Docker

Place your ORCA 6.0+ installation in an `orca/` directory next to the Dockerfile, then:

```bash
docker compose up          # builds image & runs with default settings
```

Mount your data directory and override the defaults as needed:

```bash
docker run --rm -v $(pwd)/data:/data opencosmo-conformer \
  --structures_file structures.inp --cpcm_radii_file /app/cpcm_radii.inp --n_cores 4
```

> **Note:** The image contains ORCA, which is proprietary — do not push built images to a public registry.

## Sigma Profile

This is to calculate and plot the sigma profile:
```python
from matplotlib import pyplot as plt
from input_parsers import SigmaProfileParser

spp = SigmaProfileParser(r'path.orcacosmo')
sigmas, areas = spp.cluster_and_create_sigma_profile()

plt.plot(sigmas, areas)
plt.xlim([-0.03, 0.03])
plt.savefig('sigma_profile.png')
```
