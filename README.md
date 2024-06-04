# openCOSMO-RS_conformer_pipeline

Important for the execution are following requirements:
- Working installation of ORCA 5.0.3+ (please make sure to download and extract all available compressed files for your OS)
- The folder containing the ORCA installation has to be added to the path environment variable.
- Install the xtb executable from the [official repo releases](https://github.com/grimme-lab/xtb/releases). To do this copy the main binary from the 'bin' folder inside the compressed file into the ORCA installation directory and rename it to 'otool_xtb' or 'otool_xtb.exe' depending on your OS.
- If you want to run parallelized calculations please check the ORCA manual to see how to setup this properly (installing MPI, setting paths, etc.)

The idea is that you can run the conformer generator as follows:
```python
python ConformerGenerator.py --structures_file file.inp --cpcm_radii cpcm_radii.inp --n_cores 2
```

with file.inp beeing a TAB separated file similar to the following:

name&nbsp;_[TAB]_&nbsp;SMILES&nbsp;_[TAB]_&nbsp;optional xyz file&nbsp;_[TAB]_&nbsp;charge

Examples:
```
methane[TAB]C[TAB]xyz_file[TAB]0
ethanol[TAB]CCO[TAB][TAB]0
```
