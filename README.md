# openCOSMO-RS_conformer_pipeline

The idea is that you can run the conformer generator as follows.
Important is that the folder containing the executable has to be added to the path environment variable.

```python
python ConformerGenerator.py --structures_file file.inp --n_cores 2
```

with file.inp beeing a TAB separated file similar to the following:

name&nbsp;_[TAB]_&nbsp;SMILES&nbsp;_[TAB]_&nbsp;charge

Examples:
```
methane[TAB]C[TAB]0
ethanol[TAB]CCO[TAB]0
```
