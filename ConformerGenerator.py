import copy
import os
import re
import shutil
import time
import zipfile
import platform
import argparse
import csv

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

import scipy.constants as spcon
import subprocess as spr

from abc import ABC, abstractmethod

from input_parsers import COSMOParser

kJ_per_kcal = 4.184                   # kJ/kcal
kJdivmol_per_hartree = 2625.499639479 # (kJ/mol)/hartree

class ConformerGenerator(object):

    @staticmethod
    def get_embedded_mol(smiles):

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Could not generate mol struct from smiles: {}'.
                             format(smiles))

        mol = Chem.AddHs(mol)
        retVal = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        if retVal < 0 :
            retVal = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if retVal == 0 :
            if AllChem.UFFOptimizeMolecule(mol) != 0 :
                retVal = AllChem.UFFOptimizeMolecule(mol, maxIters=10000)
                if retVal != 0:
                    raise ValueError('Molecule could not be embedded.')

        mol.GetConformer(0).SetDoubleProp('energy', 0)

        return mol

    @staticmethod
    def save_conformer_to_xyz_file(conformer, filepath_xyz, comment=None):

        if comment is None:
            if conformer.HasProp('energy'):
                energy_value = '{:.16f}'.format(conformer.GetDoubleProp('energy'))
                comment = f'energy: {energy_value}'

        with open(filepath_xyz, 'w') as xyzf:
            mol = conformer.GetOwningMol()
            n_atoms = len(mol.GetAtoms())
            xyzf.write(f'{n_atoms}\n') 
            xyzf.write(f'{comment}\n')
            atom_positions = conformer.GetPositions()
            for i in range(n_atoms):
                xyzf.write('{:s}  {:.16f}  {:.16f}  {:.16f}\n'.format(
                    mol.GetAtomWithIdx(i).GetSymbol(), atom_positions[i][0],
                    atom_positions[i][1], atom_positions[i][2]))

    @staticmethod
    def calculate_best_rms(probe_molecule, reference_molecule, 
                            probe_molecule_conformer_index=0, reference_molecule_conformer_index=0,
                            only_heavy_atoms=False):

        # make the appropriate copy to not modify input
        if only_heavy_atoms:
            probe_mol = Chem.RemoveHs(probe_molecule)
            reference_mol = Chem.RemoveHs(reference_molecule)
        else:
            probe_mol = Chem.Mol(probe_molecule)
            reference_mol = Chem.Mol(reference_molecule)

        return rdMolAlign.GetBestRMS(probe_mol, reference_mol,
                                    probe_molecule_conformer_index, reference_molecule_conformer_index)

    def __init__(self, name, smiles, charge, dir_job=None, n_cores=1, max_RAM_per_core_in_MB=None):

        self._n_cores = n_cores

        if dir_job is None:
            self.dir_job = name
        else:
            self.dir_job = dir_job

        self._setup_folder()

        self.name = name
        self.charge = charge 

        self.smiles = smiles
        self.mol = ConformerGenerator.get_embedded_mol(self.smiles)

        if max_RAM_per_core_in_MB is None:
            max_RAM_per_core_in_MB = 2000

            if Chem.AddHs(self.mol).GetNumAtoms() > 20:
                max_RAM_per_core_in_MB = 4000

        self._max_RAM_per_core_in_MB = max_RAM_per_core_in_MB

        charge_from_molecule = Chem.rdmolops.GetFormalCharge(self.mol)

        if self.charge != charge_from_molecule:
            raise ValueError('The charge of following molecule does not agree with the smiles: ' + self.name)

        self.step_names = []
        self.step_folders = []
        self.current_step_number = None

    def __enter__(self):

        self._add_step('init')
        self.save_to_disk()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):

        if exception_type is None:
            self._log_results(last=True)
            self._cleanup()

    def _setup_folder(self):

        if os.path.isdir(self.dir_job):
            while True:
                inp = input(f'Should dir "{self.dir_job}" be removed [y/n] ')
                # inp = 'y'
                if inp.lower() == 'n':
                    exit()
                elif inp.lower() == 'y':
                    shutil.rmtree(self.dir_job)
                    time.sleep(1)
                    break
        os.makedirs(self.dir_job)

    def _add_step(self, step_name):

        if self.step_folders:
            self.current_step_number = int(self.step_folders[-1].split('_')[0]) + 1
        else:
            self.current_step_number = 0

        subdir = f'{self.current_step_number:02d}_{step_name}'

        self._dir_step = os.path.join(self.dir_job, subdir)
        os.mkdir(self._dir_step)

        self.step_names.append(step_name)
        self.step_folders.append(subdir)

        self._log_results()

    def _update_conformer_coordinates(self, conformer_index, atom_positions):
        conf = self.mol.GetConformer(conformer_index)
        for i in range(self.mol.GetNumAtoms()):
            x = atom_positions[i, 0]
            y = atom_positions[i, 1]
            z = atom_positions[i, 2]
            conf.SetAtomPosition(i, Point3D(x,y,z))

    def calculate_rdkit(self, number_of_conformers_generated=None, rms_threshold=0.5, rms_only_heavy_atoms=False):

        if number_of_conformers_generated is None:
            number_of_rotable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(self.mol)

            # heuristic taken from https://doi.org/10.1021/ci2004658
            if number_of_rotable_bonds <= 7:
                number_of_conformers_generated = 50
            if number_of_rotable_bonds >= 8 and number_of_rotable_bonds <= 12:
                number_of_conformers_generated = 200
            if number_of_rotable_bonds >= 13:
                number_of_conformers_generated = 300

        print(f'- settings used: [number_of_conformers_generated: {number_of_conformers_generated}, rms_threshold: {rms_threshold}, rms_only_heavy_atoms: {rms_only_heavy_atoms}]')

        self._add_step('rdkit_generate_conformers')
        self.save_to_disk(output_folder=os.path.join(self._dir_step, 'in'))

        _ = Chem.AllChem.EmbedMultipleConfs(
            self.mol, numConfs=number_of_conformers_generated,
            randomSeed=0xf00d)

        mmff_optimized = Chem.AllChem.MMFFOptimizeMoleculeConfs(
            self.mol, numThreads=4, maxIters=2000)

        energies = np.array([res[1] for res in mmff_optimized if res[0] == 0])
        energies *= kJ_per_kcal*1E3 # J
        rel_probs = (np.exp(-energies/(spcon.R*298.15)) /
                     (np.exp(-(energies/(spcon.R*298.15)).min())))

        cnfs = self.mol.GetConformers()
        n_conf = len(cnfs)
        
        for idx, energy_in_kJdivmol in zip(range(n_conf), energies / 1000):
            self.mol.GetConformer(idx).SetDoubleProp('energy', energy_in_kJdivmol / kJdivmol_per_hartree)
        
        idx_to_keep = np.ones(n_conf, dtype='bool')
        idx_to_keep[rel_probs < 1e-2] = False
        for idx0 in range(n_conf):
            for idx1 in range(idx0+1, n_conf):

                rms = None
                if idx_to_keep[idx0] and idx_to_keep[idx1]:
                    rms = ConformerGenerator.calculate_best_rms(self.mol, self.mol, idx0, idx1, only_heavy_atoms=rms_only_heavy_atoms)

                    if rms < rms_threshold:
                        if energies[idx1] > energies[idx0]:
                            idx_to_keep[idx1] = False
                        else:
                            idx_to_keep[idx0] = False

        self._filter(idx_to_keep)

        conformers = np.array(self.mol.GetConformers())
        energies = np.array([conformer.GetDoubleProp('energy') for conformer in conformers])

        idx_arr = energies.argsort()
        self._sort(new_conformer_indices=idx_arr)

        self.save_to_disk()
        return self.current_step_number

    def calculate_orca(self, method):

        self._add_step(method.step_name)
        self.save_to_disk(output_folder=os.path.join(self._dir_step, 'in'))

        dir_calc = os.path.join(self._dir_step, 'calculate')
        os.mkdir(dir_calc)

        for conformer_index in range(len(self.mol.GetConformers())):
            
            conformer_basename = f'{self.name}_c{conformer_index:03d}'
            dir_struct = os.path.join(os.path.join(dir_calc, conformer_basename))
            self.save_to_disk(conformer_indices_to_output=[conformer_index], output_folder=dir_struct)

            _ = method.execute(os.path.join(dir_struct, f'{conformer_basename}.xyz'), self._dir_step, self.charge, self._n_cores, self._max_RAM_per_core_in_MB)

            orcacosmo_filepath = os.path.join(dir_struct, f'{conformer_basename}.orcacosmo')
            orcacosmo = COSMOParser(orcacosmo_filepath, 'orca')

            self.mol.GetConformer(conformer_index).SetProp('attached_file', orcacosmo_filepath)
            self.mol.GetConformer(conformer_index).SetDoubleProp('energy', orcacosmo.cosmo_info['energy_tot'] / kJdivmol_per_hartree)

            self._update_conformer_coordinates(conformer_index, orcacosmo.cosmo_info['atm_pos'])

        self.save_to_disk()
        return self.current_step_number

    def _sort(self, new_conformer_indices):

        conformers = np.array(self.mol.GetConformers())
        conformers = conformers[new_conformer_indices]
        for conformers_index, conformer in enumerate(conformers):
            conformer.SetId(conformers_index)

        mol2 = copy.deepcopy(self.mol)
        mol2.RemoveAllConformers()

        for conformer in conformers:
            mol2.AddConformer(conformer)

        self.mol = mol2

    def sort_by_energy(self, step_name='sort_energy'):

        self._add_step(step_name)
        self.save_to_disk(output_folder=os.path.join(self._dir_step, 'in'))

        conformers = np.array(self.mol.GetConformers())
        energies = np.array([conformer.GetDoubleProp('energy') * kJdivmol_per_hartree for conformer in conformers])

        idx_arr = energies.argsort()
        self._sort(new_conformer_indices=idx_arr)
        self.save_to_disk()
        return self.current_step_number

    def _filter(self, conformer_indices_to_keep):
        
        cnfs = np.array(self.mol.GetConformers())

        cnfs = cnfs[conformer_indices_to_keep]
        for idx, cnf in enumerate(cnfs):
            cnf.SetId(idx)

        mol2 = copy.deepcopy(self.mol)
        mol2.RemoveAllConformers()

        for cnf in cnfs:
            mol2.AddConformer(cnf)

        self.mol = mol2
    
    def filter_by_function(self, filtering_function, step_name='filter'):

        self._add_step(step_name)
        self.save_to_disk(output_folder=os.path.join(self._dir_step, 'in'))
        self._filter(filtering_function(self.mol.GetConformers()))
        self.save_to_disk()
        return self.current_step_number

    def filter_by_energy_window(self, energy_window_in_kJdivmol):

        def internal_filter(conformers):
            energies = np.array([conf.GetDoubleProp('energy') * kJdivmol_per_hartree for conf in conformers])
            relative_energies = energies - energies.min()
            idx_to_keep = np.argwhere(relative_energies <= energy_window_in_kJdivmol).flatten()
            return idx_to_keep

        self.filter_by_function(internal_filter, step_name='filter_by_energy_window')
        return self.current_step_number

    def filter_by_rms_window(self, rms_threshold, only_heavy_atoms=False):

        def internal_filter(conformers):
            energies = np.array([conf.GetDoubleProp('energy') * kJdivmol_per_hartree for conf in conformers])
            n_conf = len(conformers)

            idx_to_keep = np.ones(n_conf, dtype='bool')

            for idx0 in range(n_conf):
                for idx1 in range(idx0+1, n_conf):

                    rms = None
                    if idx_to_keep[idx0] and idx_to_keep[idx1]:
                        rms = ConformerGenerator.calculate_best_rms(self.mol, self.mol,
                                                        idx0, idx1, only_heavy_atoms=only_heavy_atoms)

                        if rms < rms_threshold:
                            if energies[idx1] > energies[idx0]:
                                idx_to_keep[idx1] = False
                            else:
                                idx_to_keep[idx0] = False
            
            return idx_to_keep

        self.filter_by_function(internal_filter, step_name='filter_by_rms_window')
        return self.current_step_number

    def _log_results(self, last = False):
        # try logging the previous step results
        index_to_log = -2
        if last:
            index_to_log = -1
        try:
            print('- %s, n_confs: %s' % (self.step_names[index_to_log], len(self.mol.GetConformers())))
        except:
            pass

    def save_to_disk(self, conformer_indices_to_output=None, output_folder=None, save_xyz_file=True, save_attached_file=False):
 
        if conformer_indices_to_output is None:
            conformer_indices_to_output = range(len(self.mol.GetConformers()))

        if output_folder is None:
            output_folder = os.path.join(self._dir_step, 'out')

        os.makedirs(output_folder, exist_ok=True)
        for conformer_index in conformer_indices_to_output:
            conformer = self.mol.GetConformer(conformer_index)
            conformer_basename = f'{self.name}_c{conformer_index:03d}'
            if save_xyz_file:
                ConformerGenerator.save_conformer_to_xyz_file(conformer,
                                                    filepath_xyz=os.path.join(output_folder, f'{conformer_basename}.xyz'))

            if save_attached_file:
                if conformer.HasProp('attached_file'):
                    attached_file = conformer.GetProp('attached_file')
                    _, file_extension = os.path.splitext(attached_file)
                    shutil.copy(attached_file,
                                os.path.join(output_folder, f'{conformer_basename}{file_extension}'))

    def copy_output(self, destination_folder, excluded_extensions=[], step_number=-1):
        shutil.copytree(os.path.join(self.dir_job, self.step_folders[step_number], 'out'),
                destination_folder, ignore=shutil.ignore_patterns(*excluded_extensions))

    def _cleanup(self):

        # Zip results
        zipfile_path = os.path.join(self.dir_job, 'calculation_files.zip')
        with zipfile.ZipFile(zipfile_path, 'w', compression=zipfile.ZIP_LZMA) as zipf:
            for step_folder in self.step_folders:
                for root, _, files in os.walk(os.path.join(self.dir_job, step_folder)):
                    for file in files:
                        zipf.write(os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file),
                                                os.path.join(self.dir_job,
                                                                '..')))
        # Delete zipped folders
        for step_folder in self.step_folders:
            shutil.rmtree(os.path.join(self.dir_job, step_folder))

class ORCA(ABC):
    def __init__(self, method_description, step_name, filename_base='geo_opt'):

        self.filepath_inp = ''
        self.charge = None

        self.method = method_description
        self.step_name = step_name
        self.filename_base = filename_base
        self.filename_final_log = 'log_output.dat'
        self.filename_final_xyz = f'{self.filename_base}.xyz'
        self.filename_final_cpcm = None
        if platform.system().lower() == 'windows':
            self._orca_full_path = 'orca'
        else:
            output = spr.run(["whereis", "orca"], capture_output=True)
            self._orca_full_path = output.stdout.decode('utf-8').split()[1].strip()


    def execute(self, filepath_inp, dir_step, charge, n_cores = None, max_RAM_per_core_in_MB=None):

        if n_cores is None:
            n_cores = 1
        else:
            self._n_cores = int(n_cores)

        if max_RAM_per_core_in_MB is None:
            self._max_RAM_per_core_in_MB = 2000
        else:
            self._max_RAM_per_core_in_MB = int(max_RAM_per_core_in_MB)

        self.dir_step = dir_step
        self.dir_old = os.getcwd()
        self.dir_work = os.path.dirname(filepath_inp)

        self.charge = charge
        self.filepath_inp = filepath_inp
        self.filename = os.path.basename(filepath_inp)
        self.structname = os.path.splitext(self.filename)[0]

        os.chdir(self.dir_work)
        self._write_input()
        self._call_orca()
        self._concatenate_output()
        os.chdir(self.dir_old)

    @abstractmethod
    def _write_input(self):
        pass

    def _call_orca(self):
        with open('log_output.dat', 'w') as out:
            spr.run([self._orca_full_path, 'input.dat'], stdout=out, stderr=out)

    def _concatenate_output(self):

        with open(f'{self.structname}.orcacosmo', 'w') as file:

            file.write(f'{self.structname} : {self.method}\n')

            file.write('\n'+'#'*50+'\n')
            file.write('#ENERGY\n')
            line_final_energy = ''
            dipole_moment = None
            with open(self.filename_final_log, 'r') as log_file:
                for line in log_file:
                    re_match = re.match(
                        r'.*FINAL\s+SINGLE\s+POINT\s+ENERGY.+', line)
                    if re_match:
                        line_final_energy = line

                    if line.strip().startswith('x,y,z [Debye]:'):
                        dipole_moment = ' '.join(line.strip().split()[-3:])

            file.write(line_final_energy)
            if dipole_moment:
                file.write('\n'+'#'*50+'\n')
                file.write('#DIPOLE MOMENT (Debye)\n')  
                file.write(f'{dipole_moment}\n')    

            file.write('\n'+'#'*50+'\n')
            file.write('#XYZ_FILE\n')
            with open(self.filename_final_xyz, 'r') as xyz_file:
                for line in xyz_file:
                    file.write(line)
            
            if self.filename_final_cpcm is not None:
                file.write('\n'+'#'*50+'\n')
                file.write('#COSMO\n')
                with open(self.filename_final_cpcm, 'r') as cpcm_file:
                    for line in cpcm_file:
                        file.write(line)

class ORCA_XTB2_ALPB(ORCA):
    def __init__(self, solvent):

        super().__init__(method_description='XTB2_ALPB',
                        step_name='orca_xtb2_alpb')

        self.solvent = solvent

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')

        lines.append('')
        lines.append(f'! XTB2 OPT ALPB({self.solvent})')

        lines.append('')
        lines.append(f'%base "{self.filename_base}"')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} 1 {self.filename} ')
        lines.append('')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_FAST(ORCA):
    def __init__(self, method_description='DFT_FAST', step_name='ORCA_DFT_FAST'):

        super().__init__(method_description=method_description,
                        step_name=step_name)
    
    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        lines.append('')
        lines.append(f'! DFT OPT BP86 def2-TZVP(-f){parallel_string}')

        lines.append('')
        lines.append(f'%base "{self.filename_base}"')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} 1 {self.filename} ')
        lines.append('')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_FINAL(ORCA):
    def __init__(self, method_description='DFT_BP86_def2-TZVP+def2-TZVPD_SP', step_name='ORCA_DFT_final'):

        super().__init__(method_description=method_description, step_name=step_name)

        self.filename_base = 'geo_opt_tzvp'

        self.filename_final_xyz = f'{self.filename_base}.xyz'
        self.filename_final_base = 'single_point_tzvpd'

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        lines.append(f'! DFT OPT BP86 def2-TZVP{parallel_string}')
        lines.append('')

        lines.append(f'%base "{self.filename_base}"')
        lines.append('')

        lines.append(f'* xyzfile {self.charge} 1 {self.filename}')
        lines.append('')

        lines.append('$new_job')
        lines.append('')

        lines.append(f'! def2-TZVPD SP{parallel_string}')
        lines.append('')

        lines.append(f'%base "{self.filename_final_base}"')
        lines.append('')

        lines.append(f'* xyzfile {self.charge} 1 {self.filename_final_xyz}')
        lines.append('')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_CPCM(ORCA):
    def __init__(self, method_description='DFT_CPCM', step_name='ORCA_DFT_CPCM', cpcm_radii=None):

        super().__init__(method_description=method_description,
                        step_name=step_name)

        self.filename_final_cpcm = f'{self.filename_base}.cpcm'

        if cpcm_radii is None:
            cpcm_radii = {
                1:  1.300, # H
                5:  2.048, # B
                6:  2.000, # C
                7:  1.830, # N
                8:  1.720, # O
                9:  1.720, # F
                14: 2.480, # Si
                15: 2.130, # P
                16: 2.160, # S
                17: 2.050, # Cl
                35: 2.160, # Br
                53: 2.320, # I
            }

        self.cpcm_radii = cpcm_radii

    def _get_radii_lines(self):
        lines = []
        for element_number, radius in self.cpcm_radii.items():
            lines.append(f'radius[{str(element_number)}]  {str(radius)}')

        return lines

class ORCA_DFT_CPCM_FAST(ORCA_DFT_CPCM):
    def __init__(self, cpcm_radii=None):

        super().__init__(method_description='DFT_CPCM_BP86_def2-TZVP(-f)',
                        step_name='ORCA_DFT_CPCM_fast',
                        cpcm_radii=cpcm_radii)

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        lines.append('%cpcm')
        lines.extend(self._get_radii_lines())
        lines.append('end')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        lines.append('')
        lines.append(f'! DFT OPT CPCM BP86 def2-TZVP(-f){parallel_string}')

        lines.append('')
        lines.append(f'%base "{self.filename_base}"')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} 1 {self.filename} ')
        lines.append('')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_CPCM_FINAL(ORCA_DFT_CPCM):
    def __init__(self, cpcm_radii=None):

        super().__init__(method_description='DFT_CPCM_BP86_def2-TZVP+def2-TZVPD_SP',
                        step_name='ORCA_DFT_CPCM_final',
                        cpcm_radii=cpcm_radii)

        self.filename_base = 'geo_opt_tzvp'

        self.filename_final_xyz = f'{self.filename_base}.xyz'
        self.filename_final_base = 'single_point_tzvpd'
        self.filename_final_cpcm = f'{self.filename_final_base}.cpcm'

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        lines.append('%cpcm')
        lines.extend(self._get_radii_lines())
        lines.append('end')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        lines.append(f'! DFT OPT CPCM BP86 def2-TZVP{parallel_string}')
        lines.append('')

        lines.append(f'%base "{self.filename_base}"')
        lines.append('')

        lines.append(f'* xyzfile {self.charge} 1 {self.filename}')
        lines.append('')

        lines.append('$new_job')
        lines.append('')

        lines.append(f'! def2-TZVPD SP{parallel_string}')
        lines.append('')

        lines.append(f'%base "{self.filename_final_base}"')
        lines.append('')

        lines.append(f'* xyzfile {self.charge} 1 {self.filename_final_xyz}')
        lines.append('')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--structures_file',
                        help='File including structures as tab separated values like so: name TAB SMILES TAB charge',
                        required=True)

    parser.add_argument('--n_cores',
                        help='Number of cores used for parallelization when possible.',
                        default=1, type=int)

    args = parser.parse_args()

            
    name_smiles_dct = {}
    with open(args.structures_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if len(line) == 3:
                SMILES = line[1]
                name_smiles_dct[line[0]] = (SMILES, int(line[2]))

    for name, (smiles, charge) in name_smiles_dct.items():

        print(f'starting: {name}')

        with ConformerGenerator(name, smiles, charge, n_cores=args.n_cores) as cg:

            n = cg.calculate_rdkit(rms_threshold=1.0)

            method = ORCA_XTB2_ALPB('water')
            cg.calculate_orca(method)

            cg.sort_by_energy()

            cg.filter_by_energy_window(6 * kJ_per_kcal)

            cg.filter_by_rms_window(rms_threshold=1.0)

            method = ORCA_DFT_CPCM_FAST()
            cg.calculate_orca(method)

            cg.filter_by_function(lambda _: [0])

            method = ORCA_DFT_CPCM_FINAL()
            cg.calculate_orca(method)

            cg.save_to_disk(save_xyz_file=False, save_attached_file=True)
            cg.copy_output(os.path.join(cg.dir_job, 'COSMO_TZVPD'), ['*.xyz'])
            

        print(f'finished: {name}')
        print()

    print('\nLa fin')
