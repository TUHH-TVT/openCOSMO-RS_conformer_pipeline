import os
import re
import shutil
import time
import zipfile
import platform
import argparse
import csv
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

import scipy.constants as spcon
import subprocess as spr

from abc import ABC, abstractmethod

from input_parsers import SigmaProfileParser
from input_parsers import kJ_per_kcal
from input_parsers import kJdivmol_per_hartree
from input_parsers import angstrom_per_bohr

class ExecutableFinder:
    def __init__(self, executable_name: str, error_message: str):
        self.executable_name = executable_name
        self.error_message = error_message
        self.executable_path = self._find_executable()

    def _find_executable(self):
        system_name = platform.system().lower()
        if system_name == 'windows':
            search_command = 'where'
            index_to_be_used = 0
        elif system_name == 'linux':
            search_command = 'whereis'
            index_to_be_used = 1
        else:
            raise NotImplementedError(f'For the following OS, getting the full path of the {self.executable_name} executable needs to be programmed: {platform.system()}')

        output = spr.run([search_command, self.executable_name], capture_output=True)
        result = output.stdout.decode('utf-8').split()

        if output.returncode != 0 or len(result) <= index_to_be_used:
            raise FileNotFoundError(self.error_message)

        return result[index_to_be_used].strip()

class ConformerGenerator(object):

    @staticmethod
    def get_mol_from_xyz(xyz_file, charge, use_xyz2mol_vdw=False, vdw_covalent_factor=1.3):

        
        xyz_mol = None
        if use_xyz2mol_vdw != 'force':
            try:
                xyz_mol = Chem.MolFromXYZFile(xyz_file)
                rdDetermineBonds.DetermineBonds(xyz_mol, charge=charge)
            except Exception:
                xyz_mol = None

        if use_xyz2mol_vdw in ['force', 'fallback']:
            # sometimes this does not work, so we allow falling back on the original xy2mol algorithm
            # more details: https://github.com/rdkit/rdkit/issues/8006
            if use_xyz2mol_vdw == 'force' or not xyz_mol:
                conformer, atom_symbols, comment = ConformerGenerator.load_conformer_from_xyz_file(xyz_file)
                rwMol = Chem.RWMol()
                for symbol in atom_symbols:
                    atom = Chem.Atom(symbol)
                    rwMol.AddAtom(atom)  
                rwMol.AddConformer(conformer)

                dMat = Chem.Get3DDistanceMatrix(rwMol)
                pt = Chem.GetPeriodicTable()
                num_atoms = rwMol.GetNumAtoms()
                for i in range(num_atoms):
                    a_i = rwMol.GetAtomWithIdx(i)
                    Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * vdw_covalent_factor
                    for j in range(i + 1, num_atoms):
                        a_j = rwMol.GetAtomWithIdx(j)
                        Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * vdw_covalent_factor
                        if dMat[i, j] <= Rcov_i + Rcov_j:
                            rwMol.AddBond(i, j, Chem.BondType.SINGLE)

                xyz_mol = rwMol.GetMol()
        return xyz_mol

    # should work for ground state properties of neutrals and ions
    @staticmethod
    def calculate_multiplicity(molecule, charge):
        molecule = Chem.AddHs(molecule)
        electrons = 0
        for atom in molecule.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            electrons += atomic_number

        if (electrons - charge) % 2 == 0:
            return 1
        else:
            return 2
        
    @classmethod
    def find_balloon_executable(cls):
        if not hasattr(cls, '_balloon_executable'):
            try:
                balloon_finder = ExecutableFinder('balloon', 'Could not find balloon executable.')
                cls._balloon_executable = balloon_finder.executable_path
            except FileNotFoundError as e:
                print(e)
                cls._balloon_executable = None
        return cls._balloon_executable

    @classmethod
    def EmbedMolecule_with_balloon(cls, mol, number_of_conformers_to_generate=1):

        balloon_executable = cls.find_balloon_executable()
        # generate 3D structure without energy minimization
        number_of_conformers_missing = number_of_conformers_to_generate - mol.GetNumConformers()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                
                temporary_in_MOL_file = os.path.join(temp_dir, 'in.mol')
                temporary_out_SDF_file = os.path.join(temp_dir, 'out.sdf')
                Chem.MolToMolFile(mol, temporary_in_MOL_file)
                command = [balloon_executable, '--nconfs', str(number_of_conformers_missing), '--noGA', '--njobs', '4', '--randomSeed', '42', '--output-format', 'sdf', temporary_in_MOL_file, temporary_out_SDF_file]
                _ = spr.run(command, capture_output=True)
                
                with Chem.SDMolSupplier(temporary_out_SDF_file, removeHs=False, sanitize=False) as suppl:
                    for conf_mol in suppl:
                        conf = Chem.Conformer(conf_mol.GetConformer())
                        conf.SetDoubleProp('energy', conf_mol.GetDoubleProp('energy')) # kcal/mol
                        mol.AddConformer(conf, assignId=True)

            return 0
        except Exception as e:
            return -1
            
    @classmethod
    def get_embedded_mol(cls, smiles, xyz_file=None, charge=0):

        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)    
        
        if mol is None:
            raise ValueError(f'Could not generate mol struct from smiles: {smiles}')

        if xyz_file is None:    
            mol = Chem.AddHs(mol)
            retVal = -1
            try:
                retVal = rdDistGeom.EmbedMolecule(mol, randomSeed=0xf00d)
            except Exception:
                pass
            if retVal < 0:
                try:
                    retVal = rdDistGeom.EmbedMolecule(mol, randomSeed=0xf00d, useRandomCoords=True)
                except Exception:
                    pass

            if retVal < 0:
                retVal = cls.EmbedMolecule_with_balloon(mol)
            if retVal == 0:
                structure_was_optimized_successfully = False
                if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
                    structure_was_optimized_successfully = rdForceFieldHelpers.MMFFOptimizeMolecule(mol) != 0
                    if not structure_was_optimized_successfully:
                        structure_was_optimized_successfully = rdForceFieldHelpers.MMFFOptimizeMolecule(mol, maxIters=10000) != 0
                
                if not structure_was_optimized_successfully:
                    if rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
                        structure_was_optimized_successfully = rdForceFieldHelpers.UFFOptimizeMolecule(mol) != 0
                        if not structure_was_optimized_successfully:
                            rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=10000)
        else:
            try:
                xyz_mol = ConformerGenerator.get_mol_from_xyz(xyz_file, charge)
                mol = AssignBondOrdersFromTemplate(mol, xyz_mol)
            except Exception:
                xyz_mol = ConformerGenerator.get_mol_from_xyz(xyz_file, charge, use_xyz2mol_vdw='force')
                mol = AssignBondOrdersFromTemplate(mol, xyz_mol)

            mol.UpdatePropertyCache(strict=False)
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
            for i_atom in range(n_atoms):
                xyzf.write('{:s}  {:.16f}  {:.16f}  {:.16f}\n'.format(
                    mol.GetAtomWithIdx(i_atom).GetSymbol(), atom_positions[i_atom][0],
                    atom_positions[i_atom][1], atom_positions[i_atom][2]))
                
    @staticmethod
    def load_conformer_from_xyz_file(filepath_xyz):

        with open(filepath_xyz, 'r') as xyzf:
            n_atoms = int(xyzf.readline().strip())
            comment = xyzf.readline().strip()
            atom_symbols = []
            
            conformer = Chem.Conformer(n_atoms)
            for i_atom in range(n_atoms):
                line = xyzf.readline().strip().split()
                atom_symbols.append(line[0].strip())
                conformer.SetAtomPosition(i_atom, Point3D(float(line[1]), float(line[2]), float(line[3])))

        return conformer, atom_symbols, comment

    @staticmethod
    def calculate_best_rms(probe_molecule, reference_molecule, 
                            probe_molecule_conformer_index=0, reference_molecule_conformer_index=0,
                            rms_only_heavy_atoms=True,
                            sanitize=False):

        # make the appropriate copy to not modify input
        if rms_only_heavy_atoms:
            probe_mol = Chem.RemoveHs(probe_molecule, sanitize=sanitize)
            reference_mol = Chem.RemoveHs(reference_molecule, sanitize=sanitize)
        else:
            probe_mol = Chem.Mol(probe_molecule, sanitize=sanitize)
            reference_mol = Chem.Mol(reference_molecule, sanitize=sanitize)

        return rdMolAlign.GetBestRMS(probe_mol, reference_mol,
                                    probe_molecule_conformer_index, reference_molecule_conformer_index)

    @staticmethod
    def guess_needed_ressources(mol):

        number_of_heavy_atoms = sum([1 if atom.GetSymbol() != 'H' else 0 for atom in mol.GetAtoms()])
        number_of_atoms = Chem.AddHs(mol).GetNumAtoms()

        max_RAM_per_core_in_MB = 2000

        if number_of_heavy_atoms > 15:
            max_RAM_per_core_in_MB = 4000

        if number_of_heavy_atoms > 25:
            max_RAM_per_core_in_MB = 8000

        n_cores = 1
        if number_of_atoms <= 8:
            n_cores = 2
        elif number_of_atoms <= 16:
            n_cores = 4
        elif number_of_atoms >= 32:
            n_cores = 8

        return max_RAM_per_core_in_MB, n_cores

    def __init__(self, name, dir_job=None, continue_calculation=False, n_cores=-1, max_RAM_per_core_in_MB=None):

        self._n_cores = n_cores

        self._start_time = time.monotonic()
        self._start_time_step = self._start_time

        if dir_job is None:
            self.dir_job = name
        else:
            self.dir_job = dir_job

        self._continue_calculation = continue_calculation and os.path.exists(self.dir_job)
        self.step_names = []
        self.step_folders = []
        self.current_step_number = None
        self.last_completed_step_number = None
        self._setup_folder()

        self.name = name
        self.charge = None
        self.mol = None
        
        self._max_RAM_per_core_in_MB = max_RAM_per_core_in_MB


    def __enter__(self):
        print(f'start: {self.name}')
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):

        if exception_type is None:
            self._log_results(last=True)
            self._cleanup()            
            print(f'end: {self.name}, time_spent: {"{:.2f}".format(time.monotonic() - self._start_time)} s')

    def setup_initial_structure(self, smiles_or_molecule, xyz_file=None, charge=None, title=None):

        self.smiles = None
        self.mol = None

        try:            
            if isinstance(xyz_file, str):
                xyz_file = xyz_file.strip()
                if not xyz_file:
                    xyz_file = None

            if isinstance(smiles_or_molecule, str):
                smiles_or_molecule = smiles_or_molecule.strip()
                if smiles_or_molecule:
                    self.smiles = smiles_or_molecule
                    self.mol = ConformerGenerator.get_embedded_mol(self.smiles, xyz_file, charge)
            else:
                self.smiles = Chem.MolToSmiles(smiles_or_molecule)
                if len(smiles_or_molecule.GetConformers()) > 0 and not xyz_file:
                    self.mol = smiles_or_molecule
                else:
                    self.mol = ConformerGenerator.get_embedded_mol(self.smiles, xyz_file, charge)

            if xyz_file:
                if not self.smiles:
                    self.mol = ConformerGenerator.get_mol_from_xyz(xyz_file, charge)
                    self.smiles = Chem.MolToSmiles(self.mol)

        except Exception:
            self.smiles = None
            self.mol = None
            raise ValueError(f'The structure could not be setup using the given values, {smiles_or_molecule=}, {xyz_file=}, {charge=}')

        self.charge = Chem.rdmolops.GetFormalCharge(self.mol)
        self.multiplicity = ConformerGenerator.calculate_multiplicity(self.mol, charge)

        if self.charge != charge:
            raise ValueError(f'The charge of following molecule does not agree with the smiles: {self.name}')

        # # canonicalize atom order to always get same order in xyz file as in the molecule idependent of input
        # canonical_order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(self.mol))])))[1]
        # self.mol = Chem.RenumberAtoms(self.mol, canonical_order)

        auto_max_RAM_per_core_in_MB, auto_n_cores = ConformerGenerator.guess_needed_ressources(self.mol)
        if self._max_RAM_per_core_in_MB is None:
            auto_max_RAM_per_core_in_MB = auto_max_RAM_per_core_in_MB
            
        if self._n_cores == -1:
            self._n_cores = auto_n_cores

        should_skip_step_execution = self._add_step('setup_initial_structure',title=title)
        if not should_skip_step_execution:
            self.save_to_disk(save_attached_file=False)
        return self.current_step_number

    def _setup_folder(self):

        if not os.path.isdir(self.dir_job):
            os.makedirs(self.dir_job)
        else:
            if not self._continue_calculation:
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
            else:
                step_dirs = []
                all_step_dirs = []
                for dir_ in os.listdir(self.dir_job):
                    if '_' in dir_:
                        step_number = dir_.split('_')[0]
                        if step_number.isdigit():
                            all_step_dirs.append(dir_)
                            step_dir = os.path.join(self.dir_job, dir_)
                            if os.path.exists(step_dir):
                                step_in_dir = os.path.join(step_dir, 'in')
                                if dir_.endswith('setup_initial_structure'):
                                    step_dirs.append(dir_)
                                else:
                                    if os.path.exists(step_dir):
                                        if len(os.listdir(step_in_dir)) > 0:
                                            step_dirs.append(dir_)

                if step_dirs:
                    last_step_dir = sorted(step_dirs)[-1]
                    self.last_completed_step_number = int(last_step_dir.split('_')[0])
                    all_step_dirs = sorted(all_step_dirs)[1:]
                    for dir_ in all_step_dirs:
                        if dir_ not in step_dirs:
                            if int(dir_.split('_')[0]) > self.last_completed_step_number:
                                shutil.rmtree(os.path.join(self.dir_job, dir_))
                                time.sleep(1)
                            else:
                                raise RuntimeError('Unexpected step was going to be deleted while continuing a previous calculation.')
                            
                    last_step_out_dir = os.path.join(self.dir_job, last_step_dir, 'out')
                    if os.path.exists(last_step_out_dir):
                        shutil.rmtree(last_step_out_dir)
                        time.sleep(1)
                    else:
                        last_step_calculate_dir = os.path.join(self.dir_job, last_step_dir, 'calculate')

                        if os.path.exists(last_step_calculate_dir):
                            last_step_calculate_folders = sorted(os.listdir(last_step_calculate_dir))
                            if last_step_calculate_folders:
                                last_step_calculate_folder = last_step_calculate_folders[-1]
                                shutil.rmtree(os.path.join(self.dir_job, dir_, 'calculate', last_step_calculate_folder))
                                time.sleep(1)
                        else:
                            shutil.rmtree(os.path.join(self.dir_job, last_step_dir))
                            time.sleep(1)

                                # last_step_out_files = sorted(os.listdir(os.path.join(self.dir_job, last_step_dir, 'out')))
                                # if last_step_out_files:
                                #     if len(last_step_calculate_folders) - 1 > len(last_step_out_files):
                                #         last_step_last_out_file = last_step_out_files[-1]
                                #         os.remove(os.path.join(self.dir_job, last_step_dir, 'out', last_step_last_out_file))
                                #         time.sleep(1)

                                #     if len(last_step_calculate_folders) - 1 == len(last_step_out_files) - 1:
                                #         raise RuntimeError('Unexpected number of files after clening up a previous calculation.')

                if not last_step_dir:
                    raise RuntimeError('Could not find last step while continuing a previous calculation.')

    def _add_step(self, step_name, title=None):

        if self.step_folders:
            self.current_step_number = int(self.step_folders[-1].split('_')[0]) + 1
        else:
            self.current_step_number = 0

        should_skip_step_execution = False
        if self._continue_calculation:
            should_skip_step_execution = self.current_step_number < self.last_completed_step_number

        subdir = f'{self.current_step_number:02d}_{step_name}'

        self._dir_step = os.path.join(self.dir_job, subdir)

        self.step_names.append(step_name)
        self.step_folders.append(subdir)

        os.makedirs(self._dir_step, exist_ok=self._continue_calculation)

        self._log_results(title=title,should_skip_step_execution=should_skip_step_execution)

        if should_skip_step_execution:
            self.mol.RemoveAllConformers()
            mol_atom_symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
            for conformer_id, filepath_xyz in enumerate(sorted(os.listdir(os.path.join(self._dir_step, 'out')))):
                conformer, atom_symbols, comment = ConformerGenerator.load_conformer_from_xyz_file(os.path.join(self._dir_step, 'out', filepath_xyz))

                if mol_atom_symbols != atom_symbols:
                    raise ValueError(f'Atom order from the molecule does not agree with the atom order of the conformer.')
                conformer.SetId(conformer_id)
                if comment.startswith('energy:'):
                    energy = float(comment.replace('energy:', ''))
                    conformer.SetDoubleProp('energy', energy)
                else:
                    raise ValueError(f'Unexpected comment "{comment}"')
                
                self.mol.AddConformer(conformer)

        return should_skip_step_execution

    def _update_conformer_coordinates(self, conformer_index, atom_positions):
        conf = self.mol.GetConformer(conformer_index)
        for i in range(self.mol.GetNumAtoms()):
            x = atom_positions[i, 0]
            y = atom_positions[i, 1]
            z = atom_positions[i, 2]
            conf.SetAtomPosition(i, Point3D(x,y,z))

    def calculate_rdkit(self, number_of_conformers_to_generate=None, rms_threshold=0.5, rms_only_heavy_atoms=True, fallback_on_balloon_for_conformer_generation=True):

        if number_of_conformers_to_generate is None:
            temp = self.mol
            try:
                temp = Chem.RemoveHs(self.mol)
            except Exception:
                pass
            try:
                number_of_rotable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(temp)
            except Exception:
                number_of_rotable_bonds = 8 # in case errors happen, use middle heuristic

            # heuristic taken from https://doi.org/10.1021/ci2004658
            if number_of_rotable_bonds <= 7:
                number_of_conformers_to_generate = 50
            if number_of_rotable_bonds >= 8 and number_of_rotable_bonds <= 12:
                number_of_conformers_to_generate = 200
            if number_of_rotable_bonds >= 13:
                number_of_conformers_to_generate = 300

        print(f'- settings used: [number_of_conformers_generated: {number_of_conformers_to_generate}, rms_threshold: {rms_threshold}, rms_only_heavy_atoms: {rms_only_heavy_atoms}]')

        should_skip_step_execution = self._add_step('rdkit_generate_conformers')

        if not should_skip_step_execution:
            self.save_to_disk(output_folder=os.path.join(self._dir_step, 'in'))

            try:
                _ = rdDistGeom.EmbedMultipleConfs(
                    self.mol, numConfs=number_of_conformers_to_generate,
                    randomSeed=0xf00d)
                
                if self.mol.GetNumConformers() != number_of_conformers_to_generate:
                    _ = rdDistGeom.EmbedMultipleConfs(
                        self.mol, numConfs=number_of_conformers_to_generate,
                        randomSeed=0xf00d, useRandomCoords=True)

                if rdForceFieldHelpers.MMFFHasAllMoleculeParams(self.mol):
                    optimized_energies = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                        self.mol, numThreads=4, maxIters=2000)
                else:
                    if rdForceFieldHelpers.UFFHasAllMoleculeParams(self.mol):
                        optimized_energies = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                            self.mol, numThreads=4, maxIters=2000)
                            
                energies = np.array([res[1] for res in optimized_energies])
                did_not_converge = np.array([res[0] for res in optimized_energies])
                            
            except Exception:
                pass

            if self.mol.GetNumConformers() != number_of_conformers_to_generate and fallback_on_balloon_for_conformer_generation:
                _ = self.EmbedMolecule_with_balloon(self.mol, number_of_conformers_to_generate)
                print(f'- rdkit conformer generation failed, balloon (MMF94) was used instead')

                energies = np.array([conf.GetDoubleProp('energy') for conf in self.mol.GetConformers()])
                # I don't know if these always converge, but I have insufficient examples to know
                did_not_converge = np.zeros_like(energies, dtype='bool')


            # for some structures, the atoms are overlapping
            # this filters them out if someone knows a better ways to deal with 
            # this, please tell me
            def has_overlapping_atoms(conformer_idx, threshold = 0.1):
                positions = self.mol.GetConformer(conformer_idx).GetPositions()
                distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
                
                # Check for overlaps (excluding self-distances)
                overlap_mask = (distances < threshold) & (distances > 0)
                return np.any(overlap_mask)

            cnfs = self.mol.GetConformers()
            n_conf = len(cnfs)
            
            idx_to_keep = np.ones(n_conf, dtype='bool')
            energies *= kJ_per_kcal*1E3 # J
            for idx, energy_in_kJdivmol in zip(range(n_conf), energies / 1000):
                self.mol.GetConformer(idx).SetDoubleProp('energy', energy_in_kJdivmol / kJdivmol_per_hartree)
                idx_to_keep[idx] = not has_overlapping_atoms(idx)

            
            idx_to_keep[did_not_converge] = False
            rel_probs = (np.exp(-energies/(spcon.R*298.15)) /
                        (np.exp(-(energies[idx_to_keep]/(spcon.R*298.15)).min())))
            idx_to_keep[rel_probs < 1e-2] = False

            idx_to_keep = self._get_idx_to_keep_by_rms_window(self.mol.GetConformers(), rms_threshold, rms_only_heavy_atoms, idx_to_keep)

            self._filter(idx_to_keep)

            conformers = np.array(self.mol.GetConformers())
            energies = np.array([conformer.GetDoubleProp('energy') for conformer in conformers])

            idx_arr = energies.argsort()
            self._sort(new_conformer_indices=idx_arr)

            self.save_to_disk()
        return self.current_step_number

    def calculate_orca(self, method):

        should_skip_step_execution = self._add_step(method.step_name)
        

        if not should_skip_step_execution:
            self.save_to_disk(output_folder=os.path.join(self._dir_step, 'in'))

            dir_calc = os.path.join(self._dir_step, 'calculate')
            os.makedirs(dir_calc, exist_ok=self._continue_calculation)

            for conformer_index in range(len(self.mol.GetConformers())):
                
                conformer_basename = f'{self.name}_c{conformer_index:03d}'
                dir_struct = os.path.join(os.path.join(dir_calc, conformer_basename))

                if not os.path.isdir(dir_struct):
                    self.save_to_disk(conformer_indices_to_output=[conformer_index], output_folder=dir_struct)

                    _ = method.execute(os.path.join(dir_struct, f'{conformer_basename}.xyz'), self._dir_step, self.charge, self.multiplicity, self._n_cores, self._max_RAM_per_core_in_MB)

                output_filepath = os.path.join(dir_struct, f'{conformer_basename}.orcacosmo')
                spp = SigmaProfileParser(output_filepath, 'orca')

                self.mol.GetConformer(conformer_index).SetProp('attached_file', output_filepath)
                self.mol.GetConformer(conformer_index).SetDoubleProp('energy', spp['energy_tot'] / kJdivmol_per_hartree)

                self._update_conformer_coordinates(conformer_index, spp['atm_pos'])

            self.save_to_disk()
        return self.current_step_number

    def _sort(self, new_conformer_indices):

        conformers = np.array(self.mol.GetConformers())
        conformers = conformers[new_conformer_indices]
        for conformers_index, conformer in enumerate(conformers):
            conformer.SetId(conformers_index)

        mol2 = Chem.Mol(self.mol)
        mol2.RemoveAllConformers()

        for conformer in conformers:
            mol2.AddConformer(conformer)

        self.mol = mol2

    def sort_by_energy(self, step_name='sort_energy'):
        
        should_skip_step_execution = self._add_step(step_name)
        
        if not should_skip_step_execution:
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

        mol2 = Chem.Mol(self.mol)
        mol2.RemoveAllConformers()

        for cnf in cnfs:
            mol2.AddConformer(cnf)

        self.mol = mol2
    
    def filter_by_function(self, filtering_function, step_name='filter'):

        should_skip_step_execution = self._add_step(step_name)
        
        if not should_skip_step_execution:
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

    def _get_idx_to_keep_by_rms_window(self, conformers, rms_threshold, rms_only_heavy_atoms=True, idx_to_keep=None):
        energies = np.array([conf.GetDoubleProp('energy') * kJdivmol_per_hartree for conf in conformers])
        n_conf = len(conformers)

        if idx_to_keep is None:
            idx_to_keep = np.ones(n_conf, dtype='bool')

        for idx0 in range(n_conf):
            for idx1 in range(idx0+1, n_conf):

                rms = None
                if idx_to_keep[idx0] and idx_to_keep[idx1]:
                    rms = ConformerGenerator.calculate_best_rms(self.mol, self.mol,
                                                    idx0, idx1, rms_only_heavy_atoms=rms_only_heavy_atoms)

                    if rms < rms_threshold:
                        if energies[idx1] > energies[idx0]:
                            idx_to_keep[idx1] = False
                        else:
                            idx_to_keep[idx0] = False
        
        return idx_to_keep
    
    def filter_by_rms_window(self, rms_threshold, rms_only_heavy_atoms=True):

        def internal_filter(conformers):
            return self._get_idx_to_keep_by_rms_window(conformers, rms_threshold, rms_only_heavy_atoms=rms_only_heavy_atoms)

        self.filter_by_function(internal_filter, step_name='filter_by_rms_window')
        return self.current_step_number

    def filter_by_maximum_number_of_conformers_to_keep(self, maximum_number_of_conformers_to_keep, sort_conformers_by_energy=True):

        if sort_conformers_by_energy:
            self.sort_by_energy()

        def internal_filter(conformers):
            n_conf = len(conformers)
            idx_to_keep = np.zeros(n_conf, dtype='bool')
            idx_to_keep[0:min(maximum_number_of_conformers_to_keep, n_conf)] = True
            return idx_to_keep

        self.filter_by_function(internal_filter, step_name='filter_by_maximum_number_of_conformers_to_keep')
        return self.current_step_number

    def _log_results(self, last = False, title=None, should_skip_step_execution=False):
        # try logging the previous step results
        index_to_log = -2
        if last:
            index_to_log = -1

        if self.current_step_number == 0:
            if title is not None:
                print(title)
        else:
            last_step_time = time.monotonic() - self._start_time_step
            if should_skip_step_execution:
                print(f'- {self.step_names[index_to_log]}, n_confs: {len(self.mol.GetConformers())}, was loaded from previous calculation')
            else:
                print(f'- {self.step_names[index_to_log]}, n_confs: {len(self.mol.GetConformers())}, time_spent: {"{:.2f}".format(last_step_time)} s')
            
            if title is not None:
                print(title)

            self._start_time_step = time.monotonic()

    def save_to_disk(self, conformer_indices_to_output=None, output_folder=None, save_xyz_file=True, save_attached_file=False, save_last_log_file=False):
 
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
                    
            if save_last_log_file:
                shutil.copy(os.path.join(self._dir_step,'calculate',conformer_basename,'log_output.dat'), os.path.join(output_folder, 'log_output.dat'))
            
    def copy_output(self, destination_folder, excluded_extensions=[], step_number=-1):
        shutil.copytree(os.path.join(self.dir_job, self.step_folders[step_number], 'out'),
                destination_folder, ignore=shutil.ignore_patterns(*excluded_extensions),
                dirs_exist_ok=self._continue_calculation)

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
        self.filename_input = 'input.dat'
        self.filename_final_log = 'log_output.dat'
        self.filename_final_xyz = f'{self.filename_base}.xyz'
        self.filename_final_cpcm = None
        self.filename_final_cpcm_corr = None
        self.mol = None
        self.optimize_kw = 'OPT'

        orca_finder = ExecutableFinder('orca', 'The ORCA installation could not be found. Either it is not installed or its location has not been added to the path environment variable.')
        self._orca_full_path = orca_finder.executable_path

        if platform.system().lower() == 'windows':
            search_command = 'where'
            index_to_be_used = 0
        elif platform.system().lower() == 'linux':
            search_command = 'whereis'
            index_to_be_used = 1
        else:
            raise NotImplementedError(f'For the following OS, getting the full path of the ORCA executable needs to be programmed: {platform.system()}')

        _ = ExecutableFinder('otool_xtb', 'The xtb binary could not be found. Please refer to the readme file of this github repository to check how to install it.')

    def execute(self, filepath_inp, dir_step, charge, multiplicity, n_cores = None, max_RAM_per_core_in_MB=None):

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

        self.multiplicity = multiplicity
        self.charge = charge
        self.filepath_inp = filepath_inp
        self.filename = os.path.basename(filepath_inp)
        self.structname = os.path.splitext(self.filename)[0]

        os.chdir(self.dir_work)

        def run():
            self._write_input()
            self._call_orca()
            self._concatenate_output()

        try:
            run()
        except RuntimeError as e:
            # retry if not converged
            if e.args[0] == 'The calculation did not converge or did not terminate normally.':
                if self.optimize_kw == 'OPT':
                    self.optimize_kw = 'COPT'
                    run()

        os.chdir(self.dir_old)

    @abstractmethod
    def _write_input(self):
        pass

    def _call_orca(self):
        with open(self.filename_final_log, 'w') as out:
            spr.run([self._orca_full_path, self.filename_input], stdout=out, stderr=out)

    def _concatenate_output(self):

        with open(f'{self.structname}.orcacosmo', 'w') as file:

            file.write(f'{self.structname} : {self.method}\n')

            file.write('\n'+'#'*50+'\n')
            file.write('#ENERGY\n')
            line_final_energy = ''
            dipole_moment = None
            with open(self.filename_final_log, 'r') as log_file:
                terminated_normally = False
                did_not_converge = False
                for line in log_file:
                    if '****ORCA TERMINATED NORMALLY****' in line:
                        terminated_normally = True
                    if 'The optimization did not converge but reached' in line:
                        did_not_converge = True
                    re_match = re.match(
                        r'.*FINAL\s+SINGLE\s+POINT\s+ENERGY.+', line)
                    if re_match:
                        line_final_energy = line

                    if line.strip().startswith('x,y,z [Debye]:'):
                        dipole_moment = ' '.join(line.strip().split()[-3:])
                        
                if not terminated_normally or did_not_converge:
                    raise RuntimeError('The calculation did not converge or did not terminate normally.')

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

            if self.filename_final_cpcm_corr is not None:
                if os.path.exists(self.filename_final_cpcm_corr):
                    file.write('\n'+'#'*50+'\n')
                    file.write('#COSMO_corrected\n')
                    with open(self.filename_final_cpcm_corr, 'r') as cpcm_file:
                        for line in cpcm_file:
                            file.write(line)
            if self.mol:
                file.write('\n'+'#'*50+'\n')
                file.write('#ADJACENCY_MATRIX\n')
                adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(self.mol, useBO=True)
                for i_row in range(adjacency_matrix.shape[0]):
                    line = ''.join(['{:4d}'.format(int(v)) for v in adjacency_matrix[i_row, :]]) + '\n'
                    file.write(line)

class ORCA_XTB2_ALPB(ORCA):
    def __init__(self, solvent):

        super().__init__(method_description='XTB2_ALPB',
                        step_name='orca_xtb2_alpb')

        self.solvent = solvent
        self.filename_final_cpcm = None

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')

        lines.append('')
        lines.append(f'! XTB2 {self.optimize_kw} ALPB({self.solvent})')

        lines.append('')
        lines.append(f'%base "{self.filename_base}"')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename} ')
        lines.append('')

        with open(self.filename_input, 'w') as file:
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
        lines.append(f'! DFT {self.optimize_kw} BP86 def2-TZVP(-f){parallel_string}')

        lines.append('')
        lines.append(f'%base "{self.filename_base}"')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename} ')
        lines.append('')

        with open(self.filename_input, 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_FINAL(ORCA):
    def __init__(self, method_description='DFT_BP86_def2-TZVP+def2-TZVPD_SP', step_name='ORCA_DFT_final', do_geometry_optimization=True):

        super().__init__(method_description=method_description, step_name=step_name)

        self.filename_base = 'geo_opt_tzvp'
        self.do_geometry_optimization = do_geometry_optimization

        self.filename_final_xyz = f'{self.filename_base}.xyz'
        self.filename_final_base = 'single_point_tzvpd'

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        if self.do_geometry_optimization:
            lines.append(f'! DFT {self.optimize_kw} BP86 def2-TZVP{parallel_string}')
            lines.append('')

            lines.append(f'%base "{self.filename_base}"')
            lines.append('')

            lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename}')
            lines.append('')

            lines.append('$new_job')
            lines.append('')

        lines.append(f'! def2-TZVPD SP{parallel_string}')
        lines.append('')

        lines.append(f'%base "{self.filename_final_base}"')
        lines.append('')

        if not self.do_geometry_optimization:
            self.filename_final_xyz = self.filename
            
        lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename_final_xyz}')
        lines.append('')

        with open(self.filename_input, 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_CPCM(ORCA):
    def __init__(self, method_description='DFT_CPCM', step_name='ORCA_DFT_CPCM', cpcm_radii=None, cut_area_in_angstrom_squared=None):

        super().__init__(method_description=method_description,
                        step_name=step_name)

        self.filename_final_cpcm = f'{self.filename_base}.cpcm'

        if cpcm_radii is None:
            cpcm_radii = {}

        self.cpcm_radii = cpcm_radii

        self._cut_area_in_bohr_squared = None
        if cut_area_in_angstrom_squared is not None:
            self._cut_area_in_bohr_squared = cut_area_in_angstrom_squared / (angstrom_per_bohr * angstrom_per_bohr)


    def _get_radii_lines(self):
        lines = []
        for element_number, radius in self.cpcm_radii.items():
            lines.append(f'radius[{str(element_number)}]  {str(radius)}')

        return lines

class ORCA_DFT_CPCM_FAST(ORCA_DFT_CPCM):
    def __init__(self, cpcm_radii=None, cut_area_in_angstrom_squared=None):

        super().__init__(method_description='DFT_CPCM_BP86_def2-TZVP(-f)',
                        step_name='ORCA_DFT_CPCM_fast',
                        cpcm_radii=cpcm_radii,
                        cut_area_in_angstrom_squared=cut_area_in_angstrom_squared)

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        lines.append('')
        lines.append(f'%base "{self.filename_base}"')

        lines.append('%cpcm')
        lines.extend(self._get_radii_lines())
        if self._cut_area_in_bohr_squared is not None:
            lines.append(f'cut_area {self._cut_area_in_bohr_squared}')
        lines.append('end')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        lines.append('')
        lines.append(f'! DFT {self.optimize_kw} CPCM BP86 def2-TZVP(-f){parallel_string}')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename} ')
        lines.append('')

        with open(self.filename_input, 'w') as file:
            file.write('\n'.join(lines))

class ORCA_DFT_CPCM_FINAL(ORCA_DFT_CPCM):
    def __init__(self, mol, cpcm_radii=None, cut_area_in_angstrom_squared=None, do_geometry_optimization=True):

        super().__init__(method_description='DFT_CPCM_BP86_def2-TZVP+def2-TZVPD_SP',
                        step_name='ORCA_DFT_CPCM_final',
                        cpcm_radii=cpcm_radii,
                        cut_area_in_angstrom_squared=cut_area_in_angstrom_squared)

        self.filename_base = 'geo_opt_tzvp'
        self.do_geometry_optimization = do_geometry_optimization

        self.filename_final_xyz = f'{self.filename_base}.xyz'
        self.filename_final_base = 'single_point_tzvpd'
        self.filename_final_cpcm = f'{self.filename_final_base}.cpcm'
        self.filename_final_cpcm_corr = f'{self.filename_final_base}.cpcm_corr'
        self.mol = mol

    def _write_input(self):

        lines = []
        lines.append(f'%MaxCore {self._max_RAM_per_core_in_MB}')
        lines.append('')

        if self.do_geometry_optimization:
            lines.append(f'%base "{self.filename_base}"')
            lines.append('')
        else:
            lines.append(f'%base "{self.filename_final_base}"')
            lines.append('')

        lines.append('%cpcm')
        lines.extend(self._get_radii_lines())
        if self._cut_area_in_bohr_squared is not None:
            lines.append(f'cut_area {self._cut_area_in_bohr_squared}')
        lines.append('end')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        if self.do_geometry_optimization:
            lines.append(f'! {self.optimize_kw} CPCM BP86 def2-TZVP{parallel_string}')
            lines.append('')

            lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename}')
            lines.append('')

            lines.append('$new_job')
            lines.append('')

        lines.append(f'! CPCM BP86 def2-TZVPD SP{parallel_string}')
        lines.append('')

        lines.append(f'%base "{self.filename_final_base}"')
        lines.append('')

        if not self.do_geometry_optimization:
            self.filename_final_xyz = self.filename
        lines.append(f'* xyzfile {self.charge} {self.multiplicity} {self.filename_final_xyz}')
        lines.append('')

        with open(self.filename_input, 'w') as file:
            file.write('\n'.join(lines))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--structures_file',
                        help='File including structures as tab separated values like so: name TAB SMILES TAB xyz_file TAB charge',
                        required=True)

    parser.add_argument('--cpcm_radii_file',
                        help='File including the radii used in CPCM calculations as tab separated values like so: atomic number TAB radius',
                        required=True)

    parser.add_argument('--n_cores',
                        help='Number of cores used for parallelization when possible.',
                        default=1, type=int)
    
    parser.add_argument('--continue_calculation',
                        help='Continue a previous calculation.',
                        default=True, type=bool)

    args = parser.parse_args()

    cpcm_radii = {}
    with open(args.cpcm_radii_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if len(line) == 2:
                cpcm_radii[int(line[0])] = float(line[1])
            
    name_smiles_dct = {}
    available_atomic_numbers = set()
    number_of_atoms = []
    with open(args.structures_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if len(line) not in [4, 5]:
                raise ValueError(f'Following line does not contain 4 values seperated by tabs [name, SMILES, xyz_file, charge]: {line}')
            else:
                do_geometry_optimization = True
                if len(line) == 5:
                    name, smiles, xyz_file, charge, do_geometry_optimization = [val.strip() for val in line]
                    if do_geometry_optimization:
                        if do_geometry_optimization.lower() in ['0', 'no', 'false']:
                            do_geometry_optimization = False
                else:
                    name, smiles, xyz_file, charge = [val.strip() for val in line]
                mol = None
                name_smiles_dct[name] = (smiles, xyz_file, int(charge), do_geometry_optimization)
                if len(smiles) > 0:
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    mol.UpdatePropertyCache(strict=False)
                    mol = Chem.AddHs(mol)
                    available_atomic_numbers.update([atom.GetAtomicNum() for atom in mol.GetAtoms()])
                    number_of_atoms.append(mol.GetNumAtoms())

                if mol is None:
                    if len(xyz_file) > 0:
                        with open(xyz_file, 'r') as f:
                            xyz_file_lines = f.readlines()
                            this_number_of_atoms = 0
                            for xyz_file_line in xyz_file_lines[2:]:
                                xyz_file_line = xyz_file_line.split()
                                if len(xyz_file_line) == 4:
                                    atom_symbol = xyz_file_line[0]
                                    available_atomic_numbers.add(Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol))
                                    this_number_of_atoms += 1

                            number_of_atoms.append(this_number_of_atoms)


    if len(cpcm_radii) > 0:
        atomic_numbers_without_radii = available_atomic_numbers - set(cpcm_radii.keys())
        if len(atomic_numbers_without_radii) > 0:
            raise RuntimeError(f'No CPCM radii were specified for the following atomic numbers: {sorted(atomic_numbers_without_radii)}')

    for (name, (smiles, xyz_file, charge, do_geometry_optimization)), n_atoms in zip(name_smiles_dct.items(), number_of_atoms, strict=True):

        with ConformerGenerator(name, continue_calculation=args.continue_calculation, n_cores=args.n_cores) as cg:

            is_single_atom = n_atoms == 1
            is_single_ion = is_single_atom and not charge != 0
            if is_single_atom:
                do_geometry_optimization = False

            # skip gas phase calculation if single ion
            initial_rdkit_molecule_with_conformers = None
            if not is_single_ion:
                # # gas
                n = cg.setup_initial_structure(smiles, xyz_file, charge, title='gas calculation')
               
                if do_geometry_optimization: 
                    cg.calculate_rdkit(rms_threshold=1.0)

                    cg.sort_by_energy()

                    cg.filter_by_energy_window(6 * kJ_per_kcal)

                    cg.filter_by_rms_window(rms_threshold=1.0)
                    initial_rdkit_molecule_with_conformers = Chem.Mol(cg.mol) # create copy for later

                    method = ORCA_DFT_FAST()
                    cg.calculate_orca(method)

                    cg.sort_by_energy()

                    cg.filter_by_function(lambda conformers: [0])

                method = ORCA_DFT_FINAL(do_geometry_optimization=do_geometry_optimization)
                cg.calculate_orca(method)

                cg.copy_output(os.path.join(cg.dir_job, 'energy_TZVPD'))

            # CPCM
            if not initial_rdkit_molecule_with_conformers:
                cg.setup_initial_structure(smiles, xyz_file, charge, title='CPCM calculation')
                if do_geometry_optimization: 
                    cg.calculate_rdkit(rms_threshold=1.0)

                    cg.sort_by_energy()

                    cg.filter_by_energy_window(6 * kJ_per_kcal)

                    cg.filter_by_rms_window(rms_threshold=1.0)
            else:
                cg.setup_initial_structure(initial_rdkit_molecule_with_conformers, xyz_file, charge, title='CPCM calculation')


            # skip steps if single atoms
            if do_geometry_optimization:
                method = ORCA_XTB2_ALPB('water')
                cg.calculate_orca(method)

                cg.sort_by_energy()

                cg.filter_by_energy_window(6 * kJ_per_kcal)

                cg.filter_by_rms_window(rms_threshold=1.0)

                cg.filter_by_maximum_number_of_conformers_to_keep(3)

                method = ORCA_DFT_CPCM_FAST(cpcm_radii=cpcm_radii)
                cg.calculate_orca(method)

                cg.filter_by_energy_window(6 * kJ_per_kcal)

                cg.filter_by_rms_window(rms_threshold=1.0)

                cg.filter_by_maximum_number_of_conformers_to_keep(1)
                
            method = ORCA_DFT_CPCM_FINAL(cg.mol, cpcm_radii=cpcm_radii, cut_area_in_angstrom_squared=0.01, do_geometry_optimization=do_geometry_optimization)
            cg.calculate_orca(method)
            
            cg.save_to_disk(save_xyz_file=False, save_attached_file=True, save_last_log_file=True)
            cg.copy_output(os.path.join(cg.dir_job, 'COSMO_TZVPD'), ['*.xyz'])
            
        print()

    print('\nLa fin')
