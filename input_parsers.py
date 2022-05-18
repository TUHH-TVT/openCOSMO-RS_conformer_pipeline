#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:31:24 2021

@author: lergos
"""

import os
import re

import numpy as np


bohr_per_angstrom = 0.52917721092
hartree_per_kJdivmol = 2625.499639479
elpotatmu_per_volt = 27.211386245988

class COSMOParser(object):
    def __init__(self, filepath, qc_program):

        self.filepath = filepath
        self.cosmo_info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'qc_program': qc_program,
                'method': '',
                'area': None,
                'volume': None,
                'energy_tot': None,
                'energy_dielectric': None,
                'atm_nr': [],
                'atm_pos': [],
                'atm_elmnt': [],
                'atm_rad': [],
                'seg_nr': [],
                'seg_atm_nr': [],
                'seg_pos': [],
                'seg_charge': [],
                'seg_area': [],
                'seg_sigma_raw': [],
                'seg_potential': []
                }

        if qc_program == 'turbomole':
            self._read_turbomole_cosmo()
        elif qc_program == 'orca':
            self._read_orca_cosmo()
        else:
            raise ValueError(f'Unknown QC file format: {qc_program}')

    def get_cosmo_info(self):

        return self.cosmo_info

    def save_xyz_file(self, filepath_xyz):

        with open(filepath_xyz, 'w') as xyzf:
            xyzf.write(f'{len(self.cosmo_info["atm_nr"])}\n') 
            xyzf.write('\n')
            atom_positions = self.cosmo_info['atm_pos'] / 0.52917721092
            for i in self.cosmo_info['atm_nr']:
                xyzf.write('{:s}  {:.16f}  {:.16f}  {:.16f}\n'.format(
                    self.cosmo_info['atm_elmnt'][i], atom_positions[i][0],
                    atom_positions[i][1], atom_positions[i][2]))


    def _read_single_float(self, line, variable, regex, scaling_factor):
        
        if self.cosmo_info[variable] is None:
            re_match = re.match(regex, line)
            if re_match:
                self.cosmo_info[variable] = float(re_match.groups()[0])
                self.cosmo_info[variable] *= scaling_factor
                
    
    def _read_turbomole_atom_section(self, cosmofile):
        
        line = next(cosmofile).strip()
        
        while line:
            line = next(cosmofile).strip()
            if line == '$coord_car':
                break
            line_splt = line.split()
            
            self.cosmo_info['atm_nr'].append(int(line_splt[0])-1)
            
            self.cosmo_info['atm_pos'].append(
                [float(val) for val in line_splt[1:4]])
            self.cosmo_info['atm_elmnt'].append(line_splt[4].title())
            self.cosmo_info['atm_rad'].append(float(line_splt[5]))
        
        self.cosmo_info['atm_nr'] = np.array(
            self.cosmo_info['atm_nr'], dtype='int64')
        self.cosmo_info['atm_pos'] = (np.array(
            self.cosmo_info['atm_pos'], dtype='float64') *
            bohr_per_angstrom)
        self.cosmo_info['atm_rad'] = np.array(
            self.cosmo_info['atm_rad'], dtype='float64')
        
    
        
    def _read_turbomole_seg_section(self, cosmofile):
        
        for ind in range(10):
            line = next(cosmofile)
            
        while line:
            try:
                line = next(cosmofile).strip()
            except StopIteration:
                break
            if not line:
                break
            line_splt = line.split()
            
            self.cosmo_info['seg_nr'].append(int(line_splt[0])-1)
            self.cosmo_info['seg_atm_nr'].append(int(line_splt[1])-1)
            self.cosmo_info['seg_pos'].append(
                [float(val) for val in line_splt[2:5]])
            self.cosmo_info['seg_charge'].append(float(line_splt[5]))
            self.cosmo_info['seg_area'].append(float(line_splt[6]))
            self.cosmo_info['seg_sigma_raw'].append(float(line_splt[7]))
            #TODO: CLARIFY UNITS
            # self.cosmo_info['seg_potential'].append(float(line_splt[8]))
        

        self.cosmo_info['seg_nr'] = np.array(
            self.cosmo_info['seg_nr'], dtype='int64')
        self.cosmo_info['seg_atm_nr'] = np.array(
            self.cosmo_info['seg_atm_nr'], dtype='int64')
        self.cosmo_info['seg_pos'] = (np.array(
            self.cosmo_info['seg_pos'], dtype='float64') *
            bohr_per_angstrom)
        self.cosmo_info['seg_charge'] = np.array(
            self.cosmo_info['seg_charge'], dtype='float64')
        self.cosmo_info['seg_area'] = np.array(
            self.cosmo_info['seg_area'], dtype='float64')
        self.cosmo_info['seg_sigma_raw'] = np.array(
            self.cosmo_info['seg_sigma_raw'], dtype='float64')
        # self.cosmo_info['seg_potential'] = np.array(
            # self.cosmo_info['seg_potential'], dtype='float64')  
    
    def _read_turbomole_cosmo(self):
        
        with open(self.filepath, 'r') as cosmofile:

            for line in cosmofile:
                
                line = line.strip()

                if line == '$info':
                    line = next(cosmofile).strip()
                    self.cosmo_info['method'] = (
                        f'{line.split(";")[-2]}_{line.split(";")[-1]}'.lower())
                
                self._read_single_float(
                    line, 'area', r'area\s*=\s*([0-9+-.eE]+)',
                    bohr_per_angstrom**2)
                
                self._read_single_float(
                    line, 'volume', r'volume\s*=\s*([0-9+-.eE]+)',
                    bohr_per_angstrom**3)
        
                self._read_single_float(
                    line, 'energy_tot',
                    r'Total\s+energy\s+corrected.*=\s*([0-9+-.eE]+)',
                    hartree_per_kJdivmol)
    
                self._read_single_float(
                    line, 'energy_dielectric',
                    r'Dielectric\s+energy\s+\[a\.u\.\]\s*=\s*([0-9+-.eE]+)',
                    hartree_per_kJdivmol)
                           
                if line == '$coord_rad':
                    self._read_turbomole_atom_section(cosmofile)
                
                if line == '$segment_information':
                    self._read_turbomole_seg_section(cosmofile)


    def _read_orca_atom_coordinates(self, cosmofile):
        
        next(cosmofile)
        line = next(cosmofile).strip()
        
        atm_nr = 0
        while line:
            line = next(cosmofile).strip()
            if not line or '###' in line:
                break
            line_splt = line.split()
            
            self.cosmo_info['atm_nr'].append(atm_nr)
            atm_nr += 1
            
            self.cosmo_info['atm_pos'].append(
                [float(val) for val in line_splt[1:]])
            self.cosmo_info['atm_elmnt'].append(line_splt[0].title())
        
        self.cosmo_info['atm_nr'] = np.array(
            self.cosmo_info['atm_nr'], dtype='int64')
        self.cosmo_info['atm_pos'] = (np.array(
            self.cosmo_info['atm_pos'], dtype='float64'))

        

    def _read_orca_atom_radii(self, cosmofile):
        
        line = next(cosmofile)
        
        while line:
            line = next(cosmofile).strip()
            if not line or '---' in line:
                break
            line_splt = line.split()
            
            self.cosmo_info['atm_rad'].append(line_splt[3])

        self.cosmo_info['atm_rad'] = np.array(
            self.cosmo_info['atm_rad'], dtype='float64')*bohr_per_angstrom 
        

    def _read_orca_seg_section(self, cosmofile):
        
        next(cosmofile)
        line = next(cosmofile)
        
        seg_nr = 0
        while line:
            try:
                line = next(cosmofile).strip()
            except StopIteration:
                break
            if not line:
                break
            line_splt = line.split()
            
            self.cosmo_info['seg_nr'].append(seg_nr)
            seg_nr += 1
            self.cosmo_info['seg_atm_nr'].append(int(line_splt[-1]))
            self.cosmo_info['seg_pos'].append(
                [float(val) for val in line_splt[0:3]])
            self.cosmo_info['seg_charge'].append(float(line_splt[5]))
            self.cosmo_info['seg_area'].append(float(line_splt[3]))
            #TODO: ALIGN UNITS WITH TURBOMOLE
            # self.cosmo_info['seg_potential'].append(float(line_splt[8]))
        

        self.cosmo_info['seg_nr'] = np.array(
            self.cosmo_info['seg_nr'], dtype='int64')
        self.cosmo_info['seg_atm_nr'] = np.array(
            self.cosmo_info['seg_atm_nr'], dtype='int64')
        self.cosmo_info['seg_pos'] = (np.array(
            self.cosmo_info['seg_pos'], dtype='float64') *
            bohr_per_angstrom)
        self.cosmo_info['seg_charge'] = np.array(
            self.cosmo_info['seg_charge'], dtype='float64')
        self.cosmo_info['seg_area'] = np.array(
            self.cosmo_info['seg_area'], dtype='float64')*bohr_per_angstrom**2
        self.cosmo_info['seg_sigma_raw'] = (self.cosmo_info['seg_charge'] /
                                            self.cosmo_info['seg_area'])

  
    def _read_orca_cosmo(self):
        
        with open(self.filepath, 'r') as cosmofile:
            
            line = next(cosmofile).strip()
            self.cosmo_info['name'], self.cosmo_info['method'] = (
                entry.strip() for entry in line.split(':'))

            for line in cosmofile:
                
                line = line.strip()

                self._read_single_float(
                    line, 'area', r'\s*([0-9+-.eE]+)\s+#\s*Area',
                    bohr_per_angstrom**2)
                
                self._read_single_float(
                    line, 'volume', r'\s*([0-9+-.eE]+)\s+#\s*Volume',
                    bohr_per_angstrom**3)
        
                self._read_single_float(
                    line, 'energy_tot',
                    r'FINAL\s+SINGLE\s+POINT\s+ENERGY\s*([0-9+-.eE]+)',
                    hartree_per_kJdivmol)
    
                self._read_single_float(
                    line, 'energy_dielectric',
                    r'\s*([0-9+-.eE]+)\s+#\s*CPCM\s+dielectric\s+energy',
                    hartree_per_kJdivmol)
                           
                if line == '#XYZ_FILE':
                    self._read_orca_atom_coordinates(cosmofile)
                
                if 'CARTESIAN COORDINATES (A.U.)' in line:
                    self._read_orca_atom_radii(cosmofile)
                
                if 'SURFACE POINTS (A.U.)' in line:
                    self._read_orca_seg_section(cosmofile)

             
class PyCrsError(Exception):
    pass




if __name__ == "__main__":
    main()
