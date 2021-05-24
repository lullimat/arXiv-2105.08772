from idpy.LBM.SCThermo import ShanChanEquilibriumCache, ShanChen
from idpy.LBM.LBM import XIStencils, FStencils, NPT, LBMTypes
from idpy.LBM.LBM import ShanChenMultiPhase
from idpy.LBM.LBM import CheckUConvergence, CheckCenterOfMassDeltaPConvergence
from idpy.LBM.LBM import PosFromIndex


from idpy.IdpyCode import GetTenet, GetParamsClean, CheckOCLFP
from idpy.IdpyCode import CUDA_T, OCL_T, idpy_langs_sys
from idpy.IdpyCode.IdpySims import IdpySims

import sympy as sp
import numpy as np
from pathlib import Path
import os, h5py
from functools import reduce
from collections import defaultdict
from scipy import interpolate, optimize

'''
Temporary Thermodynamic Variables
'''
n, eps = sp.symbols('n \varepsilon')    
_eps_f = sp.Rational(10, 31)
TLPsis = [sp.exp(-1/n), 1 - sp.exp(-n), 
          ((eps/n + 1) ** (-1 / eps)).subs(eps, _eps_f)]

TLPsiCodes = {TLPsis[0]: 'exp((NType)(-1./ln))', 
              TLPsis[1]: '1. - exp(-(NType)ln)', 
              TLPsis[2]: ('pow(((NType)ln/(' + str(float(_eps_f)) + ' + (NType)ln)), '
                          + str(float(1/_eps_f)) + ')')}

def AddPosPeriodic(a, b, dim_sizes):
    _swap_add = tuple(map(lambda x, y: x + y, a, b))
    _swap_add = tuple(map(lambda x, y: x + y, _swap_add, dim_sizes))
    _swap_add = tuple(map(lambda x, y: x % y, _swap_add, dim_sizes))
    return _swap_add

def PosNorm2(pos):
    return reduce(lambda x, y: x + y, map(lambda x: x ** 2, pos))

class LatticePressureTensor:
    '''
    Ideally, I would need to pass the whole stencil and select the correct lengths
    '''
    def __init__(self, n_field = None, f_stencil = None, psi_sym = None, G = None):
        if n_field is None:
            raise Exception("Parameter n_field must not be None")
        if f_stencil is None:
            raise Exception("Parameter f_stencil must not be None")
        if psi_sym is None:
            raise Exception("Parameter psi_sym must not be None")
        if G is None:
            raise Exception("Parameter G must not be None")

        self.n_field, self.f_stencil, self.psi_sym, self.G = n_field, f_stencil, psi_sym, G
        self.psi_f = sp.lambdify(n, self.psi_sym)

        '''
        Geometric constants
        '''
        self.dim = len(self.n_field.shape)
        self.dim_sizes = self.n_field.shape
        self.dim_strides = np.array([reduce(lambda x, y: x*y, self.dim_sizes[0:i+1]) 
                                     for i in range(len(self.dim_sizes) - 1)], 
                                    dtype = NPT.C[LBMTypes['SType']])
        self.V = reduce(lambda x, y: x * y, self.dim_sizes)

        '''
        Finding the square-lengths
        '''
        self.l2_list = []
        for _e in self.f_stencil['Es']:
            _norm2 = PosNorm2(_e)
            if _norm2 not in self.l2_list:
                self.l2_list += [_norm2]

        self.l2_list = np.array(self.l2_list, dtype = np.int32)

        '''
        Init the basis vectors dictionary
        '''
        self.e_l2 = {}
        for l2 in self.l2_list:
            self.e_l2[l2] = []
        
        for _e in self.f_stencil['Es']:
            _norm2 = PosNorm2(_e)
            self.e_l2[_norm2] += [_e]

        '''
        Finding the weights
        '''
        self.w_list, _w_i = {}, 0
        _, _swap_idx = np.unique(self.f_stencil['Ws'], return_index = True)
        for l2 in self.l2_list:
            self.w_list[l2] = np.array(self.f_stencil['Ws'])[np.sort(_swap_idx)][_w_i]
            _w_i += 1

        '''
        Index the lattice pressure tensor contrbutions as a function of the suqare lengths
        '''
        self.pt_groups_f = {1: self.PTLen1, 2: self.PTLen2}

    def PTLen1(self, _pos, _l_psi, l2):
        if l2 != 1:
            raise Exception("Parameter l2 must be equal to 1!")
        
        for _ea in self.e_l2[1]:
            #print(_ea)
            '''
            Neighbors
            '''
            _n_ea = tuple(AddPosPeriodic(_pos, np.flip(_ea), self.dim_sizes))
            #print(_pos, np.flip(_ea), _n_ea, self.w_list[1])
            _n_psi = self.psi_field[_n_ea]
            _swap_lpt = self.G * self.w_list[1] * _l_psi * _n_psi / 2
            for i in range(self.dim):
                for j in range(i, self.dim):
                    _lpt_index = (j - i) + self.lpt_dim_strides[i]
                    ##print(j - i, i, self.lpt_dim_strides[i], _lpt_index)
                    self.LPT[(_lpt_index,) +  _pos] += _swap_lpt * _ea[i] * _ea[j]

    def PTLen2(self, _pos, _l_psi, l2):
        if l2 != 2:
            raise Exception("Parameter l2 must be equal to 2!")
        
        for _ea in self.e_l2[2]:
            #print(_ea)
            '''
            Neighbors
            '''
            _n_ea = tuple(AddPosPeriodic(_pos, np.flip(_ea), self.dim_sizes))
            #print(_pos, np.flip(_ea), _n_ea, self.w_list[2])
            _n_psi = self.psi_field[_n_ea]
            _swap_lpt = self.G * self.w_list[2] * _l_psi * _n_psi / 2
            for i in range(self.dim):
                for j in range(i, self.dim):
                    _lpt_index = (j - i) + self.lpt_dim_strides[i]
                    self.LPT[(_lpt_index,) + _pos] += _swap_lpt * _ea[i] * _ea[j]

        
    def GetLPT(self):
        self.psi_field = self.psi_f(self.n_field)
        self.n_lpt = self.dim * (self.dim + 1)//2
        self.lpt_dim_sizes = [self.dim - i for i in range(self.dim - 1)]
        self.lpt_dim_strides = np.array([0] + [reduce(lambda x, y: x + y, self.lpt_dim_sizes[0:i+1]) 
                                         for i in range(len(self.lpt_dim_sizes))],
                                        dtype = np.int32)
        
        self.LPT = np.zeros([self.n_lpt] + list(self.dim_sizes))

        for _pos_i in range(self.V):
            _pos = PosFromIndex(_pos_i, self.dim_strides)
            _l_psi = self.psi_field[_pos]

            for l2 in self.l2_list:
                self.pt_groups_f[l2](_pos, _l_psi, l2)
            #sbreak

        '''
        End of interaction pressure tensor
        Adding ideal contribution on diagonal
        '''

        for i in range(self.dim):
            _lpt_index = self.lpt_dim_strides[i]
            self.LPT[_lpt_index, :, :] +=  self.n_field/3
            
        return self.LPT     

class SurfaceOfTension:
    def __init__(self, n_field = None, f_stencil = None, psi_sym = None, G = None):
        if n_field is None:
            raise Exception("Parameter n_field must not be None")
        if f_stencil is None:
            raise Exception("Parameter f_stencil must not be None")
        if psi_sym is None:
            raise Exception("Parameter psi_sym must not be None")
        if G is None:
            raise Exception("Parameter G must not be None")

        self.n_field, self.f_stencil, self.psi_sym, self.G = \
            n_field, f_stencil, psi_sym, G
        self.dim = len(self.n_field.shape)
        self.dim_center = np.array(list(map(lambda x: x//2, self.n_field.shape)))
        self.dim_sizes = self.n_field.shape
        self.psi_f = sp.lambdify(n, self.psi_sym)

        '''
        Preparing common variables
        '''
        _LPT_class = LatticePressureTensor(self.n_field, self.f_stencil, self.psi_sym, self.G)
        self.LPT = _LPT_class.GetLPT()

        self.r_range = np.arange(self.dim_sizes[2] - self.dim_center[2])

        self.radial_n = self.LPT[0, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]
        self.radial_t = self.LPT[3, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]
        self.radial_profile = self.n_field[self.dim_center[0],
                                           self.dim_center[1], self.dim_center[2]:]
        
    def GetSurfaceTension(self, grains_fine = 2 ** 10, cutoff = 2 ** 7):
        self.r_fine = np.linspace(self.r_range[0], self.r_range[-1], grains_fine)

        self.radial_t_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_t, k = 5, s = 0)
        self.radial_n_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_n, k = 5, s = 0)
        
        '''
        Rowlinson: 4.217
        '''
        def st(R):
            _p_jump = \
                (self.radial_n[0] - (self.radial_n[0] - self.radial_n[-1]) *
                 np.heaviside(self.r_fine - R, 1))

            _swap_spl = \
                interpolate.UnivariateSpline(self.r_fine, 
                                             (self.r_fine ** 2) *
                                             (_p_jump - self.radial_t_spl(self.r_fine)),
                                             k = 5, s = 0)

            return _swap_spl.integral(self.r_fine[0], self.r_fine[-1]) / (R ** 2)

        _swap_st = np.array([st(rr) for rr in self.r_fine[1:]])
        _swap_st_spl = interpolate.UnivariateSpline(self.r_fine[1:], _swap_st, k = 5, s = 0)
        _swap_rs = optimize.newton(_swap_st_spl.derivative(), x0 = 0)
        _swap_smin = _swap_st_spl(_swap_rs)
        
        return {'sigma_4.217': _swap_smin, 'Rs_4.217': _swap_rs,
                'st_spl_4.217': _swap_st_spl, 'r_fine_4.217': self.r_fine[1:]}

class EquimolarRadius:
    def __init__(self, mass = None, n_in_n_out = None, dim_sizes = None):
        if mass is None:
            raise Exception("Paramtere mass must not be None")
        if n_in_n_out is None:
            raise Exception("Parameter n_in_n_out must not be None")
        if dim_sizes is None:
            raise Exception("Parameter dim_sizes must not be None")

        self.mass, self.n_in_n_out, self.dim_sizes = mass, n_in_n_out, dim_sizes
        self.V = reduce(lambda x, y: x * y, self.dim_sizes)

    def GetEquimolarRadius(self):
        _r_swap = \
            ((3 / (4 * np.pi)) * (self.mass - self.n_in_n_out[1] * self.V)
             / (self.n_in_n_out[0] - self.n_in_n_out[1]))
        return {'Re': _r_swap ** (1 / 3)}

class TolmanSimulations:
    def __init__(self, *args, **kwargs):
        self.InitClass(*args, **kwargs)
        
        self.DumpName()
        '''
        Check if dump exists
        '''
        self.is_there_dump = os.path.isfile(self.dump_name)
        if self.is_there_dump:
            self.full_kwargs = {**self.full_kwargs, **{'empty_sim': True}}

        self.mp_sim = ShanChenMultiPhase(**self.full_kwargs)

        '''
        Get/Compute Equilibrium values from/and store in cache
        '''
        _sc_eq_cache = \
            ShanChanEquilibriumCache(stencil = self.params_dict['EqStencil'], 
                                     psi_f = self.params_dict['psi_sym'],
                                     G = self.params_dict['SC_G'], 
                                     c2 = self.params_dict['xi_stencil']['c2'])

        self.eq_params = _sc_eq_cache.GetFromCache()


    def End(self):
        self.mp_sim.End()
        del self.mp_sim
    
    def GetDensityField(self):
        _swap_class_name = self.mp_sim.__class__.__name__        
        if self.is_there_dump:
            _n_swap = \
                self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             _swap_class_name + '/idpy_memory/n')
        else:
            _n_swap = \
                self.mp_sim.sims_idpy_memory['n'].D2H()
            
        _n_swap = _n_swap.reshape(np.flip(self.mp_sim.sims_vars['dim_sizes']))
        return _n_swap

    def GetDensityStrip(self, direction = 0):
        _n_swap = self.GetDensityField()
        _dim_center = self.mp_sim.sims_vars['dim_center']
        '''
        I will need to get a strip that is as thick as the largest forcing vector(y) (x2)
        '''
        _delta = 1
        if len(self.params_dict['dim_sizes']) == 2:
            _n_swap = _n_swap[_dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
            
        if len(self.params_dict['dim_sizes']) == 3:
            _n_swap = _n_swap[_dim_center[2] - _delta:_dim_center[2] + _delta + 1,
                              _dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]            
        return _n_swap

    def GetDataEquimolar(self):
        _swap_class_name = self.mp_sim.__class__.__name__
        if self.is_there_dump:
            _mass_swap = \
                self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             _swap_class_name + '/vars/mass')
            _n_in_n_out = \
                self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             _swap_class_name + '/vars/n_in_n_out')

            _dim_sizes = \
                self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             _swap_class_name + '/vars/dim_sizes')            
        else:
            _mass_swap = \
                self.mp_sim.sims_vars['mass']
            _n_in_n_out = \
                self.mp_sim.sims_vars['n_in_n_out']
            _dim_sizes = \
                self.mp_sim.sims_vars['dim_sizes']
            
        _output = {'mass': _mass_swap, 'n_in_n_out': _n_in_n_out, 'dim_sizes': _dim_sizes}
        return _output

    def GetDataEquimolarIntegral(self):
        return {'n_field': self.GetDensityStrip(),
                'n_in': (self.eq_params['n_l']
                         if self.params_dict['full_flag'] else
                         self.eq_params['n_g']),
                'n_out': (self.eq_params['n_g']
                          if self.params_dict['full_flag'] else
                          self.eq_params['n_l'])}

    
    def GetDataDeltaP(self):
        _swap_class_name = self.mp_sim.__class__.__name__
        if self.is_there_dump:
            _swap_delta_p = \
                self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             _swap_class_name + '/vars/delta_p')
        else:
            _swap_delta_p = self.mp_sim.sims_vars['delta_p']

        _output = {'delta_p': _swap_delta_p[-1]}
        return _output


    def GetDataSurfaceOfTension(self):            
        _output = {'n_field': self.GetDensityStrip(),
                   'f_stencil': self.params_dict['force_stencil'],
                   'psi_sym': self.params_dict['psi_sym'],
                   'G': self.params_dict['SC_G']}
        return _output

            
    def Simulate(self):
        if not self.is_there_dump:            
            '''
            Perform Simulation
            '''
            self.mp_sim.InitRadialInterface(n_g = self.eq_params['n_g'], 
                                            n_l = self.eq_params['n_l'], 
                                            R = self.params_dict['R'],
                                            full_flag = self.params_dict['full_flag'])

            self.mp_sim.MainLoop(range(0, self.params_dict['max_steps'],
                                       self.params_dict['delta_step']), 
                                 convergence_functions = [CheckUConvergence,
                                                          CheckCenterOfMassDeltaPConvergence])

            '''
            Check if bubble/droplet burested
            '''
            if abs(self.mp_sim.sims_vars['delta_p'][-1]) < 1e-9:
                print("The", self.params_dict['type'], "has bursted! Dumping Empty simulation")
                '''
                Writing empty simulation file
                '''
                self.mp_sim.sims_dump_idpy_memory_flag = False
                self.mp_sim.sims_vars['empty'] = 'burst'
                self.mp_sim.DumpSnapshot(file_name = self.dump_name,
                                         custom_types = self.mp_sim.custom_types)
                
                return 'burst'
            elif not self.mp_sim.sims_vars['is_centered_seq'][-1]:
                print("The", self.params_dict['type'], "is not centered! Dumping Empty simulation")
                '''
                Writing empty simulation file
                '''
                self.mp_sim.sims_dump_idpy_memory_flag = False
                self.mp_sim.sims_vars['empty'] = 'center'
                self.mp_sim.DumpSnapshot(file_name = self.dump_name,
                                         custom_types = self.mp_sim.custom_types)

                return 'center'
            else:
                print("Dumping in", self.dump_name)
                self.mp_sim.sims_dump_idpy_memory += ['n']
                self.mp_sim.DumpSnapshot(file_name = self.dump_name,
                                         custom_types = self.mp_sim.custom_types)
                
            return True
        else:
            print("Dump file", self.dump_name, "already exists!")
            _swap_class_name = self.mp_sim.__class__.__name__                
            if self.mp_sim.CheckSnapshotData(file_name = self.dump_name,
                                             full_key =
                                              _swap_class_name + '/vars/empty'):

                _swap_val = \
                    self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                                 full_key =
                                                 _swap_class_name + '/vars/empty')
                _swap_val = np.array(_swap_val, dtype='<U10')
                print("Empty simulation! Value:", _swap_val)
                return _swap_val
            else:
                return False

        
    def DumpName(self):
        _unique_ws = str(np.unique(self.params_dict['f_stencil']['Ws']))        
        self.dump_name = \
            (self.__class__.__name__ + '_'
             + str(self.params_dict['dim_sizes']) + '_'
             + 'R' + str(self.params_dict['R']) + '_'
             + str(self.params_dict['type']) + '_'
             + 'G' + str(self.params_dict['SC_G']) + '_'
             + 'psi_' + str(self.params_dict['psi_sym']) + '_'
             + 'ews_' + _unique_ws)

        self.dump_name = self.dump_name.replace("[","_").replace("]","").replace(" ", "_")
        self.dump_name = self.dump_name.replace("/", "_").replace(".","p").replace(",","")
        self.dump_name = self.dump_name.replace("\n", "").replace("(", "").replace(")", "")
        self.dump_name = self.params_dict['data_dir'] / (self.dump_name + '.hdf5')
        
        print(self.dump_name)

    def InitClass(self, *args, **kwargs):
        self.needed_params = ['psi_sym', 'R', 'type', 'EqStencil',
                              'force_stencil', 'max_steps', 'data_dir']
        self.needed_params_mp = ['dim_sizes', 'xi_stencil', 'f_stencil',
                                 'f_stencil', 'psi_code', 'SC_G',
                                 'tau', 'optimizer_flag', 'e2_val',
                                 'lang', 'cl_kind', 'device']

        
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}
        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = self.needed_params + self.needed_params_mp)

        if 'max_steps' not in self.params_dict:
            self.params_dict['max_steps'] = 2 ** 22

        '''
        Merging the dictionaries for passthrough
        ''' 
        self.params_dict['xi_stencil'] = XIStencils['D3Q19']
        self.params_dict['delta_step'] = 2 ** 11
        if 'data_dir' not in self.params_dict:
            self.params_dict['data_dir'] = Path('data/three-dimensions')

        self.params_dict['psi_code'] = TLPsiCodes[self.params_dict['psi_sym']]
        if 'EqStencil' not in self.params_dict:
            self.params_dict['f_stencil'] = self.params_dict['force_stencil'].PushStencil()
            self.params_dict['EqStencil'] = self.params_dict['force_stencil']
        else:
            self.params_dict['f_stencil'] = self.params_dict['force_stencil']
        
        self.params_dict['full_flag'] = \
            True if self.params_dict['type'] == 'droplet' else False
        
        self.full_kwargs = {**self.kwargs, **self.params_dict}

class FlatAnalysis:
    def __init__(self, n_field = None, f_stencil = None, psi_sym = None, G = None):
        if n_field is None:
            raise Exception("Parameter n_field must not be None")
        if f_stencil is None:
            raise Exception("Parameter f_stencil must not be None")
        if psi_sym is None:
            raise Exception("Parameter psi_sym must not be None")
        if G is None:
            raise Exception("Parameter G must not be None")

        self.n_field, self.f_stencil, self.psi_sym, self.G = \
            n_field, f_stencil, psi_sym, G

        self.dim_sizes = self.n_field.shape
        self.dim_center = np.array(list(map(lambda x: x//2, self.dim_sizes)))
        self.dim = len(self.dim_sizes)

        if self.dim == 2:
            self.n_line = self.n_field[self.dim_center[0], self.dim_center[1]:]
            self.z_range = np.arange(self.dim_sizes[1] - self.dim_center[1])
            
        if self.dim == 3:
            self.n_line = self.n_field[self.dim_center[0], self.dim_center[1],
                                       self.dim_center[2]:]
            self.z_range = np.arange(self.dim_sizes[2] - self.dim_center[2])
            
        self.psi_f = sp.lambdify(n, self.psi_sym)

    def GetFlatSigma(self):
        _LPT_class = LatticePressureTensor(self.n_field, self.f_stencil, self.psi_sym, self.G)
        self.LPT = _LPT_class.GetLPT()

        if self.dim == 2:
            self.p_n = self.LPT[0, self.dim_center[0], self.dim_center[1]:]
            self.p_t = self.LPT[2, self.dim_center[0], self.dim_center[1]:]

        if self.dim == 3:
            self.p_n = self.LPT[0, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]
            self.p_t = self.LPT[3, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]

        self.p_n_minus_t = self.p_n - self.p_t
        self.p_n_minus_t_spl = interpolate.UnivariateSpline(self.z_range,
                                                            self.p_n_minus_t,
                                                            k = 5, s = 0)
        return {'sigma_lattice': self.p_n_minus_t_spl.integral(self.z_range[0],
                                                               self.z_range[-1]),
                'p_n_minus_t_spl': self.p_n_minus_t_spl,
                'p_n': self.p_n, 'p_t': self.p_t}


class TolmanSimulationsFlat:
    def __init__(self, *args, **kwargs):
        self.InitClass(*args, **kwargs)
        
        self.DumpName()
        '''
        Check if dump exists
        '''
        self.is_there_dump = os.path.isfile(self.dump_name)
        if self.is_there_dump:
            self.full_kwargs = {**self.full_kwargs, **{'empty_sim': True}}
        
        self.mp_sim = ShanChenMultiPhase(**self.full_kwargs)

    def End(self):
        self.mp_sim.End()
        del self.mp_sim

    def GetDataFlatExpansion(self):
        _output = {'n_field': self.GetDensityStrip(),
                   'f_stencil': self.params_dict['force_stencil'],
                   'psi_sym': self.params_dict['psi_sym'],
                   'G': self.params_dict['SC_G']}
        return _output

    
    def GetDensityField(self):
        if self.is_there_dump:
            _n_swap = \
                self.mp_sim.ReadSnapshotData(file_name = self.dump_name,
                                             full_key =
                                             self.mp_sim.__class__.__name__ + '/idpy_memory/n')
        else:
            _n_swap = \
                self.mp_sim.sims_idpy_memory['n'].D2H()
            
        _n_swap = _n_swap.reshape(np.flip(self.mp_sim.sims_vars['dim_sizes']))
        return _n_swap

    def GetDensityStrip(self, direction = 0):
        _n_swap = self.GetDensityField()
        _dim_center = self.mp_sim.sims_vars['dim_center']
        '''
        I will need to get a strip that is as thick as the largest forcing vector(y) (x2)
        '''
        _delta = 1
        if len(self.params_dict['dim_sizes']) == 2:
            _n_swap = _n_swap[_dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]
            
        if len(self.params_dict['dim_sizes']) == 3:
            _n_swap = _n_swap[_dim_center[2] - _delta:_dim_center[2] + _delta + 1,
                              _dim_center[1] - _delta:_dim_center[1] + _delta + 1,:]            
        return _n_swap

            
    def Simulate(self):
        if not self.is_there_dump:
            '''
            Get/Compute Equilibrium values from/and store in cache
            '''
            _sc_eq_cache = \
                ShanChanEquilibriumCache(stencil = self.params_dict['EqStencil'], 
                                         psi_f = self.params_dict['psi_sym'],
                                         G = self.params_dict['SC_G'], 
                                         c2 = self.params_dict['xi_stencil']['c2'])

            _eq_params = _sc_eq_cache.GetFromCache()
            
            '''
            Perform Simulation
            '''
            self.mp_sim.InitFlatInterface(n_g = _eq_params['n_g'], 
                                          n_l = _eq_params['n_l'], 
                                          width = self.params_dict['width'],
                                          full_flag = True)

            self.mp_sim.MainLoop(range(0, self.params_dict['max_steps'],
                                       self.params_dict['delta_step']), 
                                 convergence_functions = [CheckUConvergence])

            print("Dumping in", self.dump_name)
            self.mp_sim.sims_dump_idpy_memory += ['n']
            self.mp_sim.DumpSnapshot(file_name = self.dump_name,
                                     custom_types = self.mp_sim.custom_types)
                
            return True
        else:
            print("Dump file", self.dump_name, "already exists!")
            return False

        
    def DumpName(self):
        _unique_ws = str(np.unique(self.params_dict['f_stencil']['Ws']))        
        self.dump_name = \
            (self.__class__.__name__ + '_'
             + str(self.params_dict['dim_sizes']) + '_'
             + 'W' + str(self.params_dict['width']) + '_'
             + 'G' + str(self.params_dict['SC_G']) + '_'
             + 'psi_' + str(self.params_dict['psi_sym']) + '_'
             + 'ews_' + _unique_ws)

        self.dump_name = self.dump_name.replace("[","_").replace("]","").replace(" ", "_")
        self.dump_name = self.dump_name.replace("/", "_").replace(".","p").replace(",","")
        self.dump_name = self.dump_name.replace("\n", "").replace("(", "").replace(")", "")
        self.dump_name = self.params_dict['data_dir'] / (self.dump_name + '.hdf5')
        
        print(self.dump_name)

    def InitClass(self, *args, **kwargs):
        self.needed_params = ['psi_sym', 'width', 'EqStencil',
                              'force_stencil', 'max_steps', 'data_dir']
        self.needed_params_mp = ['dim_sizes', 'xi_stencil', 'f_stencil',
                                 'f_stencil', 'psi_code', 'SC_G',
                                 'tau', 'optimizer_flag', 'e2_val',
                                 'lang', 'cl_kind', 'device']

        
        if not hasattr(self, 'params_dict'):
            self.params_dict = {}
        self.kwargs = GetParamsClean(kwargs, [self.params_dict],
                                     needed_params = self.needed_params + self.needed_params_mp)

        if 'max_steps' not in self.params_dict:
            self.params_dict['max_steps'] = 2 ** 22
        if 'dim_sizes' not in self.params_dict:
            self.params_dict['dim_sizes'] = (100, 5)
        if 'width' not in self.params_dict:
            self.params_dict['width'] = 50

        '''
        Merging the dictionaries for passthrough
        '''
        if len(self.params_dict['dim_sizes']) == 2:
            self.params_dict['xi_stencil'] = XIStencils['D2Q9']
            self.params_dict['delta_step'] = 2 ** 14
            if 'data_dir' not in self.params_dict:
                self.params_dict['data_dir'] = Path('data/two-dimensions')            
            
        if len(self.params_dict['dim_sizes']) == 3:
            self.params_dict['xi_stencil'] = XIStencils['D3Q19']
            self.params_dict['delta_step'] = 2 ** 11
            if 'data_dir' not in self.params_dict:
                self.params_dict['data_dir'] = Path('data/three-dimensions')
            

        self.params_dict['psi_code'] = TLPsiCodes[self.params_dict['psi_sym']]
        if 'EqStencil' not in self.params_dict:
            self.params_dict['f_stencil'] = self.params_dict['force_stencil'].PushStencil()
            self.params_dict['EqStencil'] = self.params_dict['force_stencil']
        else:
            self.params_dict['f_stencil'] = self.params_dict['force_stencil']
        
        self.full_kwargs = {**self.kwargs, **self.params_dict}
