'''
This module contains classes for active learning using the Data object.
Each class has visualization methods that use the Plotly Express package. 
Such methods need not be called, so that those without Plotly Express installed may still use multipAL
'''

import pandas as pd
import numpy as np

import warnings

from sklearn.ensemble import RandomForestRegressor
import forestci as fci
from scipy.stats import norm

import multiprocessing
from joblib import Parallel, delayed

import os
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.tasks.vasp.vasp import JobFactory
from jarvis.analysis.topological.spillage import Spillage

__author__ = "James T. Glazar, Nathan C. Frey"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "James T. Glazar"
__email__ = "jglazar@seas.upenn.edu"
__status__ = "Development"
__date__ = "Feb 2021"
     
class AL:
  def __init__(self, data, prop):
    '''
    Modifies the passed Data object and runs active learning
    
    Args:
      data (Data): Data object that holds DataFrame for active learning
      prop (string): Name of property to maximize
    
    Attributes:
      data (Data): Data object that holds DataFrame for active learning
      prop (string): Name of property to maximize

    Returns:
      None
    '''
    
    self.data = data
    self.prop = prop

  def predict(self, al_df):
    '''
    Train a machine learning model and predict property for all materials
    
    Args:
      al_df (DataFrame): DataFrame used to train model and hold predictions. 
                         Must have 'train' column with 0/1 indicator
    
    Returns:
      preds (Series): machine learning predictions of property values for all materials
      uncs (Series): uncertainties in machine learning predictions for all materials
    '''
    
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    
    model = RandomForestRegressor(random_state=0)
    model.fit(al_df[al_df.train==1][ self.data.ftrs_list ], al_df[al_df.train==1][self.prop])
    
    preds = model.predict( al_df[ self.data.ftrs_list ] )
    uncs  = np.sqrt( fci.random_forest_error(model, al_df[al_df.train==1][ self.data.ftrs_list ], al_df[ self.data.ftrs_list ]) )
    
    return preds, uncs

  def acquire(self, pred_df, aq='maxv'):
    '''
    Select the next material to study, given predictions and uncertainties
    
    Args:
      pred_df (DataFrame): DataFrame with machine learning predictions and uncertainties, and material IDs
      aq (string): name of acquisition function to use when selecting the next material.
                   currently supports 'maxu', 'maxv', 'expi', or 'rand'
    
    Returns:
      id (int): ID number of selected material
    '''
    
    if aq == 'maxu':
      id = int( pred_df[pred_df.train==0].nlargest(1, 'unc')['id'] )
    
    if aq == 'maxv':
      id = int( pred_df[pred_df.train==0].nlargest(1, 'pred')['id'] )
    
    if aq == 'rand':
      id = int( pred_df[pred_df.train==0].sample(1, random_state=1)['id'] )
    
    if aq == 'expi':
      pred_df['z'] = (pred_df['pred'] - pred_df[pred_df.train==1][self.prop].max() )/(pred_df['unc'])
      pred_df['expi'] = df['unc'] * ( norm.pdf( pred_df['z'] ) + pred_df['z']*norm.cdf(pred_df['z']) )
      id = int( df[df.train==0].nlargest(1, 'expi')['id'] )
    
    return id
    
  def update(self, id_added, al_df):
    '''
    Performs action after next material is selected. E.g., calculate that material's properties using VASP
    
    Args:
      id_added (int): ID number of next material to investigate and add to training set
      al_df (DataFrame): active learning DataFrame currently in use
      
    Returns:
      out_df (DataFrame): DataFrame with updated information 
    '''

    out_df = al_df
    return out_df
  
  def al( self, df, aq='maxv', n_steps=20):
    '''
    Perform the active learning loop
    
    Args:
      df (DataFrame): DataFrame with starting training sample and featurized unknown materials to investigate
      aq (string): acquisition function to use to select next material to investigate
      n_steps (int): number of iterations for the active learning loop
      
    Returns:
      ids (list): ID numbers of each material selected in order
    '''
    
    # make sure original dataframe isn't edited
    al_df = df.copy()  
    ids = [] 
    
    for i in range(n_steps):
      al_df['pred'], al_df['unc'] = self.predict( al_df ) 
      id_added = self.acquire( al_df, aq=aq )
      al_df = self.update(id_added, al_df)  # update AL dataframe with new info, if needed 
      ids.append( id_added )
      al_df.at[ al_df['id']==id_added, 'train'] = 1  
    
    return ids
  
  def improv(self, al_df, ids):
    '''
    Calculate improvement of active learning loop given starting DataFrame and list of IDs selected.
    Currently under construction.
    
    Args:
      al_df (DataFrame): starting active learning DataFrame with initial training set
      ids (list): ID numbers of materials selected by the active learning loop
    
    Returns:
      improv (list): improvement from 0 (no improvement over starting set) to 1 (found best possible material) to track active learning
    '''
    pass


class JarvisAL( AL ):
  def __init__(self, data, prop):
    '''
    Active learner for JARVIS dataset as proof-of-concept
    
    Args:
      data (Data): Data object that holds DataFrame for active learning
      prop (string): Name of property to maximize
    
    Attributes:
      data (Data): Data object that holds DataFrame for active learning
      prop (string): Name of property to maximize

    Returns:
      None
    '''

    super().__init__(data, prop)

  def df_setup(self, train_size=5, top_res_pct=0.05, seed=0):
    '''
    Generate a DataFrame with test and training subsets
    
    Args:
      train_size (int): number of training materials to use
      top_res_pct (float): top percentile of materials to disqualify from training set
      seed (int): random seed
      
    Returns:
      out_df (DataFrame): Edited self.df DataFrame with 'train' column indicating training set 
    '''
    
    out_df = self.data.filter_df(prop=self.prop, inplace=False)
    
    # reserve top x percent of data set so we don't accidentally start with a great material
    top_res = int( top_res_pct * len(out_df))
    
    out_df['train'] = 0
    out_df.at[ out_df.nsmallest(len(out_df)-top_res, self.prop).sample(n=train_size, random_state=seed).index, 'train'] = 1
    
    return out_df

  def avg_improv( self, aq='maxv', n_avg=10, n_steps=100, train_size=5, top_res_pct=0.05 ):
    '''
    Calculate average improvement metric over different starting training sets for given acquisition function
    
    Args:
      aq (string): name of acquisition function to use when selecting the next material.
                   currently supports 'maxu', 'maxv', 'expi', or 'rand'    
      n_avg (int): number of different starting training sets to average results over
      n_steps (int): number of active learning loop iterations
      train_size (int): number of training materials to use
      top_res_pct (float): top percentile of materials to disqualify from training set
      
    Returns:
      improv_mat (array): improvement at each step for each active learning run. rows are runs, columns are steps.
    '''
    
    improv_mat = np.zeros( (n_avg, n_steps) )
    
    num_cores = multiprocessing.cpu_count()
    jobs = list( range(n_avg) )
    if __name__ == "__main__":
      job_list = Parallel(n_jobs=num_cores)(delayed(self.al)( self.df_setup(train_size=train_size, top_res_pct=top_res_pct, seed=j), aq=aq, n_steps=n_steps) for j in jobs )
    
    for j in jobs:
      al_df = self.df_setup(train_size=train_size, top_res_pct=top_res_pct, seed=j)
      start = al_df[ al_df.train==1 ][self.prop].max()
      best  = al_df[ self.prop ].max()
      ids = job_list[j]
      
      for i in range(len(ids)):
        improv_mat[j][i] = (al_df.loc[al_df['id'].isin(ids[:i+1])][self.prop].max() - start) / (best - start)      
    
    # get rid of "negative improvement," which is nonsensical 
    improv_mat[ improv_mat < 0 ] = 0  
    
    return improv_mat  

  def compare_aq( self, aqs=['maxu', 'maxv', 'rand'], n_avg=10, n_steps=100, train_size=5, top_res_pct=0.05 ):
    '''
    Compare different acquisition functions in terms of average improvement.
    
    Args:
      aqs (list): names of acquisition functions to compare.  
      n_avg (int): number of different starting training sets to average results over
      n_steps (int): number of active learning loop iterations
      train_size (int): number of training materials to use
      top_res_pct (float): top percentile of materials to disqualify from training set
      
    Returns:
      comp_df (DataFrame): mean and standard deviation of improvement at each step for each acquisition function.
    '''
    
    compare_mat = np.zeros( (n_steps, len(aqs) * 2) )
    
    for ind, aq in enumerate(aqs):
      run = self.avg_improv( aq=aq, n_avg=n_avg, n_steps=n_steps )
      compare_mat[:, ind] = np.mean( run, axis=0)
      compare_mat[:, ind+len(aqs)] = np.std( run, axis=0, ddof=1)
    
    comp_df = pd.DataFrame( columns = [i+'_mean' for i in aqs] + [i+'_sd' for i in aqs], data=compare_mat )
    comp_df['step'] = range( 1, n_steps+1 )
    
    return comp_df

  def plot_racetrack(self, comp_df, error_bars=False, size=(600,400)):
    '''
    Plot a 'racetrack' graph to visually compare performance of different acquisition functions
  
    Args:
      comp_df (DataFrame): mean and standard deviation of improvement at each step for different acquisition functions.
      error_bars (bool): include error bands around mean improvement curves?
      size (tuple): width and height of figure
    
    Returns:
      line (Plotly Express line): graph comparing performance of acquisition functions
    '''
    
    import plotly.express as px
    import plotly.graph_objs as go
    
    aqs = [i.split('_')[0] for i in comp_df.columns.to_list() if '_mean' in i]
    
    plot_df = pd.DataFrame( 
        {'step': list(range(1,len(comp_df)+1)) * len(aqs),
         'mean': comp_df[ [i+'_mean' for i in aqs] ].values.flatten(order='F'),
         'sd':   comp_df[ [i+'_sd' for i in aqs] ].values.flatten(order='F'),
         'aq':   [j for i in [ [i]*len(comp_df) for i in aqs ] for j in i]
          }
        )
    
    if error_bars:
      line = go.Figure()
      for x, i in enumerate(aqs): 
        y_upper = (plot_df[plot_df['aq']==i]['mean']+plot_df[plot_df['aq']==i]['sd']).to_list()
        y_lower = (plot_df[plot_df['aq']==i]['mean']-plot_df[plot_df['aq']==i]['sd']).to_list()[::-1]
        x_long = plot_df[plot_df['aq']==i]['step'].to_list() + plot_df[plot_df['aq']==i]['step'].to_list()[::-1]
        line.add_trace( go.Scatter( x=plot_df[plot_df['aq']==i]['step'], y=plot_df[plot_df['aq']==i]['mean'], mode='lines', name=i, line=dict(color=px.colors.qualitative.Plotly[x])) )
        line.add_trace( go.Scatter( x=x_long, y=y_upper + y_lower, fill='toself', opacity=0.2, showlegend=False, fillcolor=px.colors.qualitative.Plotly[x]) )
      line.update_layout( width=size[0], height=size[1], yaxis_range=[0, 1] )
    
    else:
      line = px.line(plot_df, x='step', y='mean', color='aq' , width=size[0], height=size[1], range_y=[0, 1]) 
    
    return line
      
    
class VaspAL( AL ):
  def __init__(self, data, prop):
    '''
    Active learner that conducts search using VASP
    
    Args:
      data (Data): Data object that holds DataFrame for active learning
      prop (string): Name of property to maximize
    
    Attributes:
      data (Data): Data object that holds DataFrame for active learning
      prop (string): Name of property to maximize
      al_df (DataFrame): lone DataFrame used for active learning search

    Returns:
      None
    '''
    
    super().__init__(data, prop)
    self.df_setup()

  def df_setup(self):
    '''
    Sets up active learning DataFrame using all known materials as training set
      
    Args:
      None
    
    Returns:
      None (creates self.al_df)
    '''

    self.al_df = self.data.df.copy()
    self.al_df['train'] = 0
    self.al_df.at[ self.al_df[self.prop].notna(), 'train'] = 1
  
  def update(self, id_added, al_df):
    '''
    Calculates material's property after it is selected.
    Overwrites dummy version in parent class.
    
    Args:
      id_added (int): ID number of next material to investigate and add to training set
      al_df (DataFrame): active learning DataFrame currently in use
      
    Returns:
      out_df (DataFrame): DataFrame with updated information 
    '''
    
    out_df = al_df.copy()
    os.environ['VASP_PSP_DIR'] = '/home/jglazar/potcar'
    atom = Atoms.from_dict( out_df[ out_df['id']==id_added]['atoms'].values[0] )

    if self.prop == 'spillage':
      my_j_fac = JobFactory(
                            poscar = Poscar(atom, comment=str(id_added)),
                            vasp_cmd = 'mpirun vasp_std',
                            optional_params = { 
                           'kppa': 1000,
                           'encut': 500,
                           'kpleng': 20,
                           'line_density': 20,
                           'nbands': 32*2, 
                           'run_wannier':False,
                           'extension':''
                            },
                           steps = ['ENCUT', 'KPLEN', 'RELAX', 'BANDSTRUCT']
                           )
      # run basic calculations / convergences
      my_j_fac.step_flow()

      # run non-collinear VASP for SOC calculations
      my_j_fac.vasp_cmd = 'mpirun vasp_ncl'
      my_soc_job = my_j_fac.soc_spillage(mat=my_j_fac.mat,
                                         encut=my_j_fac.optional_params["encut"],
                                         nbands=None,
                                         kppa=my_j_fac.optional_params["kppa"])[0]
      spl = Spillage(wf_noso='MAIN-MAGSCFBAND-'+str(id_added)+'/WAVECAR', wf_so='MAIN-SOCSCFBAND-'+str(id_added)+'/WAVECAR')
      info = spl.overlap_so_spinpol()
      prop_val = float( info['spillage'] )

    out_df.at[ out_df['id']==id_added, self.prop] = prop_val
    return out_df
