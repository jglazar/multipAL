'''
This module contains classes for storing machine learning data.
Each class has visualization methods that use the Plotly Express package. 
Such methods need not be called, so that those without Plotly Express installed may still use multipAL
'''

import pandas as pd
import numpy as np

import os.path

from sklearn.manifold import TSNE

from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from matminer.featurizers.structure import JarvisCFID
from matminer.featurizers.composition import ElectronegativityDiff
from pymatgen.core.composition import Composition
from pymatgen.core.molecular_orbitals import MolecularOrbitals

__author__ = "James T. Glazar, Nathan C. Frey"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "James T. Glazar"
__email__ = "jglazar@seas.upenn.edu"
__status__ = "Development"
__date__ = "Feb 2021"


class Data:
  def __init__(self, ftrs_list, prop_list, main_df_file = '/scratch/vshenoy1/jglazar/multipal/main_df.pkl'):
    '''
    Carries the dataframe used for machine learning.
    Can create a custom subclass generating the features and properties of interest. Simply write a subclass of Data with a custom `featurize` method.
    One such subclass is already created for piezoelectricity and spin-orbit spillage.

    Args:
      ftrs_list (list): list of features for machine learning
      prop_list(list): list of properties to predict/study
      main_df_file (string): location of .pkl file for starting dataframe with all data

    Attributes:
      df (DataFrame): the dataframe itself
      ftrs_list (list): list of features for machine learning
      prop_list (list): list of properties to predict/study
      main_df_file (string): location of .pkl file for starting dataframe with all data
      tsne (DataFrame): TSNE features, if desired
      
     Returns:
      None
    '''
    
    self.ftrs_list = ftrs_list
    self.prop_list = prop_list
    self.main_df_file = main_df_file
    self.get_df(file=main_df_file)
    #self.df = self.df[ ['id'] + self.ftrs_list + self.prop_list ] -- this would interfere with the Pz_tp subclass
    self.tsne = None

  def get_df(self, file=''):
    '''
    Load DataFrame from file, or make/clean up/save DataFrame
    
    Args:
      file (string): file with DataFrame in .pkl format 
    
    Returns:
      None (creates self.df attribute)
    '''
    
    if os.path.isfile( self.main_df_file ):
      # if file is available, just load it
      self.df = pd.read_pickle( self.main_df_file )
      
    else:
      # otherwise, make it and save for later
      from jarvis.db.figshare import data
      from matminer.featurizers.structure import JarvisCFID
      cfid_3d = pd.DataFrame( data('cfid_3d') )
      dft_3d  = pd.DataFrame( data('dft_3d' ) )
      self.df = cfid_3d.merge(dft_3d[['jid', 'atoms']], on='jid')
      jrvs = JarvisCFID()
      labels = jrvs.feature_labels()
      self.df[ labels ] = pd.DataFrame(self.df.desc.tolist(), index= self.df.index).iloc[:,:-1]  # adds an extra column for some reason
      self.df.drop(labels='desc', axis=1, inplace=True)
      
      # clean up basic features
      self.df['id'] = self.df['jid'].str.replace(r'JVASP-', '').apply(int)
      self.df.drop('jid', axis=1, inplace=True)
      self.df = self.df.replace('na', np.nan)
      self.df = self.df[ self.df['jml_ndunfill'] > -5 ]
      self.df.to_pickle(self.main_df_file)

  def filter_df(self, prop='all', inplace=True):
    '''
    Drop any rows with 'None' type in prop column
    
    Args:
      prop (string): property around which to refine DataFrame. If 'all', drop rows with any None
      inplace (bool): edit self.df or output a DataFrame?
      
    Returns:
      None if inplace==True (edits self.df)
      out_df (DataFrame): edited DataFrame
    '''
    
    if prop=='all':
      out_df = self.df.dropna(inplace=inplace)
    else:
      out_df = self.df[ self.df[prop].notna() ]
    
    if inplace:
      self.df = out_df
    else:
      return out_df
  
  def combine_props(self, way='multiply', inplace=True):
    '''
    Create new column in self.df that combines different properties. 
    Used to simultaneously optimize multiple properties.
    
    Args:
      way (string): function to combine many properties into one. 'multiply' or 'add' currently supported.
      inplace (bool): edit self.df or output a DataFrame?
    
    Returns:
      None if inplace==True (edits self.df)
      out_df (DataFrame): edited DataFrame
    '''
    
    if inplace:
      if way == 'multiply':
        self.df[way] = self.df.loc[:, self.prop_list].prod( axis=1, min_count=len(self.prop_list) )
      if way == 'add':
        self.df[way] = self.df.loc[:, self.prop_list].sum( axis=1, min_count=len(self.prop_list) )
        
    else:
      out_df = self.df.copy()
      if way == 'multiply':
        out_df[way] = out_df.loc[:, self.prop_list].prod( axis=1, min_count=len(self.prop_list) )
      if way == 'add':
        out_df[way] = out_df.loc[:, self.prop_list].sum( axis=1, min_count=len(self.prop_list) )
      return out_df

  def plot_compete(self, x, y, size=(400,400)):
    '''
    Plot competition graph between two properties 
    Scatter plot of one property vs. the other for all materials with both properties tabulated
    
    Args:
      x (list-like): property 1
      y (list-like): property 2
      size (tuple): width and height of figure
      
    Returns:
      scatter (Plotly Express scatter): scatterplot of property 2 vs. property 1
    '''
    
    import plotly.express as px
    scatter = px.scatter(self.df, x=x, y=y, hover_data=self.df.columns.to_list(), width=size[0], height=size[1], template='plotly_white')
    return scatter
    
  def plot_compare(self, ft, percentile=0.98, size=(500,300)):
    '''
    Make violin plots for a specific feature, with populations corresponding to high-performers for each property
    
    Args:
      ft (string): feature to compare between high-performers
      percentile (float): percentile cutoff for defining 'high-performers' for each property
      size (tuple): width and height of figure
    
    Returns:
      violin (Plotly Express violin): violin plot of high-performers in each category
    '''
    
    import plotly.express as px
    plot_df = self.df.copy()
    for i, p in enumerate( self.prop_list ):
      plot_df['high_' + p] = pd.qcut( plot_df[p], [0,percentile,1], labels=[0,i+1] )
    
    # only works for 2 props. If 3, then a which_high value of 3 could mean 1+2 or just 3.
    plot_df['which_high'] = plot_df[ ['high_'+p for p in self.prop_list] ].sum(axis=1)
    plot_df = plot_df[ plot_df['which_high']>0 ]
    
    # replace 'which_high' with prop names, and add last option for case where material is high in both (which_high=3, if only two props)
    plot_df['which_high'].replace( {i+1:p for i, p in enumerate(self.prop_list) }, inplace=True )
    
    violin = px.violin(plot_df, y=ft, color='which_high', hover_data=self.df.columns.to_list(), width=size[0], height=size[1], template='plotly_white')
    return violin
    
  def add_tsne(self, t=(30,12,200,1000,1), inplace=True):
    '''
    Add TSNE features to DataFrame.
    Must be used before calling 'self.plot_map()' method.
    
    Args:
      t (tuple): arguments for TSNE. (perplexity, early_exaggeration, learning_rate, n_iter, random_state)
      inplace (bool): edit self.df or output a DataFrame?
      
    Returns:
      None if inplace == True (edits self.df)
      out_df (DataFrame): new DataFrame with 'tsne_0' and 'tsne_1' columns
    '''
    
    # TSNE options are ~5 < perplexity < ~50, early_exaggeration ~ 12, ~10 < learning_rate < ~1000, n_iter >= 250, random_state = postive integer
    tsne = TSNE(n_components=2, perplexity=t[0], early_exaggeration=t[1], learning_rate=t[2], n_iter=t[3], random_state=t[4])
    tsne_results = tsne.fit_transform(self.df[self.ftrs_list])
    self.tsne = t
    
    if inplace:
      self.df['tsne_0'] = tsne_results[:,0]
      self.df['tsne_1'] = tsne_results[:,1]
    else:
      out_df = self.df.copy()
      out_df['tsne_0'] = tsne_results[:,0]
      out_df['tsne_1'] = tsne_results[:,1]
      return out_df

  def plot_map(self, prop, size=(500,300)):
    '''
    Use the 2 TSNE features to map the materials. 
    Need to run 'self.add_tsne()' method before.
    Displays islands of similar materials.
    
    Args:
      prop (string): property to display as color for each point on the map.
      size (tuple): width and height of the figure
      
    Returns:
      map (Plotly Express scatter): map of materials using x = tsne_0, y = tsne_1, color = prop
    '''
    
    import plotly.express as px  
    map = px.scatter(self.df[ self.df[prop].notna() ], x='tsne_0', y='tsne_1', color=prop, color_continuous_scale='OrRd', width=size[0], height=size[1], hover_data=self.ftrs_list + ['formula'])
    return map
    
  def gif_map(self, prop, ids, name='al_tsne.gif'):
    '''
    Animate the TSNE map with a list of material selections. 
    This shows the AL search as it selects each material to material.
    
    Args:
      prop (string): property to display as color for each point on the map.
      ids (list): list of material id numbers selected by the active learning search.
      name (string): name of file to output
      
    Returns:
      None (saves .gif file)
    '''
    
    import plotly.express as px
    
    # animate TSNE map to show each material being selected in order by the active learning search
    Writer = animation.writers['pillow']
    writer = Writer(fps=15, bitrate=1800)
    tsne_df = self.df.copy()
    tsne_df['chosen'] = 1
    tsne_df['y_level'] = pd.qcut( tsne_df[prop], [0.25,0.50,0.75,0.95]) 
    fig = plt.figure(figsize=(10,8))
    
    def animate(n):
      ''' helper function to loop over when creating animation '''
      fig.clear()
      tsne_df.at[ tsne_df.id==ids[n], 'chosen'] = 3  # adjust this number to change size of dot for chosen materials
      sns.scatterplot(
        x = "tsne_0", y = "tsne_1",
        hue = "y_level",
        size = 'chosen',
        sizes = (10,100),
        palette = sns.color_palette("rocket_r", 5),
        data = tsne_df.sort_values('y_level'),
        legend = None,
        alpha = 0.7)
    
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(ids), repeat=False, blit=True)
    ani.save(name, writer=writer)
    
class JarvisPTData(Data):

  def __init__(self, ftrs_list=['avg_mass', 'sym_elem', 'max_en_diff', 'pd_diff_div_val', 'sp_diff_div_val'], 
                     prop_list=['dfpt_piezo_max_eij', 'spillage'],
                     custom_df_file='/scratch/vshenoy1/jglazar/multipal/custom_df.pkl'):
    '''
    Subclass of Data, specifically for exploring piezoelectricity and topology using the JARVIS database.

    Args:
      ftrs_list (list): list of features for machine learning
      prop_list(list): list of properties to predict/study
      custom_df_file (string): location of .pkl file for custom dataframe with all data

    Attributes:
      custom_df_file (string): location of .pkl file for starting dataframe with all data
      df (DataFrame): the dataframe itself
      ftrs_list (list): list of features for machine learning
      prop_list (list): list of properties to predict/study
      main_df_file (string): location of .pkl file for starting dataframe with all data
      tsne (DataFrame): TSNE features, if desired
      
     Returns:
      None
    
    '''

    super().__init__(ftrs_list, prop_list)
    self.custom_df_file = custom_df_file
    self.featurize()
    self.combine_props()

  def featurize(self):
    '''
    Create custom features for piezoelectrivity/spillage analysis
    
    Args:
      None
    
    Returns:
      None (edits self.df)
    '''
    
    cols = ['id', 'formula', 'atoms'] + self.ftrs_list + self.prop_list
    if os.path.isfile( self.custom_df_file ):
      out_df = pd.read_pickle( self.custom_df_file )
    else:
    # create custom features
    # create structural data (takes a while)
      unknown_df = self.df.copy()
      unknown_df['struct_data'] = unknown_df['atoms'].apply( lambda x: Spacegroup3D(Atoms.from_dict(x)).spacegroup_data() )
      unknown_df['formula'] = unknown_df['atoms'].apply(lambda x: Atoms.from_dict(x).composition.reduced_formula)

      # size of highest point group
      sym_dict = {'1':1, 
          '-1':2, '2':2, 'm':2,
          '3':3,
          '4':4, '-4':4, '2/m':4, '222':4, 'mm2':4,
          '-3':6, '6':6, '-6':6, '32':6, '3m':6,
          'mmm':8, '4/m':8, '422':8, '4mm':8, '-42m':8,
          '6/m':12, '23':12, '-3m':12, '622':12, '6mm':12, '-6m2':12,
          '4/mmm':16,
          '6/mmm':24, 'm-3':24, '432':24, '-43m':24, 
          'm-3m':48
          }
      unknown_df['sym_elem'] = unknown_df.struct_data.apply( lambda x: x._dataset['pointgroup'] )
      unknown_df['sym_elem'].replace(sym_dict, inplace=True)

      # space group number
      unknown_df['space_group'] = unknown_df.struct_data.apply( lambda x: x._dataset['number'] )

      # crystal system number
      unknown_df['crystal_system'] = 0
      for ind, bounds in enumerate( [ (1,2), (3,15), (16,74), (75,142), (143,167), (168,194), (195,230) ] ):
        unknown_df['crystal_system'] = np.where( unknown_df['space_group'].between(bounds[0], bounds[1]), ind+1, unknown_df['crystal_system'] )

      # electronegativity difference
      elneg = ElectronegativityDiff()
      comp = unknown_df['formula'].apply( Composition ).apply( Composition.add_charges_from_oxi_state_guesses )
      
      def en(x):
        ''' helper function used to apply to pandas Series '''
        try:
          return elneg.featurize(x)[1]
        except:
          return 0.0  # elneg.featurize throws error if atoms' oxidation states = 0
      
      unknown_df['max_en_diff'] = comp.apply(lambda x: en(x))

      jrvs = JarvisCFID()

      # electron fillings
      def electron_tot(x, orbital):
        ''' helper function used to apply to pandas Series '''
        return sum( [jrvs.el_chem_json[i][orbital] for i in x['elements'] if jrvs.el_chem_json[i][orbital] != 0] )
      
      for orb in [ 'nsvalence', 'npvalence', 'ndvalence', 'nfvalence' ]:
        unknown_df[orb] = unknown_df['atoms'].apply( electron_tot, orbital=orb ).fillna(0)
      
      unknown_df['num_atoms'] = unknown_df['atoms'].apply( lambda x: len( x['elements'] ) ).fillna(0)
      unknown_df['num_val'] = unknown_df['nsvalence'] + unknown_df['npvalence'] + unknown_df['ndvalence'] + unknown_df['nfvalence'] 
      unknown_df['pd_diff_div_atoms'] = (unknown_df['ndvalence'] - unknown_df['npvalence']) / unknown_df['num_atoms']
      unknown_df['sp_diff_div_atoms'] = (unknown_df['npvalence'] - unknown_df['nsvalence']) / unknown_df['num_atoms']
      unknown_df['pd_diff_div_val']   = (unknown_df['ndvalence'] - unknown_df['npvalence']) / unknown_df['num_val']
      unknown_df['sp_diff_div_val']   = (unknown_df['npvalence'] - unknown_df['nsvalence']) / unknown_df['num_val']

      # average atomic mass
      def avg_mass(x):
        ''' helper function used to apply to pandas Series '''
        return np.mean( [ jrvs.el_chem_json[i]['atom_mass'] for i in x['elements'] ] )
      
      unknown_df['avg_mass'] = unknown_df['atoms'].apply( avg_mass ).fillna(0)

    # return only the desired information
      #out_df = out_df.append( unknown_df[ cols ], ignoreindex=True )
      out_df = unknown_df
      out_df.to_pickle(self.custom_df_file)      
    
    self.df = out_df[cols]
   
