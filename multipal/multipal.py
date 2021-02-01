class Data:
  '''
  This class carries the dataframe used for machine learning.
  You can make a custom subclass concerning the features and properties of interest. Just write a subclass of Data with a custom `featurize` method.
  One is already created for piezoelectricity and band topology

  Attributes:
    df -- the dataframe itself
    ftrs_list -- list of desired features for machine learning
    prop_list -- list of properties to predict/study
    main_df_file -- location of .pkl file for starting dataframe with all data
    tsne -- TSNE features, if desired

  Methods:
    __init__ -- initialize the object
    filter_df -- drop any records with None type
    plot_compete -- show competition between two properties with a scatter plot of one property vs. the other, for materials with both calculated 
    plot_compare -- make violin plots for a specific feature, with populations corresponding to high-performers for each property
    add_tsne -- add 2 TSNE features calculated from the existing features in order to make a map
    plot_map -- use the 2 TSNE features to map the materials. hopefully shows islands of similar materials
    gif_map -- animate the TSNE map with a list of material selections. this shows the AL search as it hops from material to material.
  '''

  def __init__(self, ftrs_list, prop_list, main_df_file = '/home/james/Downloads/piezo_ti/main_df.pkl'):
    # initialize Data object with starter dataframe (from JARVIS), 
    # list of desired features for machine learning, 
    # and the desired properties to predict
    self.ftrs_list = ftrs_list
    self.prop_list = prop_list
    self.main_df_file = main_df_file
    self.get_df()
    #self.df = self.df[ ['id'] + self.ftrs_list + self.prop_list ] -- this would interfere with the Pz_tp subclass
    self.tsne = None

  def get_df(self):
    if os.path.isfile( self.main_df_file ):
      self.df = pd.read_pickle( self.main_df_file )  # if file is available, just load it
    else:  # otherwise, make it and save for later
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
    if prop == 'all':
      self.df.dropna(inplace=inplace)
    elif inplace == True:
      self.df = self.df[ self.df[prop].notna() ]
    else:
      return self.df[ self.df[prop].notna() ]
  
  def combine_props(self, way='multiply', inplace=True):
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
    import plotly.express as px
    return px.scatter(self.df, x=x, y=y, hover_data=self.df.columns.to_list(), width=size[0], height=size[1], template='plotly_white')

  def plot_compare(self, ft, size=(500,300)):
    import plotly.express as px
    plot_df = self.df.copy()
    for i, p in enumerate( self.prop_list ):
      plot_df['high_' + p] = pd.qcut( plot_df[p], [0,0.98,1], labels=[0,i+1] )
    plot_df['which_high'] = plot_df[ ['high_'+p for p in self.prop_list] ].sum(axis=1)  # only works for 2 props. If 3, then a which_high value of 3 could mean 1+2 or just 3.
    plot_df = plot_df[ plot_df['which_high']>0 ]
    # replace 'which_high' with prop names, and add last option for case where material is high in both (which_high=3, if only two props)
    plot_df['which_high'].replace( {i+1:p for i, p in enumerate(self.prop_list) }, inplace=True ) #.update({sum(range(len(self.prop_list)+1)):'all'})
    return px.violin(plot_df, y=ft, color='which_high', hover_data=self.df.columns.to_list(), width=size[0], height=size[1], template='plotly_white')

  def add_tsne(self, t=(30,12,200,1000,1), inplace=True):
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
    import plotly.express as px
    # Plot TSNE 2D map of materials, with color indicating strength of particular feature/property
    # I tried PCA, but it didn't produce nice maps with islands of similar materials
    # Plotting different properties will change the map if there isn't perfect overlap between two properties. E.g., not much population overlap b/t max_eij and spillage, so maps look different  
    return px.scatter(self.df[ self.df[prop].notna() ], x='tsne_0', y='tsne_1', color=prop, color_continuous_scale='OrRd', width=size[0], height=size[1], hover_data=self.ftrs_list + ['formula'])

  def gif_map(self, prop, ids, name='al_tsne.gif'):
    import plotly.express as px
    # animate TSNE map to show each material being selected in order by the active learning search
    Writer = animation.writers['pillow']
    writer = Writer(fps=15, bitrate=1800)
    tsne_df = self.df.copy()
    tsne_df['chosen'] = 1
    tsne_df['y_level'] = pd.qcut( tsne_df[prop], [0.25,0.50,0.75,0.95]) 
    fig = plt.figure(figsize=(10,8))
    def animate(n):
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
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=100, repeat=False, blit=True)
    ani.save(name, writer=writer)
    
class JarvisPTData(Data):
  '''
  Subclass of Data, specifically for exploring piezoelectricity and topology using the JARVIS database
  '''
  from jarvis.core.atoms import Atoms
  from jarvis.analysis.structure.spacegroup import Spacegroup3D
  from matminer.featurizers.structure import JarvisCFID
  from matminer.featurizers.composition import ElectronegativityDiff
  from pymatgen.core.composition import Composition
  from pymatgen.core.molecular_orbitals import MolecularOrbitals

  def __init__(self, ftrs_list=['avg_mass', 'sym_elem', 'max_en_diff', 'pd_diff_div_val', 'sp_diff_div_val'], prop_list=['dfpt_piezo_max_eij', 'spillage']):
    super().__init__(ftrs_list, prop_list)
    self.custom_df_file = '/home/james/Downloads/piezo_ti/custom_df.pkl'
    self.featurize()
    self.combine_props()

  def featurize(self):
    cols = ['id', 'formula'] + self.ftrs_list + self.prop_list
    known_df = pd.read_pickle( self.custom_df_file )
    out_df = known_df[ cols ]
    unknown_df = self.df[ ~self.df['id'].isin( known_df['id'] )  ]
    if len( unknown_df ) > 0:
    # create custom features
    # create structural data (takes a while)
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
        try:
          return elneg.featurize(x)[1]
        except:
          return 0.0  # elneg.featurize throws error if atoms' oxidation states = 0
      unknown_df['max_en_diff'] = comp.apply(lambda x: en(x))

      # electron fillings
      def electron_tot(x, orbital):
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
        return np.mean( [ jrvs.el_chem_json[i]['atom_mass'] for i in x['elements'] ] )
      unknown_df['avg_mass'] = unknown_df['atoms'].apply( avg_mass ).fillna(0)

      # save big dataframe in .pkl file so we don't have to remake it
      #self.df.to_pickle(custom_df_file)

    # return only the desired information
      out_df = out_df.append( unknown_df[ ['id', 'formula'] + self.ftrs_list + self.prop_list ], ignoreindex=True )
    #self.df = self.df[ ['id', 'formula'] + self.ftrs_list + self.prop_list ] 
    self.df = out_df
    
    
class AL:
  '''
  This class modifies the data and runs active learning for a Data object
  Attributes:
    data -- the Data object
    prop -- the property to maximize

  Methods:
    __init__ -- initialize the object
    df_setup -- outputs a dataframe with split test and training subsets 
    predict -- use model to train on training set and predict property values for all materials
    acquire -- choose the next material to investigate 
    al -- runs the active learning loop on the given dataframe for the specified number of steps, using the given acquisition function
    avg_improv -- calculates the average improvement on each step of a run from start to finish, with multiple runs to average over
    compare_aq -- compare the performance of different acquisition functions using average improvement
    plot_racetrack -- plot a "racetrack" graph to show the average improvement of different acquisition functions on the dataset
  '''
  def __init__(self, data, prop):
    self.data = data
    self.prop = prop

  def predict(self, al_df):
    # trains the ML model and predicts property values for all materials, with uncertainties
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    model = RandomForestRegressor(random_state=0)
    model.fit(al_df[al_df.train==1][ self.data.ftrs_list ], al_df[al_df.train==1][self.prop])
    preds = model.predict( al_df[ self.data.ftrs_list ] )
    uncs  = np.sqrt( fci.random_forest_error(model, al_df[al_df.train==1][ self.data.ftrs_list ], al_df[ self.data.ftrs_list ]) )
    return preds, uncs

  def acquire(self, pred_df, aq='maxv'):
    # acquires next material for investigation using an acquisition function
    if aq == 'maxu':
      return int( pred_df[pred_df.train==0].nlargest(1, 'unc')['id'] )
    if aq == 'maxv':
      return int( pred_df[pred_df.train==0].nlargest(1, 'pred')['id'] )
    if aq == 'rand':
      return int( pred_df[pred_df.train==0].sample(1, random_state=1)['id'] )
    if aq == 'expi':
      pred_df['z'] = (pred_df['pred'] - pred_df[pred_df.train==1][self.prop].max() )/(pred_df['unc'])
      pred_df['expi'] = df['unc'] * ( norm.pdf( pred_df['z'] ) + pred_df['z']*norm.cdf(pred_df['z']) )
      return int( df[df.train==0].nlargest(1, 'expi')['id'] )

  def update(self, id_added, al_df):
    # performs action after next material is selected. E.g., calculate that material's properties
    return al_df
  
  def al( self, df, aq='maxv', n_steps=20):
    al_df = df.copy()  # make sure original dataframe isn't edited
    ids = []  # materials selected 
    for i in range(n_steps):
      al_df['pred'], al_df['unc'] = self.predict( al_df ) 
      id_added = self.acquire( al_df, aq=aq )
      al_df = self.update(id_added, al_df)  # update AL dataframe with new info, if needed 
      ids.append( id_added )
      al_df.at[ al_df['id']==id_added, 'train'] = 1  
    return ids
  
  def improv(self, al_df, ids):
    pass


class JarvisAL( AL ):
  '''
  Active learner for JARVIS dataset as proof-of-concept

  Attributes:
    data -- the Data object
    prop -- the property to maximize

  Methods:
    __init__ -- initialize the object
    df_setup -- outputs a dataframe with split test and training subsets 
    predict -- use model to train on training set and predict property values for all materials
    acquire -- choose the next material to investigate 
    al -- runs the active learning loop on the given dataframe for the specified number of steps, using the given acquisition function
    avg_improv -- calculates the average improvement on each step of a run from start to finish, with multiple runs to average over
    compare_aq -- compare the performance of different acquisition functions using average improvement
    plot_racetrack -- plot a "racetrack" graph to show the average improvement of different acquisition functions on the dataset
  '''

  def __init__(self, data, prop):
    super().__init__(data, prop)

  def df_setup(self, train_size=5, top_res_pct=0.05, seed=0):
    # set up an AL dataframe with a random starting sample, reserving the top_res_pct materials for the test set
    out_df = self.data.filter_df(prop=self.prop, inplace=False)
    top_res = int( top_res_pct * len(out_df))  # reserve top x percent of data set so we don't accidentally start with a great material
    out_df['train'] = 0
    out_df.at[ out_df.nsmallest(len(out_df)-top_res, self.prop).sample(n=train_size, random_state=seed).index, 'train'] = 1
    return out_df

  def avg_improv( self, aq='maxv', n_avg=10, n_steps=100, train_size=5, top_res_pct=0.05 ):
    # calculate average improvement metric over n_avg different starting training sets
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
    improv_mat[ improv_mat < 0 ] = 0   # get rid of "negative improvement," which is nonsensical 
    return improv_mat  # rows are n_avg runs, columns are step numbers 

  def compare_aq( self, aqs=['maxu', 'maxv', 'rand'], n_avg=10, n_steps=100, train_size=5, top_res_pct=0.05 ):
    # compare the average improvements of different acquisition functions
    compare_mat = np.zeros( (n_steps, len(aqs) * 2) )
    for ind, aq in enumerate(aqs):
      run = self.avg_improv( aq=aq, n_avg=n_avg, n_steps=n_steps )
      compare_mat[:, ind] = np.mean( run, axis=0)
      compare_mat[:, ind+len(aqs)] = np.std( run, axis=0, ddof=1)
    comp_df = pd.DataFrame( columns = [i+'_mean' for i in aqs] + [i+'_sd' for i in aqs], data=compare_mat )
    comp_df['step'] = range( 1, n_steps+1 )
    return comp_df

  def plot_racetrack(self, comp_df, error_bars=False):
    # plot a "racetrack" graph that shows the results of compare_aq
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
      fig = go.Figure()
      for x, i in enumerate(aqs): 
        y_upper = (plot_df[plot_df['aq']==i]['mean']+plot_df[plot_df['aq']==i]['sd']).to_list()
        y_lower = (plot_df[plot_df['aq']==i]['mean']-plot_df[plot_df['aq']==i]['sd']).to_list()[::-1]
        x_long = plot_df[plot_df['aq']==i]['step'].to_list() + plot_df[plot_df['aq']==i]['step'].to_list()[::-1]
        fig.add_trace( go.Scatter( x=plot_df[plot_df['aq']==i]['step'], y=plot_df[plot_df['aq']==i]['mean'], mode='lines', name=i, line=dict(color=px.colors.qualitative.Plotly[x])) )
        fig.add_trace( go.Scatter( x=x_long, y=y_upper + y_lower, fill='toself', opacity=0.2, showlegend=False, fillcolor=px.colors.qualitative.Plotly[x]) )
      fig.update_layout( width=800, height=400, yaxis_range=[0, 1] )
      return fig
    else:
      return px.line(plot_df, x='step', y='mean', color='aq' , width=800, height=400, range_y=[0, 1])
      
    
class VaspAL( AL ):
  def __init__(self, data, prop):
    super().__init__(data, prop)
    self.df_setup()

  def df_setup(self):
    # set up an AL dataframe with all known materials as training set
    self.al_df = self.data.df.copy()
    self.al_df['train'] = 0
    self.al_df.at[ self.al_df[self.prop].notna(), 'train'] = 1
  
  def update(self, id_added, al_df):
    # run appropriate calculation(s) on selected material
    al_df.at[ al_df['id']==id_added, self.prop] = al_df[ al_df['id']==id_added ]['id'] / 10000.0
    return al_df
    
    
   
