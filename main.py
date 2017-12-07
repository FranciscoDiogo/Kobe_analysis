'''
Created on 24/11/2017

@author: francisco
'''
import numpy as np # linear algebra
import pandas as pd
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
import matplotlib as mpl
import math
import plots_field as pf

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn import mixture
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', 50)






def prob_independence(df, var_compar, var_ref):
  ct = pd.crosstab(df[var_compar], df[var_ref])
  chi2, p_val, ndf, exct = sp_stats.chi2_contingency(ct)
  if ndf==0:
    return 0
  else:
    return chi2/ndf

def color_scatter_plot(df_input, var):
  gs = df_input.groupby(var)
  cs = cm.rainbow(np.linspace(0, 1, len(gs)))
  for g, c in zip(gs, cs):
    plt.scatter(g[1].loc_x, g[1].loc_y, color=c)

def rewrite_shot_type(df_input):
  df_shot = df_input.copy()

  #make set of unique words
  class_shot = set()
  for i in range(0, len(df_shot.index)):
    stri = df_shot.at[i,'action_type'].split(' ')
    class_shot.update(stri)
  class_shot = list(class_shot)

  #initialize columns and fill them
  for i in range(0, len(class_shot)):
    df_shot["has_"+str(class_shot[i]).capitalize()] = 0    
  for i in range(0, len(df_shot.index)):
    stri = df_shot.at[i,'action_type'].split(' ')
    for ist in range(0, len(stri)):
        df_shot.at[i,"has_"+stri[ist].capitalize()] = 1
        
  for i in df_shot.columns.values:
    if i[:4]!="has_" and i!="shot_made_flag":
      df_shot = df_shot.drop(i, 1)
    
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1,2,1)
  plt.title("Fraction of shots per label")
  df_shot_order = df_shot.mean().sort_values()
  df_shot_order.plot(kind='bar')
  mean_shot = []
  for i in df_shot_order.index:
    mean_shot.append(df_shot["shot_made_flag"][df_shot[i]==1].mean())

  ax = plt.subplot(1,2,2)
  list(range(len(df_shot_order.index)))
  ax.bar(range(len(df_shot_order.index)), mean_shot, color='b')
  plt.xticks(range(len(df_shot_order.index)), df_shot_order.index)
  plt.xticks(rotation='vertical')
  plt.title("Efficiency of shots per label")
  plt.show()

  make_pretty_corr_plot(df_shot)
  
  df_shot["has_Pullup"] = df_shot["has_Pullup"] + df_shot["has_Pull-up"]
  df_shot = df_shot.drop(["has_Shot", "has_Pull-up"], 1)
  shots_pre = ["Alley", "Finger", "Step", "Follow"]
  shots_pos = ["Oop", "Roll", "Back", "Up"]
  for i in range(0, len(shots_pre)):
    df_shot["has_"+shots_pre[i]+"-"+shots_pos[i]] = df_shot["has_"+shots_pre[i]]
    df_shot = df_shot.drop(["has_"+shots_pre[i], "has_"+shots_pos[i]], 1)

  print(df_shot["has_Dunk"][df_shot["has_Slam"]==1].mean())
  df_shot["has_Slam-Dunk"] = df_shot["has_Slam"]
  df_shot["has_Other-Dunk"] = df_shot["has_Dunk"][df_shot["has_Slam"]==0]
  df_shot["has_Other-Dunk"] = df_shot["has_Other-Dunk"].fillna(0)
  df_shot = df_shot.drop(["has_Dunk", "has_Slam"],1)

  make_pretty_corr_plot(df_shot)
  
  return df_shot
  
  

def make_pretty_corr_plot(df):
    corr = df.corr()
    corr_mat = np.triu(corr, k=1)
    mask =np.zeros_like(corr, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    cmap = sb.diverging_palette(220, 10, as_cmap=True)
    f, ax = plt.subplots(figsize=(11, 9))
    sb.heatmap(corr_mat, mask=mask, cmap=cmap, xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)
    plt.show()
    
def calc_shots_in_a_row(df_input):  
    
  df_input["made_in_a_row"]=np.nan
  df_input["made_last"]=np.nan
  for i in range(2, len(df_input.index)):
    compar = df_input.at[i-1, "shot_made_flag"]
    counter = 1
        
    if np.isnan(compar):
      continue
        
    keep_loop=True
    while keep_loop==True:
      prev_val = df_input.at[i-counter-1, "shot_made_flag"]
      if np.isnan(prev_val) or prev_val!=compar:
        keep_loop=False
      else:
        counter=counter+1
    df_input.at[i, "made_last"] = (2*compar-1)*min(counter, 1) 
    df_input.at[i, "made_in_a_row"] = (2*compar-1)*min(counter, 5) 
    
  fig = plt.figure(figsize=(12, 6))
  fig.add_subplot(1,2,1)
  sb.barplot("made_in_a_row", "shot_made_flag", data=df_input)
  plt.title("Efficiency vs previous shots success")
  plt.ylim(0.3, 0.6)
  fig.add_subplot(1,2,2)
  sb.barplot("made_last", "shot_made_flag", data=df_input)
  plt.title("Efficiency vs previous shot success")
  plt.ylim(0.35, 0.5)
  plt.show()

  df_input = df_input.drop(["made_last", "made_in_a_row"], 1)
  
  return df_input

def make_predictions(df_input):
  unknown_mask = df_input['shot_made_flag'].isnull()
  data_cl = df_input.copy()
  target = data_cl['shot_made_flag'].copy()

  X = data_cl[~unknown_mask]
  X = X.drop(["shot_id","shot_made_flag"], 1);
  Y = target[~unknown_mask]

  seed = 9
  test_size = 0.25
  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

  model_rf = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=50)
  model_rf.fit(X_train, y_train)
  y_pred = model_rf.predict(X_test)
  predictions = [round(value) for value in y_pred]
  accuracy = accuracy_score(y_test, predictions)
  print("Accuracy Random Forest: %.2f%%" % (accuracy * 100.0))

  model_xgb = XGBClassifier()
  model_xgb.fit(X_train, y_train)
  # make predictions for test data
  y_pred = model_xgb.predict(X_test)
  predictions = [round(value) for value in y_pred]
  # evaluate predictions
  accuracy = accuracy_score(y_test, predictions)
  print("Accuracy XGB: %.2f%%" % (accuracy * 100.0))
    
  #accuracy of boosted tree slightly better than random trees
  target_X = df_input[unknown_mask]
  target_X = target_X.drop(["shot_id","shot_made_flag"], 1);
  target_y = model_xgb.predict_proba(target_X)[:,1]
  target_id = df_input[unknown_mask]["shot_id"]
    
  submission = pd.DataFrame({"shot_id":target_id, "shot_made_flag":target_y})
  submission.sort_values('shot_id',  inplace=True)
  
  return submission
      
      
      
def clean_data(df_input):
  
  print(df_input.at[0, "team_id"], df_input.at[0, "team_name"])
  df_input = df_input.drop(["team_id", "team_name"], 1)
  
  cols_print = ('variable', 'type', 'unique values', 'missing values [%]')
  print('{: <20} {: >20} {: >20} {: >20}'.format(*cols_print)); print()
  for col in df_input.columns.values:
    type_col = str(type(df_input.at[0, str(col)])).split("'")[1]
    uniq_col = len(df_input[str(col)].unique())
    miss_col = str(100*round(df_input[str(col)].isnull().sum()/len(df_input.index), 2))+"%"
    cols_print = (str(col), type_col, str(uniq_col), miss_col)
    print('{: <20} {: >20} {: >20} {: >20}'.format(*cols_print))    
    
  upper_number_branches = 1
  for col in df_input.columns.values:
    if col!="shot_id" and col!="shot_made_flag":
      upper_number_branches = upper_number_branches*len(df_input[str(col)].unique())
  print("There are up to", "{:.0e}".format(upper_number_branches), "combinations")      
  
  #draw scatter plots of spatial continuous variables  
  fig = plt.figure(figsize=(18, 8))
  fig.add_subplot(1,2,1)
  plt.scatter(x=df_input['loc_x'],y=df_input['loc_y'])
  plt.title("Location: y vs x")
  fig.add_subplot(1,2,2)
  plt.scatter(x=df_input['lon'],y=df_input['lat'])
  plt.title("Latitude vs Longitude")
  plt.show()  
    
  #drop lat, lon as they are the same
  df_input = df_input.drop(["lat", "lon"], 1)
  
  #efficiency vs distance
  fig = plt.figure(figsize=(12, 6))
  sb.barplot('shot_distance', 'shot_made_flag', data=df_input[df_input["shot_distance"]<35])
  plt.title("Efficiency as a function of shot distance")
  plt.show()  
  
  #plot categorical variables
  fig = plt.figure(figsize=(18, 6))
  plt.subplot(1,4,1)
  color_scatter_plot(df_input, "shot_type")
  plt.title("Shot type")
  plt.subplot(1,4,2)
  color_scatter_plot(df_input, "shot_zone_range")
  plt.title("Shot zone range")
  plt.subplot(1,4,3)
  color_scatter_plot(df_input, "shot_zone_area")
  plt.title("Shot zone area")
  plt.subplot(1,4,4)
  color_scatter_plot(df_input, "shot_zone_basic")
  plt.title("Shot zone basic")
  plt.show()
  
  
  #plot desnity plot of shot location: are there clusters?  
  cmap = mpl.cm.get_cmap('viridis')
  jp = sb.jointplot(df_input["loc_x"], df_input["loc_y"], cmap=cmap, n_levels=50,
                    stat_func=None,kind='kde', space=0, color=cmap(0.1))
  ax = jp.ax_joint
  pf.draw_court(ax=ax)
  ax.set_xlim(-250,250)
  ax.set_ylim(422.5, -47.5)
  plt.show()  
    
  #plot nice 2d density plots for two different season
  seasons = ["1999-00", "2009-10"] #first and last championship
  pf.make_kobe_vs_kobe_plots(seasons, df_input)
  
  seasons = ["2009-10", "2014-15"] #last championship and last healthy season
  pf.make_kobe_vs_kobe_plots(seasons, df_input)  
  
    
  #make clusters to get rid of meaningless spatial continuous variables
  numGaussians = 20
  gaussianMixtureModel = mixture.GaussianMixture(n_components=numGaussians, n_init=5, random_state=3)
  gaussianMixtureModel.fit(df_input[['loc_x','loc_y']])
  df_input['LocCluster'] = gaussianMixtureModel.predict(df_input[['loc_x','loc_y']])   
  fig = plt.figure(figsize=(12, 6))
  pf.draw_court(outer_lines=True); 
  plt.ylim(-60,440); plt.xlim(270,-270); 
  plt.title('cluster assignment')
  num_elem = len(df_input['LocCluster'].unique()) 
  colors = cm.rainbow(np.linspace(0, 1, num_elem))
  for i in range(0, num_elem):
    plt.scatter(x=df_input['loc_x'][df_input['LocCluster']==i],
                y=df_input['loc_y'][df_input['LocCluster']==i], 
                c=colors[i], edgecolors='none')
  plt.show()   
  
  #are new clusters indicative of shot percentage?
  x_x=[-1]
  var_y = ["loc_x", "loc_y", "shot_distance"]
  y_x = [[],[],[]]
  fig = plt.figure(figsize=(18, 6))
  for iy in range(0, len(var_y)):
    y_x[iy]=[prob_independence(df_input, var_y[iy], "shot_made_flag")]

    for i in range(0,len(df_input['LocCluster'].unique())):
      if iy==0:
        x_x.append(i)
      y_x[iy].append(prob_independence(
                     df_input[df_input["LocCluster"]==i], 
                     var_y[iy], "shot_made_flag"))

    plt.subplot(1,3,iy+1); 
    plt.bar(x_x, y_x[iy]); 
    plt.title("Chi2/ndf vs "+var_y[iy]);
    
  plt.show()
  
  #thinner distance bins for three clusters where there is an efficiency dependence on distance
  fig = plt.figure(figsize=(18, 8))
  plt.subplot(1,4,1)
  pf.draw_court(outer_lines=True); 
  plt.ylim(-60,440); plt.xlim(270,-270);
  plt.title("Bad clusters")
  bad_clusters = [1, 2, 15]
  for i in bad_clusters:
    plt.scatter(x=df_input['loc_x'][df_input['LocCluster']==i],
                y=df_input['loc_y'][df_input['LocCluster']==i], 
                edgecolors='none')
  for i in range(0,len(bad_clusters)): 
    plt.subplot(1,4,2+i)
    sb.barplot('shot_distance', 'shot_made_flag', data=df_input[df_input["LocCluster"]==bad_clusters[i]])
    plt.title('Eff. in Cluster '+str(bad_clusters[i]))
  plt.show()  
  
  bad_clusters = [1, 2, 15]
  bins_clusters = [[1,2,3],[25,26,27,28],[24,25,26]]
  bins_pos = [0, len(bins_clusters[0]), len(bins_clusters[0])+len(bins_clusters[1])]
  bin_ref = len(df_input['LocCluster'].unique())
  for ie in range(0, len(df_input.index)):
    for ic in range(0, len(bad_clusters)):
      if df_input.at[ie, 'LocCluster']==bad_clusters[ic]:
        for ib in range(0, len(bins_clusters[ic])-1):
          if df_input.at[ie, 'shot_distance']==bins_clusters[ic][ib]:
            df_input.at[ie, 'LocCluster'] = bin_ref + bins_pos[ic] + ib
          if df_input.at[ie, 'shot_distance']>=bins_clusters[ic][len(bins_clusters[ic])-1]:
            df_input.at[ie, 'LocCluster'] = bin_ref + bins_pos[ic] + len(bins_clusters[ic])-1
  
  #and now all clusters have a constant efficiency within it
  x_x=[]
  var_y = ["loc_x", "loc_y", "shot_distance"]
  y_x = [[],[],[]]
  fig = plt.figure(figsize=(18, 6))
  for iy in range(0, len(var_y)):
    for i in range(0,len(df_input['LocCluster'].unique())):
      if iy==0:
        x_x.append(i)
      y_x[iy].append(prob_independence(
                     df_input[df_input["LocCluster"]==i], 
                     var_y[iy], "shot_made_flag"))

    plt.subplot(1,3,iy+1); 
    plt.bar(x_x, y_x[iy]); 
    plt.title("Chi2/ndf vs "+var_y[iy]);  
  
  plt.show()
  
  #can drop all other spatial variables -> large decrease in dimensionality
  df_input = df_input.drop(["shot_zone_basic", "shot_zone_range", "shot_zone_area", 
                          "shot_type", "loc_x", "loc_y", "shot_distance"], 1)
  print("Decreased from spatial dimensionality of", 457*457*489*489*74*2*6*7*5, "to", 30)

  #deal with type of shot
  fig = plt.figure(figsize=(15, 6))
  plt.subplot(1,2,1)
  pl1 = sb.barplot('action_type', 'shot_made_flag', data=df_input)
  pl1.set(xticklabels=[])
  plt.subplot(1,2,2)
  pl2 = sb.barplot('combined_shot_type', 'shot_made_flag', data=df_input)
  pl2.set(xticklabels=[])
  plt.show()

  df_input = df_input.drop("combined_shot_type", 1)
  print(df_input["action_type"].unique())
  
  #take away correlated names and names in less than 0.1% of shots
  df_shot = rewrite_shot_type(df_input)
  df_shot.drop("shot_made_flag", 1)
  for i in df_shot.columns.values:
    if df_shot[i].mean()*df_shot[i].count()<30:
      df_shot.drop(i, 1)
    else:
      df_input[i] = df_shot[i]

  df_input = df_input.drop("action_type", 1)
  
  #is he more efficient in home games
  df_input["Home_game"] = 1
  for i in range(0, len(df_input.index)):
    if df_input.at[i, "matchup"].split(" ")[1]=="@":
      df_input.at[i,'Home_game']=0

  fig = plt.figure(figsize=(20, 8))
  fig.add_subplot(1, 3, 1)
  sb.barplot("Home_game", "shot_made_flag", data=df_input)
  plt.title("Efficiency vs Home_game")
  fig.add_subplot(1, 3, 2)
  sb.barplot("opponent", "shot_made_flag", data=df_input)
  plt.xticks(rotation='vertical')
  fig.add_subplot(1, 3, 3)
  sb.barplot("Home_game", "shot_made_flag", data=df_input[df_input["opponent"]=="LAC"])
  plt.title("Efficiency against the Clippers")
  plt.show()

  #he shoots as well as home as in the clippers court - sorry steve ballmer!
  df_input.loc[df_input["opponent"]=="LAC", "Home_game"] = 1
  #make opponent numeric
  ishot=0
  df_input["opponent_N"]=0
  for team in df_input['opponent'].unique():
    df_input.loc[df_input["opponent"]==team, "opponent_N"] = ishot 
    ishot = ishot + 1
    
  df_input = df_input.drop(["matchup", "opponent"], 1);

  df_input = df_input.drop("game_id", 1) 

  #turn date string in datetime for easier comparisons
  df_input['game_date'] = pd.to_datetime(df_input['game_date'])
  df_input = df_input.sort_values(by="game_date", ascending=1)
  df_input['game_month'] = df_input['game_date'].dt.month
  df_input['game_month_day'] = df_input['game_date'].dt.day
  df_input['game_weekday'] = df_input['game_date'].dt.dayofweek

  df_input.at[i, "season_N"] = 0
  for i in range(0, len(df_input.index)):
    df_input.at[i, "season_N"] = int(df_input.at[i,'season'].split('-')[0])-1995
  df_input = df_input.drop("season", 1)
  df_input['season'] = df_input['season_N'].copy()
  df_input = df_input.drop("season_N", 1)
    
  fig = plt.figure(figsize=(15, 6))
  vars = ["game_month", "game_month_day", "game_weekday", "season"]
  for i in range(0, len(vars)):
    fig.add_subplot(1, 4, i+1)
    sb.barplot(vars[i], "shot_made_flag", data=df_input)
    plt.title("Efficiency vs "+vars[i])
  plt.show()

  #does he play better when he rests a lot?
  df_input["days_between_games"] = 7
  str_days_between_games = [0]; str_game_id = [0];
    
  days_in=7
  for i in range(1, len(df_input.index)):
    if df_input.at[i, 'game_date'] != df_input.at[i-1, 'game_date']:
      days_in = min(abs((df_input.at[i, 'game_date'] - df_input.at[i-1, 'game_date']).days), 7) 
    df_input.at[i, "days_between_games"] = days_in

  fig = plt.figure(figsize=(12, 6))
  sb.barplot("days_between_games", "shot_made_flag", data=df_input)
  plt.show()
  
  #what about if he has games more than two days in a row
  df_input["N_back_to_back"] = 0
  for i in range(0, len(df_input.index)):
    if df_input.at[i, "days_between_games"]>1:
      counter=0
      continue
    df_input.at[i, "N_back_to_back"] = min(len(df_input["game_date"][i-counter:i].unique()), 3)
    counter+=1

  fig = plt.figure(figsize=(12, 6))
  sb.barplot("N_back_to_back", "shot_made_flag", data=df_input)
  plt.title("Efficiency by games in a row")
  plt.show()

  df_input = df_input.drop("days_between_games", 1)

  for i in range(0, len(df_input.index)):
    df_input.at[i, 'game_date_N'] = int(df_input.at[i, 'game_date'].strftime("%d%m%Y"))


  #what about in different games of the playoffs, are efficiencys different?
  df_po = df_input[df_input["playoffs"]==1]
  uniq_series_opp = []; uniq_series_key = [];

  for season in df_po["season"].unique():
    data_season = df_po[df_po["season"]==season]
    uniq_season = data_season["opponent_N"].unique()
    for opp in range(0,len(uniq_season)):
      data_opp = data_season["game_date_N"][df_po["opponent_N"]==uniq_season[opp]]
      for game in range(0,len(data_opp.unique())):
        date=data_opp.unique()[game]
        uniq_series_opp.append([int(season), int(uniq_season[opp]), date])
        uniq_series_key.append([opp+1, game+1])
            
  df_input["game_in_series"] = 0
  df_input["playoff_series"] = 0
  for i in range(0, len(df_input.index)):
    if df_input.at[i, "playoffs"]==0:
      continue
    for ik in range(0, len(uniq_series_key)):
      if uniq_series_opp[ik][0] == int(df_input.at[i, "season"]) \
      and uniq_series_opp[ik][1] == df_input.at[i, "opponent_N"] \
      and uniq_series_opp[ik][2] == df_input.at[i, "game_date_N"]:
        df_input.at[i, "playoff_series"] = uniq_series_key[ik][0]
        df_input.at[i, "game_in_series"] = uniq_series_key[ik][1]
            
  fig = plt.figure(figsize=(14, 8))
  fig.add_subplot(1, 2, 1)
  sb.barplot("playoff_series", "shot_made_flag", data=df_input)
  plt.title("Efficiency by playoff series (reg. season = 0)")
  fig.add_subplot(1, 2, 2)
  sb.barplot("game_in_series", "shot_made_flag", data=df_input)
  plt.title("Efficiency by game in series (reg. season = 0)")
  plt.show()
  
  df_input = df_input.drop(["playoffs", "game_date", "game_date_N", "game_month_day"], 1)

  #finally, in-game variables
  df_input["seconds_remaining"] = df_input["seconds_remaining"] + 60*df_input["minutes_remaining"]
  bin_sec = [120, 60, 30, 10, 5, 3]; bin_pos = [480, 240, 60, 20, 11, 3];
  for i in range(0,len(df_input.index)):
    for ibin in range(0,len(bin_sec)):
      if(df_input.at[i, "seconds_remaining"]>bin_pos[ibin]):
        df_input.at[i, "seconds_remaining"] = bin_sec[ibin]*int(
                   df_input.at[i, "seconds_remaining"]/bin_sec[ibin])    
  df_input = df_input.drop(["minutes_remaining", "game_event_id"], 1)

  fig = plt.figure(figsize=(12, 6))
  fig.add_subplot(1, 2, 1)
  sb.barplot("seconds_remaining", "shot_made_flag", data=df_input)
  plt.title("Efficiency vs seconds left")
  fig.add_subplot(1, 2, 2)
  sb.barplot("period", "shot_made_flag", data=df_input)
  plt.title("Efficiency vs period")
  plt.show()
  
  #does he get over confident when he make a few shots in a row? what about misses?
  df_input = calc_shots_in_a_row(df_input)
  
  #final quantifiers of decrease in dimensionality
  upper_number_branches = 1
  for col in df_input.columns.values:
    if col!="shot_id" and col!="shot_made_flag":
      upper_number_branches = upper_number_branches*len(df_input[str(col)].unique())
  print("Possible combinations decreased from 4e+35 to", "{:.0e}".format(upper_number_branches))    
  make_pretty_corr_plot(df_input)
  
      
def main():

  df_input = pd.read_csv("data/data.csv")
  df_input = clean_data(df_input)
  submission = make_predictions(df_input)
  submission.to_csv("submission_file.csv",index=False)
  
main()