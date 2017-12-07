'''
Created on 25/11/2017

@author: francisco
'''
import numpy as np
import math
import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=False, deg=0):
    # If an axes object isn't provided to plot onto, just get current one
  if ax is None:
    ax = plt.gca()

  hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
  backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
  outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                        fill=False)
  inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                        fill=False)
  top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                        linewidth=lw, color=color, fill=False)
  bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                          linewidth=lw, color=color, linestyle='dashed')
  restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
  corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
  corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
  three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                  color=color)
  center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                         linewidth=lw, color=color)
  center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                          linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
  court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                    bottom_free_throw, restricted, corner_three_a,
                    corner_three_b, three_arc, center_outer_arc,
                    center_inner_arc]

  if outer_lines:
      # Draw the half court line, baseline and side out bound lines
    outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                            color=color, fill=False)
    court_elements.append(outer_lines)

  for element in court_elements:
    r2 = mpl.transforms.Affine2D().rotate_deg(deg).translate(deg*422.5/90, 0) + ax.transData
    element.set_transform(r2)
    ax.add_patch(element)

  return ax  


#make whole field compare shots in different season in each half
def make_kobe_vs_kobe_plots(seasons, df_input):    

  df_input["eff_shot_made_flag"] = np.nan
  for i in range(0, len(df_input.index)):
    if math.isnan(df_input.at[i, "shot_made_flag"]):
      continue
    df_input.at[i, "eff_shot_made_flag"] = (float(df_input.at[i, "shot_made_flag"])*
                                        float(df_input.at[i, "shot_type"][0])/2)
 
  cmap = mpl.cm.get_cmap('viridis')
  df_input["loc_x_0"]=0; df_input["loc_x_1"]=0; 
  df_input["loc_y_0"]=0; df_input["loc_y_1"]=0;
  for i in range(0, len(df_input.index)):
    df_input.at[i, "loc_y_0"] = df_input.at[i, "loc_y_1"] = df_input.at[i, "loc_x"]
    df_input.at[i, "loc_x_0"] = min(df_input.at[i, "loc_y"] - 422.5, -1)
    df_input.at[i, "loc_x_1"] = max(-df_input.at[i, "loc_y"] + 422.5, 1)

  js1 = sb.jointplot(df_input["loc_x_0"][df_input['season']==seasons[0]], 
                   df_input["loc_y_0"][df_input['season']==seasons[0]], 
                   stat_func=None,kind='kde', space=0, color=cmap(0.1),
                   cmap=cmap, n_levels=50)
  js1.fig.set_size_inches(12,8) 
  plt.close()
    
  js2 = sb.jointplot(df_input["loc_x_1"][df_input['season']==seasons[1]], 
                   df_input["loc_y_1"][df_input['season']==seasons[1]], 
                   stat_func=None,kind='kde', space=0, color=cmap(0.1),
                   cmap=cmap, n_levels=50)
  js2.fig.set_size_inches(12,8) 
  plt.close()
    
  f = plt.figure(figsize=(12,9))
  for J in [js1, js2]:
    for A in J.fig.axes:
      A.set_xlabel('')
      A.set_ylabel('')
      A.tick_params(labelbottom='off', labelleft='off', 
                    labelright='off', labeltop='off')   
      f._axstack.add(f._make_key(A), A)
        
  top_all = 0.75; bot_all = 0.05;

  f.axes[0].set_position([0.05, 0.05, 0.45, top_all-bot_all])
  f.axes[1].set_position([0.05, top_all,  0.45, 0.05])
  f.axes[2].set_position([0.0,  0.05, 0.05, top_all-bot_all])
  f.axes[3].set_position([0.5, 0.05, 0.45,  top_all-bot_all])
  f.axes[4].set_position([0.5, top_all, 0.45,  0.05])
  f.axes[5].set_position([0.95, 0.05, 0.05, top_all-bot_all])
  
  draw_court(ax=f.axes[0], outer_lines=True, deg=-90)
  draw_court(ax=f.axes[3], outer_lines=True, deg=90)
  f.axes[0].set_ylim(-275,275); f.axes[0].set_xlim(-455,0); 
  f.axes[3].set_ylim(-275,275); f.axes[3].set_xlim(0,455);     

  height = 0.77; x_k = [0.455, 0.51]; sign_k = [-1, 1];
  plt.gcf().text(x_k[0]+0.04, height - 0.02,"FG%", fontstyle="italic", 
                 fontsize=14, color="whitesmoke", family='sans-serif') 
  plt.gcf().text(x_k[0]+0.035, height - 0.09,"eFG%", fontstyle="italic", 
                 fontsize=14, color="whitesmoke", family='sans-serif')
       
  plt.gcf().text(x_k[0]-0.10, height + 0.06, seasons[0], fontstyle="italic", 
                 fontsize=20, color="darkblue", family='sans-serif') 
  plt.gcf().text(x_k[1]+0.06, height + 0.06, seasons[1], fontstyle="italic", 
                 fontsize=20, color="darkblue", family='sans-serif')       
  
  for iage in range(0, len(seasons)):          
    shot_perc = 100*df_input['shot_made_flag'][df_input['season']==seasons[iage]].mean()
    effi_perc = 100*df_input['eff_shot_made_flag'][df_input['season']==seasons[iage]].mean()
    stri_perc=(str(float(int(10*shot_perc))/10)+"%", 
               str(float(int(10*effi_perc))/10)+"%")
    plt.gcf().text(x_k[iage] + sign_k[iage]*0.01, height - 0.05, 
                   stri_perc[0], fontstyle="italic", fontsize=14, 
                   color="whitesmoke", family='sans-serif')        
    plt.gcf().text(x_k[iage] + sign_k[iage]*0.01, height - 0.12, 
                   stri_perc[1], fontstyle="italic", fontsize=14, 
                   color="whitesmoke", family='sans-serif')   
    
  plt.axis('off')
  plt.show()     
    