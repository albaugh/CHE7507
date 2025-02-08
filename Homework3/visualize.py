import numpy as np
import matplotlib.pyplot as plt

def plot_circle(ax , x, y, r, c, e):
  circle = plt.Circle((x,y), r, facecolor=c, edgecolor=e, linewidth=1.5)
  ax.add_patch (circle)

def visualize_water_molecule (x1 , y1 , x2 , y2 , x3 , y3 ):
  fig, ax = plt.subplots(figsize=(5 ,5))
  plot_circle(ax , x2 , y2 , 0.65 , 'r', 'w')
  plot_circle(ax , x1 , y1 , 0.5 , 'w', 'r')
  plot_circle(ax , x3 , y3 , 0.5 , 'w', 'r')
  ax.text(x2 , y2 , "O", color ="white", fontsize =15 , ha="center", va="center", fontweight ="bold")
  ax.text(x1 , y1 , "H", color ="red", fontsize =12 , ha="center", va="center", fontweight ="bold")
  ax.text(x3 , y3 , "H", color ="red", fontsize =12 , ha="center", va="center", fontweight ="bold")
  ax.set_xlim([ -2.5 , 2.5])
  ax.set_ylim([ -2.5 , 2.5])
  ax.axis('off')
  ax.set_aspect('equal')
  plt.show()