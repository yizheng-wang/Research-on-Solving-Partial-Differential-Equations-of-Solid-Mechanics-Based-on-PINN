#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting and post-processing
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_pred(x, u_exact, u_pred, title):
    plt.figure()
    plt.plot(x, u_exact, 'b', label='Ground Truth')
    plt.plot(x, u_pred, '--r', label = 'Prediction')
    plt.legend()
    plt.title(title)
    title = title.replace('\\','').replace('$','')
    plt.savefig( title + '.png', dpi=300)
    #plt.show()
    rel_l2_error = np.linalg.norm(u_exact-u_pred)/np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)
    
def plot_pred1(x, y, f, title):
    plt.figure()
    plt.contourf(x, y, f)
    plt.colorbar()
    plt.title('Input ' + title)
    

def plot_pred2(x, y, u_exact, u_pred, title, savetitle):
    savetitle = savetitle.replace('\\','').replace('$','')
    plt.figure()
    plt.contourf(x, y, u_pred)
    plt.colorbar()
    plt.title('Approximate solution - ' + title)
    plt.savefig( savetitle + ' - Approximate solution' + '.png', dpi=300)

    #plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact)
    plt.colorbar()
    plt.title('Exact solution - ' + title)
    plt.savefig( savetitle + ' - Exact solution' + '.png', dpi=300)

    #plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact-u_pred)
    plt.colorbar()
    plt.title('Error - ' + title)
    plt.savefig( savetitle + ' - Error' + '.png', dpi=300)
    rel_l2_error = np.linalg.norm(u_exact-u_pred)/np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)
    #plt.show()
    
def plot_pred_timo(x, y, u_exact, u_pred, title, SaveFolder):
    if not os.path.exists(SaveFolder):
        os.makedirs(SaveFolder)
        
    plt.figure()
    plt.contourf(x, y, u_pred, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title + '- Approx')
    plt.axis('equal')
    plt.savefig(SaveFolder + title + " - Approx"  + ".png", dpi=300)
    #plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title + '- Exact')
    plt.axis('equal')
    plt.savefig(SaveFolder + title + ' - Exact'  + '.png', dpi=300)
    #plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact-u_pred, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title + '- Error')
    plt.axis('equal')
    plt.savefig(SaveFolder + title + ' - Error'  + '.png', dpi=300)
    #plt.show()
    
