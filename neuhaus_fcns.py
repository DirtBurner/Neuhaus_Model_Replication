#These functions allow quick manipulation and visualization of the Sarah Neuhaus radiocarbon and TOC
#model (Neuhaus et al., 2021, The Cryosphere). 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np

#Constants from paper:
tau = 8033              #mean lifetime of 14C based off of the Libby half life.
modern_ratio = 1.176810*10**(-12)
                        #Modern ratio of 14C/12C to transform to fraction modern - from the Neuhaus et al. text. 

DEBUG = False

def debug(*args):
    if DEBUG == True:
        print(args)



def model_retreat(t_earliest, t_readvance, a, A, N_t0=0.28):
    '''
    Iteration of the model for time period that SLM is exposed to the ocean, t_earliest to T_readvance

        Input variables:
            t_earliest (float): the calendar year before present that ice could have retreated in the Holocene. In
                Neuhaus et al., 2021, the authors use 8000 y for this input, citing Lee et al., 2017; McKay et 
                al., 2016; Spector et al., 2017
            t_readvance (float): the calendar year before present that ice readvanced over the point of interest. In
                Neuhaus et al., 2021, the authors use 4000 years. Venturelli et al. (2023) measure 6200 years for
                this timing. 
            a (float): accumulation rate of 14C in sediment, authors used 9.23*10**(-18) g/yr/100g sediment
            A (float): accumulation rate of 12C in sediment, authors used 9*10**(-6) g/yr/100g sediment
            N_0 (float, default=0.28): Non-zero initial condition of %TOC (g/100g sediment)

        Output variables: 
            fM_0 (list): the fractions modern calculated from dividing 14C by 12C and multiplying by the modern ratio
                (see constants above).
            N_0 (list): the amount of 12C deposited at each time step
            n_0 (list): the amount of 14C deposited at each time step
            t0_steps (list): the time steps of the retreat model

            
            
    '''
    
    t0_steps = np.linspace(t_earliest-t_earliest, t_earliest-t_readvance, 100)
    #Make A a list so that the function handles variable accumulation rates:
    if type(A) is not list:
        A = [A]*len(t0_steps)
        debug('A was a scalar. Changed to a vector of length of t0_steps.')
    debug('What type is A now? : ', type(A), 'Should always be a list')

    n_0 = [a*tau*(1-math.exp(-t/tau)) for t in t0_steps]
    N_0 = [N_t0 + acc*t for (acc, t) in zip(A, t0_steps)]

    fM_0 = [(C14/C12)/modern_ratio for (C14, C12) in zip(n_0, N_0)]
    

    return fM_0, N_0, n_0, t0_steps

def model_readvance(t_earliest, t_readvance, n_0, N_0, t0_steps):
    '''
    Iteration of the model for time period that SLM is exposed to the ocean, t_earliest to T_readvance

        Input variables:
            t_earliest (float): the calendar year before present that ice could have retreated in the Holocene. In
                Neuhaus et al., 2021, the authors use 8000 y for this input, citing Lee et al., 2017; McKay et 
                al., 2016; Spector et al., 2017
            t_readvance (float): the calendar year before present that ice readvanced over the point of interest. In
                Neuhaus et al., 2021, the authors use 4000 years. Venturelli et al. (2023) measure 6200 years for
                this timing. 
            t0_steps (list/array): The time steps from the model_retreat function
            n_0 (list): The amount of 14C deposition output from model_retreat function
            N_0 (list): The amount of 12C deposition output from the model_retreat function.
        
        Output variables:
            fM_1 (list): the fractions modern calculated from dividing 14C by 12C and multiplying by the modern ratio
                (see constants above).
            N_1 (list): the amount of 12C deposited at each time step
            n_1 (list): the amount of 14C deposited at each time step
            t1_steps (list): the time steps of the retreat model

    '''


    t1_steps = np.linspace(t_earliest-t_readvance, t_earliest, 100)

    n_1 = [n_0[-1]*math.exp(-(t-max(t0_steps))/tau) for t in t1_steps]
    N_1 = [N_0[-1] for t in t1_steps]
    

    fM_1 = [(C14/C12)/modern_ratio for (C14, C12) in zip(n_1, N_1)]
    
    

    return fM_1, N_1, n_1, t1_steps

def model_coupler(t_earliest, t_readvance, a, A):
    '''
    Couple the retreat and readvance models

        Input variables:
            t_earliest (float): the calendar year before present that ice could have retreated in the Holocene. In
                Neuhaus et al., 2021, the authors use 8000 y for this input, citing Lee et al., 2017; McKay et 
                al., 2016; Spector et al., 2017
            t_readvance (float): the calendar year before present that ice readvanced over the point of interest. In
                Neuhaus et al., 2021, the authors use 4000 years. Venturelli et al. (2023) measure 6200 years for
                this timing. 
            a (float): accumulation rate of 14C in sediment, authors used 9.23*10**(-18) g/yr/100g sediment
            A (float): accumulation rate of 12C in sediment, authors used 9*10**(-6) g/yr/100g sediment

        Output variables:
            retreat_output_dict (dictionary): timesteps, 12C, 14C, and fM of the retreat model ouptuts
            readvance_output_dict (dictionary): timesteps, 12C, 14C, and fM of the readvance model outputs

    
    '''
    fM_0, N_0, n_0, t0_steps = model_retreat(t_earliest, t_readvance, a, A)

    fM_1, N_1, n_1, t1_steps = model_readvance(t_earliest, t_readvance, n_0, N_0, t0_steps)

    
    debug('14C Test - should equal 0:', n_0[-1]-n_1[0], 'Actual value: ', n_0[-1], '-', n_1[0])
    debug('%TOC Test - should equal 0: ', N_0[-1]-N_1[0], 'Actual value: ', N_0[-1], '-', N_1[0])
    debug('fM Test - should equal 0: ', fM_0[-1]-fM_1[0])
    debug('Length of list test - Length of A input: ', len([A]), 'Length of fM_0 retreat output: ', len(fM_0))

    retreat_output_dict = {'time_steps':t0_steps, 'C12':N_0, 'C14':n_0, 'fM':fM_0}
    readvance_output_dict = {'time_steps':t1_steps, 'C12':N_1, 'C14':n_1, 'fM':fM_1}



    return retreat_output_dict, readvance_output_dict

def fig_4_emulator(retreat_out_dict, readvance_out_dict, colors=['tomato', 'cornflowerblue']):
    '''
    This function emulates the type of plot shown in Figure 4 of Neuhaus et al., 2021. This plot contains two panels,
    one with the retreated ice showing years since retreat on the x-axis, and a double y axis with fM on one and %TOC
    on the other. The original plot was blue (fM) and red (%TOC).

        Input variables:
            retreat_out_dict (dictionary): Output from the model_coupler function, or similar structure (dictionary
                keys of time_steps, C12, C14, fM)
            readvance_out_dict (dictionary): Output from the model_coupler function, or similar structure (dictionary
                keys of time_steps, C12, C14, fM)
            colors (list of strings corresponding to pyplot colors): list of colors. First in list is the %TOC, second
                is the fraction modern. Only first two colors from the list will be used. 
        
        Output variables;
            fig (pyplot figure object): Figure on which the axes are plotted
            ax (pyplot axes object): n=2 array of axes. ax[0] is the plot of t0 (retreated ice) and ax[1] is the plot
                of t1 (readvanced ice)
            ax0 (pyplot axes object): axis on which the right side y-axis is plotted for fM in the t0 (retreated ice)
                subplot
            ax1 (pyplot axes object): axis on which the right side y-axis is plotted for fM in the t1 (readvanced ice)
                subplot

    '''

    #Assign variables from dictionaries:
    t0_steps = retreat_out_dict['time_steps']
    N_0 = retreat_out_dict['C12']
    fM_0 = retreat_out_dict['fM']
    t1_steps = readvance_out_dict['time_steps']
    N_1 = readvance_out_dict['C12']
    fM_1 = readvance_out_dict['fM']
    
    #Plot results

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.set_facecolor('white')

    color = colors[0]
    ax[0].set_xlabel('time (y since retreat)')
    ax[0].set_ylabel('%TOC', color=color)
    ax[0].set_title('T$_0$')
    ax[0].plot(t0_steps, N_0, color='tomato')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax0 = ax[0].twinx()

    #plot fraction modern on different y-axis, same subplot
    color = colors[1]
    ax0.set_ylabel('', color=color)  # we already handled the x-label with ax1
    ax0.plot(t0_steps, fM_0, color=color)
    ax0.tick_params(axis='y', labelcolor=color)


    #Plot Phase 2 in second subplot - ice sheet has readvanced

    color = colors[0]
    ax[1].set_xlabel('time (y since retreat)')
    ax[1].set_ylabel('', color=color)
    ax[1].set_title('T$_1$')
    ax[1].plot(t1_steps, N_1, color='tomato')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax1 = ax[1].twinx()

    #plot fraction modern on different y-axis, same subplot
    color=colors[1]
    ax1.set_ylabel('fraction modern', color=color) 
    ax1.plot(t1_steps, fM_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.sharey(ax0)
    #ax0.yaxis.set_ticklabels([])
    #ax0.set_yticks([])

    return fig, ax, ax0, ax1



def sensitivity_test_plotter(model_out_dict, iter_var, colormap='copper'):
    '''
    Plots outputs of sensitivity tests of the Neuhaus et al 2021 model for radiocarbon accumulation. 

        Input variables:
            model_out_dict (dictionary of dictionaries): This is the output of a loop through different variables in the 
                model_coupler function. That function returns a dictionary of dictionaries, top level is retreat or advance
                as dictionary keys, and each of those dictionaries has the keys of time_steps,  C12, C14, and fM. This function
                sutures both output dictionaries effectively into one subplot for both TOC and fM to show a continuous time
                domain.
            colormap (string): string referring to a pyplot colormap. Default is copper, and the function reverts to default if
                an unrecognized colormap is entered. 
            iter_var (list of floats): This is generally the variable that was varied and looped through to create the model_
                _out_dict. 

        Output variables:
            axes (array of pyplot axes): array of axes to allow for additional formatting. 
    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    #Get the colormap:
    colors = plt.get_cmap(colormap)
    #Normalize the colormap to the length of the test variable:
    vmin, vmax = min(iter_var), max(iter_var)
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for ind, (k, v) in enumerate(model_out_dict.items()):
        debug('Plot iteration number: ', ind)
        col=colors(normalizer(iter_var[ind]))
        ax[0].plot(v['retreat']['time_steps'], v['retreat']['C12'], color=col, label=k) #Add T0 plot of TOC
        ax[1].plot(v['retreat']['time_steps'], v['retreat']['fM'], color=col, label=k)  #Add T0 plot of TOC
        ax[0].plot(v['readvance']['time_steps'], v['readvance']['C12'], color=col) #Add T1 plot of TOC
        ax[1].plot(v['readvance']['time_steps'], v['readvance']['fM'], color=col)  #Add T1 plot of TOC

    #Update graph labels and limits:
    top_lims = ax[0].get_ylim()
    bot_lims = ax[1].get_ylim()
    ax[0].set(ylabel='%TOC', title='%TOC')
    ax[1].set(ylabel='fM', xlabel='years since initial retreat', title='fraction modern')
    ax[0].legend()

    #Add vertical lines representing the time at which the grounding line crossed the site during readvance:
    for ind, (k, v) in enumerate(model_out_dict.items()):
        debug('Vert line iteration number: ', ind)
        col=colors(normalizer(iter_var[ind]))
        ax[0].vlines(v['retreat']['time_steps'][-1], min(top_lims), max(top_lims), color=col, linestyle='-.')
        ax[1].vlines(v['retreat']['time_steps'][-1], min(bot_lims), max(bot_lims), color=col, linestyle='-.')

def precision_round(number, digits=3):
    '''
    Rounds scientific notation numbers to number of digits. Courtesy of Josh Duran, Stack Overflow, 16 Dec 2020
    '''
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))

def convert_reservoir_to_raw_n(R):
    '''
    Converts from reservoir ages (something we all understand) to the raw numbers of 14C nuclides per 100 g of sediment
    required to run the Neuhaus et al model.

        Input variables:
            R (float or list of floats): postive integers representing the inherent of the modern ocean at the time of
                formation of a CaCO3 test or organic molecule from primary production. Lists can be accepted. 
        Ouput variables:
            n (float or list of floats): numbers on the magnitude of 10^(-18) which represent the number of 14C nuclides
                in 100 g of sediment. These numbers are necessary to run the Neuhaus model
    '''
    #Check for list:
    if type(R) is not list:
        print('Not a list - Variable R is a ', type(R), 'converting to a list.')
        if isinstance(R, int) and isinstance(R, float):
            print('Variable R is a ', type(R))
            R = [R]     #Makes it a list of one
        elif isinstance(R, np.ndarray):
            print('Variable is an ', type(R), '. Changing to list.')
            R = R.tolist()
        else:
            print("Must enter a float, integer, or a list of floats or integers. You entered a: ", type(R))
            return
        
    new_a = []
    A = 9*10**(-6)          #Accumulation rate of 12C, g/yr/100g sediment (from base conditions in paper)

    for ind, reservoir_age in enumerate(R):
        print(ind, reservoir_age)
        #Convert to fraction modern:
        F = math.exp(-reservoir_age/8033)
        #F is equal to the ratio of fluxes a/A, so we can calculate an a value for each F:
        new_a.append(F*modern_ratio*A)

    return new_a

  
