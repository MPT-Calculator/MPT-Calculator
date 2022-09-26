#Importing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import csv
import cmath
import os
import subprocess
import sys


#Edit strings
def FloattoStringPlot(Number,tick_number):
    for i in range(100):
        if Number>=1:
            if Number/(10**i-1)<1:
                power=i-1
                break
        else:
            if Number/(10**-(i-1))>=1:
                power=-(i-1)
                break
    StringNumber=str(Number/(10**power))
    SnipNumber=StringNumber[:4]+"e"+str(power)
    return SnipNumber
    
def format_func(value,tick_number):
    newvalue='%s' %'%.2g' % value
    #newvalue=round(float(value),1)
    return str(newvalue)


#PlotterSaver creates a graph saves all the inputs of the graph
#creates a script which can be run to reproduce using the original inputs
#It's inputs are as follows
#
#Plotname = string
#x = single dimension array
#ys = 2dimension array with each line to be aligned with axis=0
#styles = list of inputs which define the line type
#colours = single dimension array with integer inputs from 1-10 these 
#    correspond to each of the colours python loops through
#names = list of strings for the legend (False if no legend)
#xlabel = string for the xaxis label (False if no xaxis label)
#ylabel = string for the yaxis label (False if no yaxis label)
#xlog = boolean for whether to make the xaxis log scale
#ylog = boolean for whether to make the yaxis log scale
#title = string for the title (False if no title)
#show = boolean whether to display the plot


#Whatever you do, DO NOT edit the x or ys matrices when using
#the Replot.py file since this could potentially lose the original data



def PlotterSaver(Plotname,x,ys,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show):
    SolDims=np.shape(ys)
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    
    #Save the Data
    #Check that we are not creating subdirectories if we are in the results file already
    cwd = os.getcwd()
    if cwd.find(Plotname)==-1:
        try:
            os.makedirs("Results/Graphs/"+Plotname)
            print("Target directory Created")
        except:
            print("Target directory already exists")
        np.savetxt("Results/Graphs/"+Plotname+"/x.csv",x, delimiter=",")
        np.savetxt("Results/Graphs/"+Plotname+"/ys.csv",ys, delimiter=",")
        wr = csv.writer(open("Results/Graphs/"+Plotname+"/styles.csv",'w+'),dialect='excel')
        wr.writerow(styles)
        wr = csv.writer(open("Results/Graphs/"+Plotname+"/colours.csv",'w+'),dialect='excel')
        wr.writerow(colours)
        wr = csv.writer(open("Results/Graphs/"+Plotname+"/names.csv",'w+'),dialect='excel')
        wr.writerow(names)
    
        #Save this file to the same directory
        subprocess.call(['cp','PlotterSaver.py','Results/Graphs/'+Plotname+'/PlotterSaver.py'])
    
    #Create a file which produces the same plot
    #First create the lines of code to go into the file
    newlines=['#Importing\n']
    newlines.append('import numpy as np\n')
    newlines.append('import cmath\n')
    newlines.append('import csv\n')
    newlines.append('from PlotterSaver import PlotterSaver\n')
    newlines.append('\n')
    newlines.append('\n')
    newlines.append('#To reproduce the graph run it with the following parameters\n')
    newlines.append('Plotname = "'+Plotname+'"\n')
    newlines.append('x = np.genfromtxt("x.csv",delimiter=",")\n')
    newlines.append('ys = np.genfromtxt("ys.csv",delimiter=",")\n')
    newlines.append('with open("styles.csv","r") as f:\n')
    newlines.append('    reader = csv.reader(f)\n')
    newlines.append('    styles = list(reader)[0]\n')
    if len(SolDims)==1:
        newlines.append('styles = [styles]\n')
    newlines.append('colours = np.genfromtxt("colours.csv",delimiter=",")\n')
    if len(SolDims)==1:
        newlines.append('colours = [colours]\n')
    newlines.append('with open("names.csv","r") as f:\n')
    newlines.append('    reader = csv.reader(f)\n')
    newlines.append('    names = list(reader)[0]\n')
    newlines.append('xlabel = "'+xlabel+'"\n')
    newlines.append('ylabel = "'+ylabel+'"\n')
    if xlog==True:
        newlines.append('xlog = True\n')
    else:
        newlines.append('xlog = False\n')
    if ylog==True:
        newlines.append('ylog = True\n')
    else:
        newlines.append('ylog = False\n')
    if title!=False:
        newlines.append('title = "'+title+'"\n')
    else:
        newlines.append('title = False\n')
    newlines.append('show = True\n')
    newlines.append('\n')
    newlines.append('\n')
    newlines.append('PlotterSaver(Plotname,x,ys,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)\n')
    newlines.append('\n')

    
    #Move a copy of this file to the target directory
    if cwd.find(Plotname)==-1:
        f=open("Results/Graphs/"+Plotname+"/ReplotOriginal.py","w")
        for line in newlines:
            f.write(line)
        f.close
        
    
    #Plot the Graph
    fig, ax = plt.subplots()
    try:
        numlines=SolDims[1]
        for i in range(numlines):
            if i==0:
                lines = ax.plot(x,ys[:,i],styles[i],markersize=8,color=PYCOL[int(colours[i])-1])
            else:
                lines += ax.plot(x,ys[:,i],styles[i],markersize=8,color=PYCOL[int(colours[i])-1])
    except:
        lines = ax.plot(x,ys[:],styles[0],markersize=8,color=PYCOL[int(colours[0])-1])
    #Create the legend
    if names!=False:
        ax.legend(lines,names)
    
    #Take care of the axes labels
    if xlabel!=False:
        plt.xlabel(xlabel)
    if ylabel!=False:
        plt.ylabel(ylabel)
    
    #Currently always includes a grid
    ax.grid(True)
    
    #check whether the axes show be log
    if xlog==True:
        plt.xscale('log')
    if ylog==True:
        plt.yscale('log')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    #plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.94)
    
    #save the figure
    if cwd.find(Plotname)==-1:
        plt.savefig("Results/Graphs/"+Plotname+"/"+Plotname+".pdf")
    else:
        plt.savefig(Plotname+".pdf")
    
    #plot the graph if required
    if show==True:
        plt.show()
    
    return
    




#MultiPlotterSaver creates a graph saves all the inputs of the graph
#creates a script which can be run to reproduce using the original inputs
#It's inputs are as follows
#
#Plotname = string
#xs = list of list containing each of the x vectors to be used
#ys = 2dimension array with each line to be aligned with axis=0
#    if x vectors are different length pad out the shorter y
#    vectors with zeros (these will not be plotted anyway)
#number of lines = list of how many lines for each x vector
#styles = list of lists with inputs which define the line type
#colours = list of lists with integer inputs from 1-10 these 
#    correspond to each of the colours python loops through
#names = list of strings for the legend (False if no legend)
#xlabel = string for the xaxis label (False if no xaxis label)
#ylabel = string for the yaxis label (False if no yaxis label)
#xlog = boolean for whether to make the xaxis log scale
#ylog = boolean for whether to make the yaxis log scale
#title = string for the title (False if no title)
#show = boolean whether to display the plot


#Whatever you do, DO NOT edit the x or ys matrices when using
#the Replot.py file since this could potentially lose the original data



def MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show):
    SolDims=np.shape(ys)
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    #Save the Data
    #Check that we are not creating subdirectories if we are in the results file already
    cwd = os.getcwd()
    if cwd.find(Plotname)==-1:
        try:
            os.makedirs("Results/Graphs/"+Plotname)
            print("Target directory Created")
        except:
            print("Target directory already exists")
        with open("Results/Graphs/"+Plotname+"/xs.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerows(xs)
        with open("Results/Graphs/"+Plotname+"/ys.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerows(ys)
        np.savetxt("Results/Graphs/"+Plotname+"/linenumbers.csv",numberoflines, delimiter=",")
        with open("Results/Graphs/"+Plotname+"/styles.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerows(styles)
        with open("Results/Graphs/"+Plotname+"/colours.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerows(colours)
        wr = csv.writer(open("Results/Graphs/"+Plotname+"/names.csv",'w+'),dialect='excel')
        wr.writerow(names)
    
        #Save this file to the same directory
        subprocess.call(['cp','PlotterSaver.py','Results/Graphs/'+Plotname+'/PlotterSaver.py'])
    
    #Create a file which produces the same plot
    #First create the lines of code to go into the file
    newlines=['#Importing\n']
    newlines.append('import numpy as np\n')
    newlines.append('import cmath\n')
    newlines.append('import csv\n')
    newlines.append('from PlotterSaver import MultiPlotterSaver\n')
    newlines.append('\n')
    newlines.append('\n')
    newlines.append('#To reproduce the graph run it with the following parameters\n')
    newlines.append('Plotname = "'+Plotname+'"\n')
    newlines.append('with open("xs.csv", "r") as f:\n')
    newlines.append('    reader = csv.reader(f)\n')
    newlines.append('    xs = list(reader)\n')
    newlines.append('ys = np.genfromtxt("ys.csv",delimiter=",")\n')
    newlines.append('numberoflines = np.genfromtxt("linenumbers.csv",delimiter=",")\n')
    newlines.append('if np.size(numberoflines)==1:\n')
    newlines.append('    numberoflines = [numberoflines]\n')
    newlines.append('with open("styles.csv","r") as f:\n')
    newlines.append('    reader = csv.reader(f)\n')
    newlines.append('    styles = list(reader)\n')
    if len(SolDims)==1:
        newlines.append('styles = [styles]\n')
    newlines.append('with open("colours.csv","r") as f:\n')
    newlines.append('    reader = csv.reader(f)\n')
    newlines.append('    colours = list(reader)\n')
    newlines.append('with open("names.csv","r") as f:\n')
    newlines.append('    reader = csv.reader(f)\n')
    newlines.append('    names = list(reader)[0]\n')
    newlines.append('xlabel = "'+xlabel+'"\n')
    newlines.append('ylabel = "'+ylabel+'"\n')
    if xlog==True:
        newlines.append('xlog = True\n')
    else:
        newlines.append('xlog = False\n')
    if ylog==True:
        newlines.append('ylog = True\n')
    else:
        newlines.append('ylog = False\n')
    if title!=False:
        newlines.append('title = "'+title+'"\n')
    else:
        newlines.append('title = False\n')
    newlines.append('show = True\n')
    newlines.append('\n')
    newlines.append('\n')
    newlines.append('MultiPlotterSaver(Plotname,xs,ys,numberoflines,styles,colours,names,xlabel,ylabel,xlog,ylog,title,show)\n')
    newlines.append('\n')

    
    #Create the file and save it in the target directory
    cwd = os.getcwd()
    if cwd.find(Plotname)==-1:
        f=open("Results/Graphs/"+Plotname+"/ReplotOriginal.py","w")
        for line in newlines:
            f.write(line)
        f.close
    
    #Plot the Graph
    fig, ax = plt.subplots()
    linesplotted=0
    for l,line in enumerate(xs):
        x=list(map(float,xs[l]))
        ystop=len(x)
        linestoplot=int(numberoflines[l])
        y=ys[:ystop,linesplotted:linesplotted+linestoplot]
        linesplotted+=linestoplot
        style=styles[l]
        colour=colours[l]
        for i in range(linestoplot):
            if linestoplot==1:
                if l==0 and i==0:
                    lines = ax.plot(x,y,style[i],markersize=5,color=PYCOL[int(colour[i])-1])
                else:
                    lines += ax.plot(x,y,style[i],markersize=8,color=PYCOL[int(colour[i])-1])
            else:
                if l==0 and i==0:
                    lines = ax.plot(x,y[:,i],style[i],markersize=7,color=PYCOL[int(colour[i])-1])
                else:
                    lines += ax.plot(x,y[:,i],style[i],markersize=7,color=PYCOL[int(colour[i])-1])

    if names!=False:
        ax.legend(lines,names)
    
    #Take care of the axes labels
    if xlabel!=False:
        plt.xlabel(xlabel)
    if ylabel!=False:
        plt.ylabel(ylabel)
    
    #Currently always includes a grid
    ax.grid(True)
    
    #check whether the axes show be log
    if xlog==True:
        plt.xscale('log')
    if ylog==True:
        plt.yscale('log')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    #save the figure
    if cwd.find(Plotname)==-1:
        plt.savefig("Results/Graphs/"+Plotname+"/"+Plotname+".pdf")
    else:
        plt.savefig(Plotname+".pdf")
    
    #plot the graph if required
    if show==True:
        plt.show()
    
    return





















    
