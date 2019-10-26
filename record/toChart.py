import numpy as np
import os
import os.path
import argparse
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb

parser = argparse.ArgumentParser(description='convert record to loss curve.')
parser.add_argument('--filePath', type=str, default='./',
                    help='your record file path')
args = parser.parse_args()


def readRecord(filePath):
    with open(filePath, 'r') as f:
        txt = f.readlines()
        _txt = []
        for row in txt:
            if row == '\n':
                txt.remove(row)
            if 'inf' in row:
                txt.remove(row)    # filter inf

        for row in txt:
            row = row.rstrip('\n')
            _txt.append(row.split(','))

    return np.asarray(_txt).astype('float')

def splitToArray(txt):
    num = len(txt)                   # count from 1
    iterion = txt[:, 0].astype('int')
    loss = txt[:, 1]
    classification = txt[:, 2]
    location = txt[:, 3]
    return num, iterion, loss, classification, location

def drawGrid(imgName, iterion, loss, classification, location):
    fig = plt.figure(figsize=(50, 30), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])
    ax3 = fig.add_subplot(spec[2, 0])

    ax1.plot(iterion, loss, 'C1', label='SUM')
    ax1.legend(fontsize=50)
    ax1.grid()
    ax1.set_ylim(0, 15)

    ax2.plot(iterion, classification, 'C2', label='Confidence')
    ax2.legend(fontsize=50)
    ax2.grid()
    ax2.set_ylim(0, 10)

    ax3.plot(iterion, location, 'C3', label='location')
    ax3.legend(fontsize=50)
    ax3.grid()
    ax3.set_ylim(0, 10)

    fig.savefig(imgName)
    print('one down')



    
    

def main():
    files= os.listdir(args.filePath)
    if 'visdrone.txt' in files:
        files.remove('visdrone.txt')
    for fileName in files:
        imgName = os.path.basename(fileName).split('.')[0] + '.png'
        imgName = os.path.join(args.filePath, imgName)
        record = readRecord(os.path.join(args.filePath, fileName))  # numpy array
        num, iterion, loss, classification, location = splitToArray(record)
        drawGrid(imgName, iterion, loss, classification, location)




main()
