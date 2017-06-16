import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_from_file(filename, num_classes, iteration):
    df = pd.read_csv(filename, sep=',')

    #if (len(df)>10000):
    #    df = df[:10000]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(num_classes):
        class_name = 'class'+str(i)
        ax.scatter(df[df[class_name]==1].x, df[df[class_name]==1].y, s=1, label=class_name)

    ax.legend()
    plt.ylabel('latent_dimension[1]')
    plt.xlabel('latent_dimension[0]')
    plt.title('Distribution of training samples in the latent space - Iter '+str(iteration))

    filename=filename[:-4]+'.pdf'
    plt.savefig(filename)
    plt.close()
    return


if __name__ == '__main__':

    if len(sys.argv) == 4:
        print('Starting now...')
        plot_from_file(str(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]))
        print('Done!')
