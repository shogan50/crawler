import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

def print_rewards(rewards, ave_len=100):
    if(len(rewards)>1):
        x = np.arange(0, len(rewards[-ave_len:]))
        y = np.nan_to_num(rewards[-ave_len:])  # getting occasional NAN.  TODO: why?
        z = np.polyfit(x, y, 1)  # outputs [a, b] as in ax+b I think
        slope = z[0]
        print('step:{}  act:{:.2f} mean:{:.3f} slope:{:.3f} std:{:.3f}' \
              .format(len(rewards),
                      rewards[-1],
                      np.mean(rewards[-ave_len:]),
                      slope,
                      np.std(rewards[-ave_len:])))


class Plot_Scores:
    def __init__(self):
        matplotlib.use('tkagg')  # needed to run on AWS wiht X11 forwarding
        self.line, = plt.plot(np.array(0), np.array(0))
        self.axes = plt.gca()

        plt.ion()
        plt.xlabel = 'Episode'
        plt.ylabel = 'Mean score'


    def plot(self, scores_hist):

        if len(scores_hist) > 2:
            self.line.set_xdata(np.arange(0, len(scores_hist)))
            self.line.set_ydata(scores_hist)
            self.axes.set_xlim(max(0, len(scores_hist) - 2000), len(scores_hist))
            self.axes.set_ylim(np.min(scores_hist) * 1.05, np.max(scores_hist) * 1.05)
            plt.draw()
            plt.pause(.1)

class Logger(object):
    def __init__(self, f_name):
        self.terminal = sys.stdout
        self.log = open(f_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


