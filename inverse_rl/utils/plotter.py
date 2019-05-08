import itertools

import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon

import numpy as np
import matplotlib.pyplot as plt

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([0.4, 0.5]) - 0.5


class TabularPlotter(object):
    def __init__(self, w, h, invert_y=True, text_values=True):
        self.w = w
        self.h = h
        self.text_values = text_values
        self.invert_y = invert_y
        self.data = np.zeros((w, h))

    def set_value(self, x, y, value):
        self.data[x, y] = value

    def make_plot(self):
        plt.figure()
        ax = plt.gca()

        #normalized_values = (self.data/np.abs(np.max(self.data)))
        normalized_values = self.data
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values / np.max(normalized_values)

        cmap = cm.Blues

        for x, y in itertools.product(range(self.w), range(self.h)):
            if self.invert_y:
                y = self.h - y - 1

            xy = np.array([x, y])
            val = normalized_values[x, y]
            og_val = self.data[x, y]

            if self.text_values:
                xy_text = xy + TXT_CENTERING
                ax.text(
                    xy_text[0], xy_text[1], '%.1f' % og_val, size='x-small')
            color = cmap(val)
            ax.add_patch(
                Rectangle(xy - 0.5, 1, 1, facecolor=color, edgecolor='black'))

        ax.set_xticks(np.arange(-0.5, self.w + .5, 1))
        ax.set_yticks(np.arange(-0.5, self.h + .5, 1))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        #plt.grid()

    def show(self):
        plt.show()


if __name__ == "__main__":
    plotter = TabularPlotter(6, 8)
    plotter.set_value(4, 3, 0.5)
    plotter.set_value(4, 4, -0.5)
    plotter.make_plot()
    plt.show()
