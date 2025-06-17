import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import datetime
import os

def find_closest_index(sorted_array, target):
    """
    Find the index of the closest number in a sorted array using NumPy.

    Parameters:
        sorted_array (np.ndarray): A sorted NumPy array of numbers.
        target (float): The target number to find the closest to.

    Returns:
        int: The index of the closest number in the sorted array.
    """
    # Use searchsorted to find the insertion point
    idx = np.searchsorted(sorted_array, target, side="left")

    # Check the boundary conditions
    if idx == 0:
        return 0
    if idx == len(sorted_array):
        return len(sorted_array) - 1

    # Check the closest of the two neighbors
    prev_idx = idx - 1
    if abs(sorted_array[idx] - target) < abs(sorted_array[prev_idx] - target):
        return idx
    else:
        return prev_idx
        
class InteractiveLineBuilder:
    def __init__(self, echoviewer, filepath = None, axnr=-1):
        with echoviewer.output:
            self.echoviewer = echoviewer
            echoviewer.callback_view = self.redraw_line
            self.ax = echoviewer.axes[axnr]
            self.fig = echoviewer.fig
            self.canvas = self.fig.canvas
            self.line, = self.ax.plot([], [], marker='o', linestyle='-', color='red', picker=5, zorder=1000000000)
            self.timestamps = []
            self.xs = []
            self.ys = []
            self.selected_point = None
            self.dragging_point = False

            self.filepath = filepath
            if self.filepath is not None:
                if os.path.exists(filepath):
                    self.timestamps, self.ys = pickle.load(open(filepath, 'rb'))
                    self.timestamps = [t.timestamp() if isinstance(t,datetime.datetime) else t for t in self.timestamps]
                    self.reinit_xs()
                    self.update_line()

                    if not len(self.xs) == len(self.ys) == len(self.timestamps):
                        raise RuntimeError(f'ERROR opening {filepath}! [{len(self.xs)} ?= {len(self.ys)} ?= {len(self.timestamps)}]')
        
            self.cids = {}
            self.cids['click'] = self.canvas.mpl_connect('button_press_event', self.on_click)
            self.cids['release'] = self.canvas.mpl_connect('button_release_event', self.on_release)
            self.cids['motion'] = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.cids['keypress'] = self.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.cids['draw'] = self.canvas.mpl_connect('draw_event', self.on_draw)

    def to_file(self, filepath = None):
        if filepath is None:
            filepath = self.filepath
        if len(self.xs) > 0:
            self.reinit_xs()
            if len(self.xs) == len(self.ys) == len(self.timestamps):
                print(f'dumping to {filepath}')
                pickle.dump((self.timestamps, self.ys), open(filepath,'wb'))

    def __del__(self):
        with self.echoviewer.output:
            try:
                for cid in self.cids.values():
                    self.canvas.mpl_disconnect(cid)
                self.line.remove()
            except:
                pass
    
    def on_draw(self, event = 0):
        with self.echoviewer.output:
            """Callback for draws."""
            #self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            #self.ax.draw_artist(self.line)
    
            pass
            # self.line.remove()
            # self.line, = self.ax.plot(self.xs,self.ys, marker='o', linestyle='-', color='red', picker=5)
            # fig.canvas.draw_idle()
    
    def update_line(self):
        with self.echoviewer.output:
            self.reinit_xs()
            
            # sort the points
            sort_args = np.argsort(self.xs)
            self.xs = list(np.array(self.xs)[sort_args])
            self.ys = list(np.array(self.ys)[sort_args])
            self.selected_point = sort_args[self.selected_point]
            self.line.set_data(self.xs, self.ys)
            self.canvas.draw_idle()
            
            # self.canvas.restore_region(self.background)
            # self.ax.draw_artist(self.line)
            # self.canvas.blit(self.ax.bbox)

    def reinit_xs(self):
        with self.echoviewer.output:
            if len(self.xs) < len(self.timestamps):
                self.xs = mdates.date2num([datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t in self.timestamps])
            elif len(self.timestamps) < len(self.xs):
                self.timestamps = [t.timestamp() for t in mdates.num2date(self.xs)]

    def redraw_line(self):
        with self.echoviewer.output:
            self.reinit_xs()
            try:
                self.line.remove()
            except:
                pass
            xlim=self.ax.get_xlim()
            ylim=self.ax.get_ylim()
            self.line, = self.ax.plot(self.xs,self.ys, marker='o', linestyle='-', color='red', picker=5)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.canvas.draw_idle()
                
    def on_click(self, event):
        with self.echoviewer.output:
            if event.inaxes != self.ax:
                return
            # Check if a point is clicked
            contains, attrd = self.line.contains(event)
            if contains:
                ind = attrd['ind'][0]
                self.selected_point = ind
                self.dragging_point = True

    def on_release(self, event):
        with self.echoviewer.output:
            if self.dragging_point:
                #self.selected_point = None
                self.dragging_point = False

    def on_motion(self, event):
        with self.echoviewer.output:
            if not self.dragging_point or self.selected_point is None:
                return
            if event.inaxes != self.ax:
                return
            # Update the position of the selected point
            self.xs[self.selected_point] = event.xdata
            self.ys[self.selected_point] = event.ydata
            
            self.update_line()

    def on_key_press(self, event):
        with self.echoviewer.output:
            if event.inaxes != self.ax:
                return
            self.reinit_xs()
            match event.key:
                case 'a':
                    # Add a point at the current cursor position
                    self.xs.append(event.xdata)
                    self.ys.append(event.ydata)
                    self.update_line()
                case 'd':
                    # Delete the point closest to the cursor
                    if not self.xs:
                        return
                    # xdata = np.array(self.xs)
                    # ydata = np.array(self.ys)
                    # distances = np.hypot(xdata - event.xdata, ydata - event.ydata)
                    # min_index = np.argmin(np.abs(distances))
                    # Check if a point is clicked
                    contains, attrd = self.line.contains(event)
                    if contains:
                        ind = attrd['ind'][0]
                        del self.xs[ind]
                        del self.ys[ind]
                        del self.timestamps[ind]
                        self.update_line()
                case 'f':
                    # move the closest point to the cursor
                    if not self.xs:
                        return
                        
                    ind = find_closest_index(self.xs, event.xdata)
                    self.xs[ind] = event.xdata
                    self.ys[ind] = event.ydata
                    self.timestamps[ind] = mdates.num2date(self.xs[ind]).timestamp()
                    self.update_line()
                case 'u':
                    self.redraw_line()
                
        