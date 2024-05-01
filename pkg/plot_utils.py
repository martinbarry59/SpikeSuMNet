import torch
import matplotlib.pyplot as plt
import save_utils
import plotly.graph_objects as go
def plotly_plot(datas,legend,file = None, plot = False):
    """
    Plot data using Plotly library.

    Args:
        datas (list): A list of data arrays to be plotted.
        legend (list): A list of two strings representing the x-axis and y-axis labels.
        file (str, optional): The file path to save the plot as an image. Defaults to None.
        plot (bool, optional): Whether to display the plot. Defaults to False.
    """    
    fig = go.Figure()
    for data in datas:
        x = torch.arange(len(data))
        fig.add_trace(go.Scatter(x=x, y=data,mode='lines'))
    layout = go.Layout()
    fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=go.layout.Legend(

        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=18,
            color="black"
        ),
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text=legend[0],
            font=dict(
            family="sans-serif",
            size=18,

            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text=legend[1],
            font=dict(
                family="sans-serif",
                size=18,
            )
        )
    )
)
    if file:
        save_utils.save(data = fig, file = file, type_= 'fig_ly')
    if plot:
        fig.show()

def nice_print(params, file = None):
    if file is not None:
            file = open(file, "a")
    else:
        file = None
    print('---Network parameters---',file = file)
    
    for key, values in sorted(params.items(),key = lambda x: (x[0].lower(), x[1])):
        if isinstance(values,dict):
            print(key,':')
            for kkey, vvalues in sorted(values.items(),key = lambda x: (x[0].lower(), x[1])):
                print('   {0}: {1}'.format(kkey,vvalues),file = file)
        else:
            print('{0}: {1}'.format(key,values),file = file)
        print('------------------------',file = file)

def print_values(tensor,name,percentage = 1):
    """
    Tensor:  Results tensor of simulations to print

    name: accuracy printed
    
    percentage: 1 if useing percentage 0 otherwise
    
    This function should only be called by print criteria
    """
    n_digits = 2
    scaling = 1 + 99 * percentage
    mean = scaling * torch.mean(tensor)
    std = torch.std(tensor)/torch.sqrt(torch.tensor([len(tensor)])) * scaling
    print(name,mean.item(),'$\pm$', std.item())

def print_criteria(path):
    """
    Path:  path to the file where the criteria are saved
    
    """
    criteria = torch.load(path,map_location=torch.device('cpu'))

    values = torch.Tensor([1-criterium['stop'] for criterium in criteria])
    print_values(values,'mean success: ')
    
    values = torch.Tensor([(len(criterium['observed_mazes'])== len(criterium['activated_memory'])) for criterium in criteria ])
    print_values(values,'mean memory usage: ')
    
    values = torch.Tensor([(len(criterium['observed_mazes'])== len(criterium['activated_memory']))* (1-criterium['stop']) for criterium in criteria])
    print_values(values,'mean  total success: ')
    
    values = torch.Tensor([criterium['before_detecting_cp']  for criterium in criteria if 1*criterium['stop']==0])
    print_values(values,'mean detection time: ',percentage = 0)


def plot_spike_train(spikes_mat,neurons,n_memory,title = None,directory = None):
    """
    Plots the spike train for a given set of spikes.

    Parameters:
    spikes_mat (torch.Tensor): The spike matrix of shape (n_neurons, n_time_steps).
    neurons (int): The number of neurons to plot.
    n_memory (int): The number of memory context.
    title (str, optional): The title of the plot. Defaults to None.
    directory (str, optional): The directory to save the plot. Defaults to None.

    Returns:
    None
    """
    linewidth = 0.8
    
    event_times, event_ids = torch.where(spikes_mat.T)
    event_times = event_times[:-1]
    event_ids = event_ids[:-1]

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
    if title is not None:
        plt.title(title)
    n_t, n_n = spikes_mat.T.shape
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in')
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    if n_memory > 1:
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelright=True, labelleft=True)

    # the scatter plot:
    for n, t in zip(event_ids,event_times):
        ax_scatter.vlines(t, n - 0.25, n + 0.25, linewidth=linewidth)

    # now determine nice limits by hand:
    binwidth = 0.25
    ax_scatter.set_ylim([0 , n_n])
    ax_scatter.set_xlim([0, n_t ])
    spikes = spikes_mat.reshape(n_memory,neurons,-1)
    sample = torch.sum(spikes,dim =(0,1))
    ax_histx.bar(torch.arange(spikes.shape[-1]), sample / torch.sum(sample))
    sample = torch.sum(spikes,dim =(1,2))
    if n_memory > 1:
        ax_histy.barh(torch.arange(n_memory), sample / torch.sum(sample))
        ax_histy.yaxis.set_label_position("right")
        ax_histy.yaxis.tick_right()
    plt.ylim([-0.45, n_memory -1 + .45])
    plt.show()

def windowed_mean(x,y,window = 0.01,minmax = []):
    """
    x: x axis values of the array to average
    
    y: y axis values of the array to average
    
    window: length of average
    
    return averaged x, averaged y
    
    """
    import scipy.stats
    if len(minmax) > 0:
        steps = (minmax[1]-minmax[0]) / window
        print(steps)
    else:
        steps = torch.max(x) / window
    mean_x = torch.zeros(window)
    mean_y = torch.zeros(window)
    std_y = torch.zeros(window)
    start_value = 0
    start_idx = 0
    for i in range(window):        
        indices = (start_value<x) * (x<(start_value + steps))
        end = len(x[indices])
        m, se = torch.mean(y[indices]), scipy.stats.sem(y[indices].cpu().numpy())
        h = se * scipy.stats.t.ppf((1 + 0.9) / 2., end-1)
        mean_y[i] = torch.mean(y[indices])
        std_y[i] = h#torch.std(y[indices]) 
        mean_x[i] = torch.mean(x[indices])
        start_value += steps
    return mean_x, mean_y, std_y
