import os
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plot_utils
def save(data=None, file=None, type_=None):
    """
    Saving file plotly, matplotlib or npy

    data: the variables to save

    file: relative path of the file, will save it in a results folder

    type_: give the type of the file to save (fig_ly, fig,data)

    """
    directory = "../results/" + type_ + "/" + file

    if not os.path.exists("/".join(directory.split("/")[:-1])):
        os.makedirs("/".join(directory.split("/")[:-1]))

    if type_ == "fig_ly":
        plotly.offline.plot(data, filename=directory + ".html")
    elif type_ == "fig":
        plt.savefig(directory + ".eps")
    elif type_ == "data":
        torch.save(directory, data)
    else:
        print("NO FILE SPECIFIED: NOT SAVED")

def save_weight_updates(model_info,n_memory,save):
    dic = {}
    n = 1
    N = int(1500 * model_info['effective update_neg'].shape[0]/len(model_info['error']))
    for key in ['','_neg','_pos']:
        x = model_info['effective update'+key][N:]
        y = model_info['prediction error'+key][N:]
        sortederror, indices = torch.sort(y)
        sortedupdate = x[indices]
        max_N = n * (x.shape[0] // n)
        sortederror = sortederror[:max_N].view(-1,n)
        sortedupdate = sortedupdate[:max_N].view(-1,n)
        dic['absolute error'+key],dic['effective update'+key] = plot_utils.windowed_mean(sortederror.to('cpu'),sortedupdate.to('cpu'))
        test = torch.argmax(dic['effective update'+key])
        with open('{0}_memory_full_sim_{1}.pkl'.format(n_memory,save), 'wb') as f:
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)