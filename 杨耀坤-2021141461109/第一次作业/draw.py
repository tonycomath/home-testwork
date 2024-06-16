import numpy as np
import matplotlib.pyplot as plt

def heatmap(x = None, y = None, value = None, title = None, save_name = None):
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    if x is None:
        x = farmers
    if y is None:
        y = vegetables
    if value is None:
        value = harvest
    if title is None:
        title = "Harvest of local farmers (in tons/year)"

    # plt.figure(figsize=(16,16))
    fig, ax = plt.subplots(figsize=(12, 12))
    # color style: cmap
    # 详细的颜色选择请见：cmap.jpg
    im = ax.imshow(value, cmap = "YlGn")

    # 设置坐标轴显示内容
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.tick_params(axis = 'x', labelsize = 12)

    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(y)
    ax.tick_params(axis = 'y', labelsize = 12)

    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right", rotation_mode = "anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, str(value[i, j]), ha = "center", va = "center", color = "black", fontsize = 12)  # or white

    ax.set_title(title, fontsize = 15)

    cbar = plt.colorbar(im, ax = ax)
    cbar.ax.tick_params(labelsize = 14) 
    cbar.set_label("Label", fontsize = 20) 

    plt.subplots_adjust(left = 0.10, right = 0.99, top = 0.90, bottom = 0.12)
    plt.savefig(save_name + '.png')


def lineBarChart(y_real, y_real_normalized, y_predict, save_name):
    fig, ax = plt.subplots(figsize=(20, 12))

    x = np.linspace(1, 250, len(y_real))


    # the size of the dots
    MARK_SIZE = 3

    plt.plot(x, y_real,"--",markersize = MARK_SIZE,color = 'red',label = 'Real data')
    plt.plot(x, y_real_normalized, marker = 'o', markersize = MARK_SIZE, linestyle = '-', color = 'orange', label = 'Real and normalized data')
    plt.bar(x, y_predict, color='#8AC7F2', alpha = 0.7, label = 'Predited data')
   

    # limit the scope of x
    plt.xlim(0, 250)


    # # hide axises
    # plt.axis('off')

    # # hide the number in x axis and y axis
    # plt.xticks([])  
    # plt.yticks([]) 

    # show grid
    # plt.grid(True)

    # adjust the width of blank
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.12)

    plt.xlabel('Day',fontsize = 15)
    plt.ylabel('Share price',fontsize = 15)
    plt.title('Comparison chart between predicted stock prices and actual stock prices in 2019',fontsize = 15)
    # 设置横坐标刻度字体大小
    # plt.xticks(fontsize=12)


    # 设置纵坐标刻度字体大小
    # plt.yticks(fontsize=12)

    # # 是否使用科学技术法
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend(loc = 'upper left', fontsize= 9)

    plt.savefig(save_name + '.png')
