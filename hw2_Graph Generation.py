import networkx as nx
import collections
import matplotlib.pyplot as plt
import powerlaw 
import numpy as np
from math import *

def degHistogram(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(G)
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()
    return 0 

def Erdos_Renyi():
    G = nx.erdos_renyi_graph(1000,0.5) #0.5 or 0.8 selected 
    print('#connected component:', nx.number_connected_components(G))
    cluster_coeff = nx.algorithms.cluster.clustering(G)
    avg_cluster_coeff = nx.algorithms.cluster.average_clustering(G)
    print('average clustering coefficients: ', avg_cluster_coeff)
    diameter = nx.diameter(G)
    print('diameter',diameter)
    degHistogram(G)
    #Draw undirected network 
    nx.draw(G,with_label=True)
    plt.show()
    return 0 

def Barabasi_Albert(start, width, role_start=0, m=4):
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    #get degree dist. & power exponenet
    #print(list(nx.isolates(graph)))
    #clustering coeffi & diameter
    graph.remove_nodes_from(list(nx.isolates(graph)))
    con_graph = graph #cuz starting 4 nodes are not connected, diameter calc error occurs
    cluster_coeff = nx.algorithms.cluster.clustering(con_graph)
    avg_cluster_coeff = nx.algorithms.cluster.average_clustering(con_graph)
    print('average clustering coefficients: ', avg_cluster_coeff)
    diameter = nx.diameter(con_graph)
    print('diameter',diameter)
   
    #plot graph
    degrees = sorted([d for n, d in con_graph.degree()], reverse=True)
    fit = powerlaw.Fit(degrees,xmin=1)
    fig2 = fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig2)
    #print('power law exponent: ', power_fit.power_law.alpha)    
    #degHistogram(con_graph)
    #nx.draw(graph, with_labels=True) 
    plt.show()
    
    return graph, roles 

def Watts_Strogatz():
    G = nx.newman_watts_strogatz_graph(1000,4,0.3) #different p :0.0, 0.3
    cluster_coeff = nx.algorithms.cluster.clustering(G)
    avg_cluster_coeff = nx.algorithms.cluster.average_clustering(G)
    print('average clustering coefficients: ', avg_cluster_coeff)
    diameter = nx.diameter(G)
    print('diameter',diameter)
    degHistogram(G)
    #Draw undirected network 
    nx.draw(G,with_label=True)
    plt.show()
    return 0 
if __name__=='__main__':
    #Erdos_Renyi()
    #Eba_graph, ba_roles = Barabasi_Albert(4,96)
    #Eba_graph2, ba_roles2 = Barabasi_Albert(4,996)
    #Eba_graph3, ba_roles3 = Barabasi_Albert(4,9996)
    Watts_Strogatz()
