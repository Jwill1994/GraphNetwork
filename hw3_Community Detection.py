import networkx as nx
import logging
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
#https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
import wikipediaapi
import collections
import matplotlib.pyplot as plt
from heapq import nsmallest
import json 
import itertools 
import numpy.linalg
def getWikiText(page_py):
    total_text = ""
    for section in page_py.sections:
        if section.title not in ["See also","References","Sources","Further reading","External links","Notes"]:
            section_py = page_py.section_by_title(section.title)
            if section_py is not None:
                total_text+=section_py.title+'. ' #title 구분 
                total_text+=section_py.text
                if section_py.sections != None :
                    for section2 in section_py.sections:
                        total_text+=section2.title+'. '
                        total_text+=section2.text
                        if section2.sections != None:
                            for section3 in section2.sections:
                                total_text+=section3.title+'. '
                                total_text+=section3.text
            else:
                print("Section does not exist.")
    return total_text 

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

def mean_degree(G):
    return 2*len(G.edges()) / len(G.nodes())

def get_density(G):
    return 2*len(G.edges()) / len(G.nodes()) / (len(G.nodes())-1)

def second_smallest(numbers):
    return nsmallest(2, numbers)[-1]

def adjOne(G,word_tokens):
    for i in range(len(word_tokens)) :
        if i-1 >= 0 :
            G.add_edge(word_tokens[i-1],word_tokens[i])
        if i+1 < len(word_tokens):
            G.add_edge(word_tokens[i],word_tokens[i+1])
    return G   

def adjTwo(G,word_tokens):
    for i in range(len(word_tokens)) :
        if i-2 >= 0 :
            G.add_edge(word_tokens[i-2],word_tokens[i])
        if i+2 < len(word_tokens):
            G.add_edge(word_tokens[i],word_tokens[i+2])
    return G

def undir_completeG(G,word_tokens):
    G.add_edges_from(itertools.combinations(word_tokens, 2))
    return G

def dir_completeG(G,word_tokens):
    G.add_edges_from(itertools.permutations(word_tokens, 2))
    return G

def main_undirG():
    #Undirected Graph 
    G = nx.Graph(weight=0) 

    for t in range(len(sent_tokens)):
        word_tokens = RegexpTokenizer(r'\w+').tokenize(sent_tokens[t])
        #capital to lower case 
        word_tokens = [ele.lower() for ele in word_tokens]
        #print(word_tokens) # 테스트용 문장 
        G.add_nodes_from(word_tokens)
        #graph generation for each sentence's words 

        
        #G = nx.complete_graph(len(word_tokens)) #all pairs of nodes get edges 

                ######## edge 연결 시도 다른 방식 
        #n1=n2=1
        G = adjOne(G,word_tokens)
        #n1=n1=2
        #G = adjTwo(G,word_tokens)
        #all connected 
        #G = undir_completeG(G,word_tokens)
    
    ###########
    '''
    print(G.nodes())
    print(len(G.nodes()))
    print(G.edges())
    '''
    ###########
    
    
    
    print(G.degree())
    
    mean_deg = mean_degree(G)
    print('mean_degree: ',mean_deg)
    
    density = get_density(G)
    print('density: ',density,'\n')
    
    #Graph Laplacian 
    #laplacian_matrix = nx.normalized_laplacian_matrix(G)
    #print(laplacian_matrix)
    #eigens = numpy.linalg.eigvals(laplacian_matrix.A)
    #print('minimum laplacian eigenvalue: ',min(eigens))
    #second smallest eigenvalue
    #second_smallest_eigen = second_smallest(eigens)
    #print('seconds smallest eigenvalue: ', second_smallest_eigen,'\n')
    ###
    
    laplacian_eigenvalues = nx.linalg.spectrum.laplacian_spectrum(G, weight=0)
    print('minimum laplacian eigenvalue: ',min(laplacian_eigenvalues))
    #second smallest eigenvalue
    second_smallest_eigen = second_smallest(laplacian_eigenvalues)
    print('seconds smallest eigenvalue: ', second_smallest_eigen,'\n')
    
    #clustering coefficient
    cluster_coeff = nx.algorithms.cluster.clustering(G)

    avg_cluster_coeff = nx.algorithms.cluster.average_clustering(G)
    print('average clustering coefficients: ', avg_cluster_coeff)

    degHistogram(G)
    #Draw undirected network 
    nx.draw(G,with_label=True)
    plt.show()
    
def main_dirG():
    #Directed Graph
    G = nx.DiGraph(weight=0)
    for t in range(len(sent_tokens)):
        word_tokens = RegexpTokenizer(r'\w+').tokenize(sent_tokens[t])
        #capital to lower case 
        word_tokens = [ele.lower() for ele in word_tokens]
        #print(word_tokens) # 테스트용 문장 
        G.add_nodes_from(word_tokens)
        #graph generation for each sentence's words 

        
        #G = nx.complete_graph(len(word_tokens)) #all pairs of nodes get edges 

        ######## edge 연결 시도 다른 방식 - 1개 선택! 
        #n1=n2=1
        #G = adjOne(G,word_tokens)
        #n1=n1=2
        #G = adjTwo(G,word_tokens)
        #all connected 
        G = dir_completeG(G,word_tokens)
        


    ###########
    #print(len(G.nodes()))
    #print(G.edges())
    #degHistogram(G)
    ###########

    #Eigenvector Centrality 
    eigen_central = nx.eigenvector_centrality(G)

    #print('eigenvector centrality: ',eigen_central)
   
    eigen_key_max = max(eigen_central.keys(),key=(lambda k: eigen_central[k]))
    eigen_key_min = min(eigen_central.keys(),key=(lambda k: eigen_central[k]))
    print('max eigen centrality: ', eigen_key_max, eigen_central[eigen_key_max])
    print('min eigen centrality: ', eigen_key_min, eigen_central[eigen_key_min] )
    
    with open('eigenvector_centrality.txt','w+') as file:
        file.write(json.dumps(eigen_central))
    file.close()
   
    #############################################################################
    #Katz Centrality ; https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html#networkx.algorithms.centrality.katz_centrality
    katz_central = nx.katz_centrality(G,alpha=0.001,max_iter=20000) #default alpha = 0.1 -> not converge! so, used 0.01 / for complete graph: alpha=0.001
    #print(katz_central)
   
    katz_key_max = max(katz_central.keys(),key=(lambda k: katz_central[k]))
    katz_key_min = min(katz_central.keys(),key=(lambda k: katz_central[k]))
    print('max katz centrality: ', katz_key_max, katz_central[katz_key_max])
    print('min katz centrality: ', katz_key_min, katz_central[katz_key_min] )
    
    with open('katz_centrality.txt','w+') as file:
        file.write(json.dumps(katz_central))
    file.close()
    
    #############################################################################

    #Page Rank
    page_rank = nx.pagerank(G)
    #print(page_rank)

    pr_key_max = max(page_rank.keys(),key=(lambda k: page_rank[k]))
    pr_key_min = min(page_rank.keys(),key=(lambda k: page_rank[k]))
    print('max page rank: ', pr_key_max, page_rank[pr_key_max])
    print('min page rank: ', pr_key_min, page_rank[pr_key_min] )
    
    with open('pagerank.txt','w+') as file:
        file.write(json.dumps(page_rank))
    file.close()

    #############################################################################

    #Closeness Centrality
    closeness_central = nx.closeness_centrality(G)
    #print(closeness_central)
    
    close_key_max = max(closeness_central.keys(),key=(lambda k: closeness_central[k]))
    close_key_min = min(closeness_central.keys(),key=(lambda k: closeness_central[k]))
    print('max closeness centrality: ', close_key_max, closeness_central[close_key_max])
    print('min closeness centrality: ', close_key_min, closeness_central[close_key_min] )
  
    with open('closeness_centrality.txt','w+') as file:
        file.write(json.dumps(closeness_central))
    file.close()
  
    #############################################################################
    #Betweenness Centrality
    betweenness_central = nx.betweenness_centrality(G)
    #print(betweenness_central)
    bc_key_max = max(betweenness_central.keys(),key=(lambda k: betweenness_central[k]))
    bc_key_min = min(betweenness_central.keys(),key=(lambda k: betweenness_central[k]))
    print('max betweenness centrality: ', bc_key_max, betweenness_central[bc_key_max])
    print('min betweenness centrality: ', bc_key_min, betweenness_central[bc_key_min] )

    with open('betweenness_centrality.txt','w+') as file:
        file.write(json.dumps(betweenness_central))
    file.close()
    
    '''
    #Draw directed network 
    nx.draw(G,with_label=True)
    plt.show()
    '''
import json 
def kernighan_lin(G):
    kl = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(G)
    print(kl)
    print('len com1,com2',len(kl[0]),len(kl[1]))


def girvan_newman(G):
    gn = nx.algorithms.community.centrality.girvan_newman(G)
    print(gn)
    gn_sec = tuple(sorted(c) for c in next(gn))
    print(gn_sec)
    print('len com1,com2',len(gn_sec[0]),len(gn_sec[1]))


from sklearn.cluster import SpectralClustering
import numpy as np
from itertools import groupby
def spectral_clustering(G):
    #degHistogram(G)
    adj_mat = nx.to_numpy_matrix(G)
    print(adj_mat.shape)
    clustering = SpectralClustering(n_clusters=2,affinity="precomputed",random_state=0).fit(adj_mat)
    #print(clustering.labels_)   
    split = [list(grp) for k, grp in groupby(clustering.labels_)]
    print(split)
    print('len com1,com2',len(split[0]),len(split[1]))
    return 0

#https://mons1220.tistory.com/130
import community as lvcm
import matplotlib.cm as cm
from collections import defaultdict
def louvain(G):
    dendo = lvcm.generate_dendrogram(graph=G, weight='weight', resolution=7., randomize=True)
    
    partition = lvcm.partition_at_level(dendo, len(dendo)-1)
    #a = set(partition.values())
    print(partition)
    
    #partition = community_louvain.best_partition(G)
    #print(set(partition.values()))
    #print(len(set(partition.values())))
    out = defaultdict(list)
    for k, v in partition.items():
        out[v].append(k)

    print(out)
    

if __name__=='__main__':

    logging.basicConfig(level=logging.INFO)
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page('summer') # topic : love & friendship 
    
    print("Page - Exists: %s" % page_py.exists())
    print("Page - Id: %s" % page_py.pageid)
    print("Page - Title: %s" % page_py.title)
    
    #total text into sentences
    total_text = getWikiText(page_py)

    #each sentence into words & convert into nodes of grpah
    nltk.download('punkt')
    sent_tokens = sent_tokenize(total_text)
    #print(sent_tokens)
    G = nx.Graph(weight=0) 
    for t in range(len(sent_tokens)):
        word_tokens = RegexpTokenizer(r'\w+').tokenize(sent_tokens[t])
        #capital to lower case 
        word_tokens = [ele.lower() for ele in word_tokens]
        #print(word_tokens) # 테스트용 문장 
        G.add_nodes_from(word_tokens)
        #graph generation for each sentence's words 
        #G = nx.complete_graph(len(word_tokens)) #all pairs of nodes get edges 

                ######## edge 연결 시도 다른 방식 
        #n1=n2=1
        G = adjOne(G,word_tokens)
   
    print('######kernighan_lin#####')
    kernighan_lin(G)
    print('######girvan_newman#####')
    girvan_newman(G)
    print('######spectral_clustering#####')
    spectral_clustering(G)
    print('######louvain#####')
    louvain(G)
    #main_undirG()
    #main_dirG()

   
    ##################################################################

