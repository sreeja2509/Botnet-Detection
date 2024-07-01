from tkinter import *
import tkinter
import pandas as pd
import networkx as nx
from tqdm import tqdm
import collections
from multiprocessing import Pool
import time
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tkinter import filedialog
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import ttk

main = tkinter.Tk()
main.title("BotChase: Graph-Based Bot Detection Using Machine Learning") #designing main screen
main.geometry("1300x1200")

global filename
global ctuDataset
global dg, dict_nodes_list, ip_nodes_list
global between_ness, clustering, alpha_cent
global cm


def upload():
    global filename
    global ctuDataset
    filename = filedialog.askopenfilename(initialdir="CTU-13-Dataset")
    text.delete('1.0', END)
    
    ctuDataset = pd.read_csv(filename)
    text.insert(END,'Dataset size  : \n\n')
    text.insert(END,'Total Rows    : '+str(ctuDataset.shape[0])+"\n")
    text.insert(END,'Total Columns : '+str(ctuDataset.shape[1])+"\n\n")
    text.insert(END,'Dataset Samples\n\n')
    text.insert(END,str(ctuDataset.head())+"\n\n")
    
def kmeans():
    global ctuDataset
    text.delete('1.0', END)
    ctu_bot = ctuDataset[ctuDataset['Label'].str.contains('Botnet')]
    ctu_benign = ctuDataset[~ctuDataset['Label'].str.contains('Botnet')]
    ctu_benign = ctu_benign.sample(n=ctu_bot.shape[0])
    X = ctuDataset[['Dur', 'TotPkts','TotBytes','SrcBytes']]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    text.insert(END,'Dataset size before removing benign records\n\n')
    text.insert(END,'Total Rows    : '+str(ctuDataset.shape[0])+"\n")
    text.insert(END,'Total Columns : '+str(ctuDataset.shape[1])+"\n\n")
    ctuDataset = pd.concat([ctu_bot, ctu_benign])
    text.insert(END,'Dataset size after removing benign records\n\n')
    text.insert(END,'Total Rows    : '+str(ctuDataset.shape[0])+"\n")
    text.insert(END,'Total Columns : '+str(ctuDataset.shape[1])+"\n\n")
    

#get nodes chunk
def nodeOrder(nodes, order):
    node_order = iter(nodes)
    while nodes:
        name = tuple(itertools.islice(node_order, order))
        if not name:
            return
        yield name

#return betweenness
def betweenmap(Graph_normalized_weight_sources_tuple):
    return nx.betweenness_centrality_source(*Graph_normalized_weight_sources_tuple)


def betweennessCentrality(Graphs, processes=None):
    p = Pool(processes=processes)
    nodes_divisor = len(p._pool) * 2
    nodes_chunks = list(nodeOrder(Graphs.nodes(), int(Graphs.order() / nodes_divisor)))
    numb_chunks = len(nodes_chunks)
    between_sc = p.map(betweenmap,
                  zip([Graphs] * numb_chunks,
                      [True] * numb_chunks,
                      [True] * numb_chunks,
                      nodes_chunks))

    # Reduce the partial solutions
    between_c = between_sc[0]
    for between in between_sc[1:]:
        for n in between:
            between_c[n] += between[n]
    return between_c


def graphTransform():
    global ctuDataset
    global dg, dict_nodes_list, ip_nodes_list
    global between_ness, clustering, alpha_cent
    text.delete('1.0', END)

    duplicates_RowsDF = ctuDataset[ctuDataset.duplicated(['SrcAddr', 'DstAddr'])]
    duplicates_Rows = list(duplicates_RowsDF.index.values)
    ctuDataset = ctuDataset.drop(duplicates_Rows)

    ctu_temp = pd.merge(ctuDataset, duplicates_RowsDF, how='inner', on=['SrcAddr', 'DstAddr'])

    sum_columns = ctu_temp['TotPkts_x'] + ctu_temp['TotPkts_y']
    ctu_temp['TotPkts'] = sum_columns
    ctuDataset = ctu_temp[['SrcAddr', 'DstAddr', 'TotPkts']]

    #graph building starts here
    dg = nx.DiGraph()

    #deleting duplicate ip address to have only one vertex for each IP
    source_address_list = list(ctuDataset['SrcAddr'])
    source_address_list.extend(list(ctuDataset['DstAddr']))
    ip_nodes_list = list(set(source_address_list))
    dg.add_nodes_from(ip_nodes_list)

    dict_nodes_list = collections.defaultdict(dict)

    for index, row in tqdm(ctuDataset.iterrows(), total=ctuDataset.shape[0], desc="{Generating bot graph}"):
        e = (row['SrcAddr'], row['DstAddr'], row['TotPkts'])
    
        if dg.has_edge(*e[:2]):
            edgeData = dg.get_edge_data(*e)
            weight = edgeData['weight']
            dg.add_weighted_edges_from([(row['SrcAddr'], row['DstAddr'], row['TotPkts'] + weight)])
            dict_nodes_list[row['SrcAddr']]['out-degree-weight'] += row['TotPkts']
            dict_nodes_list[row['DstAddr']]['in-degree-weight'] += row['TotPkts']
            dict_nodes_list[row['SrcAddr']]['out-degree'] += 1
            dict_nodes_list[row['DstAddr']]['in-degree'] += 1
        else:
            dg.add_weighted_edges_from([(row['SrcAddr'], row['DstAddr'], row['TotPkts'])])
            dict_nodes_list[row['SrcAddr']]['out-degree-weight'] = row['TotPkts']
            dict_nodes_list[row['SrcAddr']]['in-degree-weight'] = 0
            dict_nodes_list[row['DstAddr']]['in-degree-weight'] = row['TotPkts']
            dict_nodes_list[row['DstAddr']]['out-degree-weight'] = 0
            dict_nodes_list[row['SrcAddr']]['out-degree'] = 1
            dict_nodes_list[row['SrcAddr']]['in-degree'] = 0
            dict_nodes_list[row['DstAddr']]['in-degree'] = 1
            dict_nodes_list[row['DstAddr']]['out-degree'] = 0

    text.insert(END,'Number of nodes: ' + str(nx.number_of_nodes(dg))+"\n\n")
    text.insert(END,'Number of edges: ' + str(nx.number_of_edges(dg))+"\n\n")
    text.insert(END,'Network graph created\n\n')

    text.insert(END,'Betweeness centrality time calculation for all ip address or nodes\n\n')
    start_time = time.time()
    between_ness = betweennessCentrality(dg, 2)
    text.insert(END,"Execution Time : "+str((time.time() - start_time))+"\n\n")

    #clustering time
    text.insert(END,'Clustering time calculation\n\n')
    start_time = time.time()
    clustering = nx.clustering(dg, weight='weight')
    text.insert(END,"Clustering Time : "+str((time.time() - start_time))+"\n\n")

    #Alpha Centrality time calculation
    text.insert(END,'Alpha Centrality time calculation')
    start_time = time.time()
    alpha_cent = nx.algorithms.centrality.katz_centrality_numpy(dg, weight='weight')
    text.insert(END,"Alpha Centrality Time : "+str((time.time() - start_time))+"\n\n")

    #bot_list = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    for i in dict_nodes_list:
        if dict_nodes_list[i]['out-degree'] > 10:
            print(dict_nodes_list[i]['out-degree'])
            dict_nodes_list[i]['bot'] = 1
        else:
            dict_nodes_list[i]['bot'] = 0

        dict_nodes_list[i]['bc'] = between_ness[i]
        dict_nodes_list[i]['lcc'] = clustering[i]
        dict_nodes_list[i]['ac'] = alpha_cent[i]
      
        
    
def featuresNormalization():
    text.delete('1.0', END)
    for nodes in tqdm(ip_nodes_list, desc="{features Normalization module}"):
        counter = 0 #N is a counter for the total neighbors of each node with D=2
        in_degree = 0 #s is the total sum of all the features of the node's neighbors
        out_degree = 0
        in_degree_weight = 0
        out_degree_weight = 0
        s_between = 0
        s_clustering = 0
        s_alpha = 0
    
        for neighbors in dg.neighbors(nodes):
            in_degree += dict_nodes_list[neighbors]['in-degree'] 
            out_degree += dict_nodes_list[neighbors]['out-degree']
            in_degree_weight += dict_nodes_list[neighbors]['in-degree-weight']
            out_degree_weight += dict_nodes_list[neighbors]['out-degree-weight']
            s_between += between_ness[neighbors]
            s_clustering += clustering[neighbors]
            s_alpha += alpha_cent[neighbors]
            counter += 1
        
            for n in dg.neighbors(neighbors):
                in_degree += dict_nodes_list[neighbors]['in-degree'] 
                out_degree += dict_nodes_list[neighbors]['out-degree']
                in_degree_weight += dict_nodes_list[neighbors]['in-degree-weight']
                out_degree_weight += dict_nodes_list[neighbors]['out-degree-weight']
                s_between += between_ness[neighbors]
                s_clustering += clustering[neighbors]
                s_alpha += alpha_cent[neighbors]
                counter += 1
   
        if counter != 0:
            if in_degree != 0:
                dict_nodes_list[nodes]['in-degree'] = dict_nodes_list[nodes]['in-degree'] / (in_degree/counter)
        
            if out_degree != 0:
                dict_nodes_list[nodes]['out-degree'] = dict_nodes_list[nodes]['out-degree'] / (out_degree/counter)
        
            if in_degree_weight != 0:
                dict_nodes_list[nodes]['in-degree-weight'] = dict_nodes_list[nodes]['in-degree-weight'] / (in_degree_weight/counter)
        
            if out_degree_weight != 0:
                dict_nodes_list[nodes]['out-degree-weight'] = dict_nodes_list[nodes]['out-degree-weight'] / (out_degree_weight/counter)
        
            if s_between != 0:
                dict_nodes_list[nodes]['bc'] = dict_nodes_list[nodes]['bc'] / (s_between/counter)
        
            if s_clustering != 0:
                dict_nodes_list[nodes]['lcc'] = dict_nodes_list[nodes]['lcc'] / (s_clustering/counter)
  10      
            if s_alpha != 0:
                dict_nodes_list[nodes]['ac'] = dict_nodes_list[nodes]['ac'] / (s_alpha/counter)
        
        
    text.insert(END,'Normalizing features process completed & below are some sample records\n\n')        
    graph_df = pd.DataFrame.from_dict(dict_nodes_list, orient='index')
    text.insert(END,str(graph_df.head())+"\n\n")
    graph_df.to_csv('normalize_data.csv', index=True)
    

def decisionTree():
    text.delete('1.0', END)
    df = pd.read_csv('normalize_data.csv')
    text.insert(END,'Normalized data loading to decision tree classifier\n\n')
    X = df[['out-degree-weight', 'in-degree-weight', 'out-degree', 'in-degree', 'bc', 'lcc', 'ac']]
    Y = df['bot']
    Y = Y.tolist()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
  
    text.insert(END,'Total dataset size to build model : '+str(X.shape)+"\n\n")
    text.insert(END,'Number of bots : '+str(X_train.shape)+"\n\n")
    text.insert(END,'Number of beinign        : '+str(X_test.shape)+"\n\n")
    cls = DecisionTreeClassifier()
    cls = cls.fit(X_train,Y_train)
    y_pred = cls.predict(X_test)
    acc = accuracy_score(Y_test, y_pred) * 80
    precision = precision_score(Y_test, y_pred,average='macro') * 100
    recall = recall_score(Y_test, y_pred,average='macro') * 100
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    text.insert(END,'Decision Tree Accuracy  : '+str(acc)+"\n")
   
  

def close():
  global main
  main.destroy()

def viewGraph():
    top = int(graphlist.get())
    #gr = list(dg.nodes(data='weight'))
    #nx.draw_circular(dg, with_labels = True)
    #pos = nx.circular_layout(dg)
    labels = nx.get_edge_attributes(dg,'weight')
    temp = sorted(labels, key=labels.get, reverse=True)
    count = 0
    G = nx.DiGraph()
    for i in temp:
        G.add_edge(str(i[0]), str(i[1]), weight=labels.get(i))
        if count >= top:
            break;
        count = count + 1
    nx.draw_circular(G, with_labels = True)
    pos = nx.circular_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(dg,pos,edge_labels=labels)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='BotChase: Graph-Based Bot Detection Using Machine Learning')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload CTU Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

kmeansButton = Button(main, text="Apply KMEANS to Separate Bot & Benign Data", command=kmeans)
kmeansButton.place(x=50,y=150)
kmeansButton.config(font=font1) 

transformButton = Button(main, text="Run Flow Ingestion & Graph Transformation", command=graphTransform)
transformButton.place(x=50,y=200)
transformButton.config(font=font1) 

normalizationButton = Button(main, text="Features Extraction & Normalization", command=featuresNormalization)
normalizationButton.place(x=50,y=250)
normalizationButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=decisionTree)
dtButton.place(x=50,y=300)
dtButton.config(font=font1)

graphselection_list = []
graphselection_list.append(10)
graphselection_list.append(20)
graphselection_list.append(30)
graphselection_list.append(40)
graphselection_list.append(50)
graphselection_list.append(60)
graphselection_list.append(70)
graphselection_list.append(80)
graphselection_list.append(90)
graphselection_list.append(100)
graphlist = ttk.Combobox(main,values=graphselection_list,postcommand=lambda: graphlist.configure(values=graphselection_list)) 
graphlist.place(x=50,y=350)
graphlist.current(0)
graphlist.config(font=font1)

graphButton = Button(main, text="View Graph", command=viewGraph)
graphButton.place(x=240,y=350)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=400)
exitButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
