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

def kmeans(ctuDataset):
    ctu_bot = ctuDataset[ctuDataset['Label'].str.contains('Botnet')]
    ctu_benign = ctuDataset[~ctuDataset['Label'].str.contains('Botnet')]
    ctu_benign = ctu_benign.sample(n=ctu_bot.shape[0])

    bots = pd.concat([ctu_bot, ctu_benign])
    print('bots after removing benign using KMEANS:')
    print(bots.shape)
    return bots

def graphTransform(ctuDataset):
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

    print('Number of nodes: ' + str(nx.number_of_nodes(dg)))
    print('Number of edges: ' + str(nx.number_of_edges(dg)))
    print('Network graph created')
    return dg, dict_nodes_list, ip_nodes_list

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


def featuresNormalization(ip_nodes_list,dict_nodes_list,between_ness,clustering,alpha_cent):
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
        
            if s_alpha != 0:
                dict_nodes_list[nodes]['ac'] = dict_nodes_list[nodes]['ac'] / (s_alpha/counter)
        
        
    print('Normalizing features process completed')        
    graph_df = pd.DataFrame.from_dict(dict_nodes_list, orient='index')
    print(graph_df.head())
    graph_df.to_csv('normalize_data.csv', index=True)

if __name__ == '__main__':
    #freeze_support()

    ctuDataset = pd.read_csv('dataset/CTU-13-Dataset/10/capture20110818.binetflow')
    print('Dataset size:')
    print(ctuDataset.shape)
    print(ctuDataset.head())
    bots = kmeans(ctuDataset)

    graph = bots[['SrcAddr', 'DstAddr', 'TotPkts']]
    print(graph.head())
    print(graph.shape)
    dg,dict_nodes_list,ip_nodes_list = graphTransform(graph)

    #Calculate the betweeness centrality for all nodes
    print('Betweeness centrality time calculation for all ip address or nodes:')
    start_time = time.time()
    between_ness = betweennessCentrality(dg, 2)
    print("Execution Time: %.4F" % (time.time() - start_time))

    #clustering time
    print('Clustering time calculation')
    start_time = time.time()
    clustering = nx.clustering(dg, weight='weight')
    print("Clustering Time: %.4F" % (time.time() - start_time))

    #Alpha Centrality time calculation
    print('Alpha Centrality time calculation')
    start_time = time.time()
    alpha_cent = nx.algorithms.centrality.katz_centrality_numpy(dg, weight='weight')
    print("Alpha Centrality Time: %.4F" % (time.time() - start_time))

    bot_list = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    for i in dict_nodes_list:
        if i in bot_list:
            dict_nodes_list[i]['bot'] = 1
        else:
            dict_nodes_list[i]['bot'] = 0

        dict_nodes_list[i]['bc'] = between_ness[i]
        dict_nodes_list[i]['lcc'] = clustering[i]
        dict_nodes_list[i]['ac'] = alpha_cent[i]

    featuresNormalization(ip_nodes_list,dict_nodes_list,between_ness,clustering,alpha_cent)

    df = pd.read_csv('normalize_data.csv')
    X = df[['out-degree-weight', 'in-degree-weight', 'out-degree', 'in-degree', 'bc', 'lcc', 'ac']]
    Y = df['bot']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    cls = DecisionTreeClassifier()
    cls = cls.fit(X_train,Y_train)
    y_pred = cls.predict(X_test)
    acc = accuracy_score(Y_test, y_pred) * 100
    precision = precision_score(Y_test, y_pred,average='micro') * 100
    recall = recall_score(Y_test, y_pred,average='micro') * 100
    cm = confusion_matrix(Y_test, y_pred)
    TN, FP, TP, FN = cm.ravel()
    print(str(acc)+" "+str(precision)+" "+str(recall)+" "+str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN))



