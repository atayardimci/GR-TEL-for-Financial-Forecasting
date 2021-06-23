import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#function to create and display networks from the correlatin matrix. 

def plot_graph(weight_matrix, min_correlation, ax, layout='circular', adjust_size_with_degree=None, node_size=300, label_size=8):
	stocks = weight_matrix.index.values
	W = np.asmatrix(weight_matrix)

	#Creates graph using the data of the correlation matrix
	G = nx.from_numpy_matrix(W)
	#relabels the nodes to match the  stocks names
	G = nx.relabel_nodes(G,lambda x: stocks[x])

	##Creates a copy of the graph
	H = G.copy()

	##Checks all the edges and removes some 
	for stock1, stock2, weight in G.edges(data=True):
		####If correlation weaker than the min, then it deletes the edge
		if weight["weight"] < min_correlation:
			H.remove_edge(stock1, stock2)

	#creates a list for the edges and the weights
	edges, weights = zip(*nx.get_edge_attributes(H,'weight').items())

	### increases the value of weights, so that they are more visible in the graph
	# weights = tuple([(1+abs(x))**2 for x in weights])


	## positions
	if layout == 'circular':
		positions=nx.circular_layout(H)
	elif layout == 'spring':
		positions=nx.spring_layout(H)


	#####calculates the degree of each node
	d = nx.degree(H)

	#####creates list of nodes and a list their degrees that will be used later for their sizes
	nodelist, node_sizes = zip(*dict(d).items())
	if adjust_size_with_degree == None:
		node_sizes = node_size
	else:
		node_sizes = tuple([x**adjust_size_with_degree for x in node_sizes])


	#draws nodes
	nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist, ax=ax,
							#####the node size will be now based on its degree
							node_size=node_sizes,
							alpha=0.8)


	#Styling for labels
	nx.draw_networkx_labels(H, positions, font_size=label_size, ax=ax,
							font_family='sans-serif')


	### increases the value of weights, so that they are more visible in the graph
	weights = tuple([(1+abs(x))**2 for x in weights])

	edge_colour = plt.cm.GnBu 
		
	#draws the edges
	nx.draw_networkx_edges(H, positions, edge_list=edges,style='solid', ax=ax,
							###adds width=weights and edge_color = weights 
							###so that edges are based on the weight parameter 
							###edge_cmap is for the color scale based on the weight
							### edge_vmin and edge_vmax assign the min and max weights for the width
							width=weights, edge_color = weights, edge_cmap = edge_colour,
							edge_vmin = min(weights), edge_vmax=max(weights))

	# displays the graph without axis
	ax.axis('off')