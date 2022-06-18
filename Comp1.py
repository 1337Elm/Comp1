#Benjamin Elm Jonsson 20011117, Independent Programming Work for Course DAT171 at Chalmers Institute of Technology
#Importing all libraries that are used within the code.
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import time
from scipy import spatial
from matplotlib.collections import LineCollection

#Function that opens the chosen textfile and produces a list of the coordinates with good format for further work.
def read_coordinate_file(file):
    R = 1
    coordinates = []
    xMerc = []
    yMerc = []

    with open (file, 'r') as f:
        for line in f:
            coordinate = [item.strip("{.}\n") for item in line.split(',')]
            coordinates.append(coordinate)
    f.close()

    for i in range(len(coordinates)):
        xMerc.append(R*((np.pi*float(coordinates[i][1]))/180))
        yMerc.append(R*(np.log(np.tan((np.pi/4) + ((np.pi*float(coordinates[i][0])))/360))))

    coord_list = np.array([yMerc,xMerc])
    return coord_list

#Goes through all the cities from the file and determines what cities are within reach of it. Returns the connections between the cities and that distance.
def construct_graph_connections(coord_list,radius):
    cit1 = []
    cit2 = []
    dista = []

    for i in range(len(coord_list[1])):
        for j in range(i+1,len(coord_list[1])):
            if radius > math.sqrt((coord_list[0][j]-coord_list[0][i])**2 + (coord_list[1][j]-coord_list[1][i])**2):
                cit1.append(i)
                cit2.append(j)
                dista.append(math.sqrt((coord_list[0][j]-coord_list[0][i])**2 + (coord_list[1][j]-coord_list[1][i])**2))
    
    indices = np.array([cit1,cit2])
    distance = np.array(dista)
    N = len(coord_list[0])
    return indices,distance,N

#Function that determines connections like above but with the cKDTree from scipy for an improvement of performance.
def construct_fast_graph_connections(coord_list,radius):
    N = len(coord_list[0])
    cit1 = []
    cit2 = []
    dista = []
    points = []

    for i in range(N):
        points.append([coord_list[1][i],coord_list[0][i]])

    tree = spatial.cKDTree(points,2)

    for i in range(N):
        indx = tree.query_ball_point([coord_list[1][i],coord_list[0][i]],radius)
        for j in range(len(indx)):
            if i!=indx[j]:
                cit1.append(i)
                cit2.append(indx[j])
                dista.append(math.sqrt((coord_list[0][indx[j]]-coord_list[0][i])**2 + (coord_list[1][indx[j]]-coord_list[1][i])**2))
    
    indices = np.array([cit1,cit2])
    distance = np.array(dista)
    return indices,distance,N

#Function that plots the solution: the cities, the connections between them and finally the shortest path found between the chosen start node and end node.
def plot_points(coord_list,indices,Path):
    start5 = time.time()
    x = coord_list[1]
    y = coord_list[0]

    Ind12x = [x[indices[0]],x[indices[1]]]
    Ind12y = [y[indices[0]],y[indices[1]]]
    segs = np.transpose(np.array([Ind12x,Ind12y]))
    lines = LineCollection(segs,linewidth = 0.3,alpha = 0.8,colors = 'k')

    Paths1 = []
    Paths2 = []

    for i in range(len(Path)):
        if i<len(Path)-1:
            Paths1.append(Path[i])
        
        if i>0:
            Paths2.append(Path[i])

    Pathx = [x[Paths1],x[Paths2]]
    Pathy = [y[Paths1],y[Paths2]]
    PathSegs = np.transpose(np.array([Pathx,Pathy]))
    PathLines = LineCollection(PathSegs,linewidth = 2,alpha = 1,colors = 'b')

    plt.scatter(x,y,s=15,c='r',alpha=0.7)
    plt.gca().add_collection(lines)
    plt.gca().add_collection(PathLines)

    end5 = time.time()
    tid = end5-start5

    plt.show()
    return tid

#Turns the connections from earlier to a matrix where the distance from city i to j is the element in the matrix at the coordinates i,j.  
def construct_graph(indices,distance,N):
    Matris = csr_matrix((distance, (indices[1],indices[0])), shape=(N,N)).toarray()
    return Matris

#Finds shortest path with the shortest_path function from scipy.
def find_shortest_path(graph,start_node,end_node):
    Path = [end_node]
    D,Pr = shortest_path(graph,directed = False,indices= [start_node],return_predecessors=True)
    
    i = end_node

    while i!=start_node:
        Path.append(Pr[0][i])
        i = Pr[0][i]
    Path.reverse()
    return Path,D[0][end_node]

#Main programme that calls the functions above. Also here the file is chosen. Furthermore all times are recorded for the individual functions
def main():
    ans = input("Would you like to use Sample, Hungary or Germany?")
    if ans == "S":
        file = "/Users/benjaminjonsson/Programmering/Comp1/SampleCoordinates.txt"
        radius = 0.08
        start_node = 0
        end_node = 5
    elif ans == "H":
        file = "/Users/benjaminjonsson/Programmering/Comp1/HungaryCities.txt"
        radius = 0.005
        start_node = 311
        end_node = 702
    else:
        file = "/Users/benjaminjonsson/Programmering/Comp1/GermanyCities.txt"
        radius = 0.0025
        start_node = 1573
        end_node = 10584
    
    start1 =time.time()
    CordList = read_coordinate_file(file)
    end1 = time.time()
    
    ans2 = input("Do you want to use fast version?")
    if ans2 == "y":
        startfast = time.time()
        Conne = construct_fast_graph_connections(CordList,radius)
        endfast = time.time()
    else:
        start2 = time.time()
        Conne = construct_graph_connections(CordList,radius)
        end2 = time.time()

    start3 =time.time()
    Mtrx = construct_graph(Conne[0],Conne[1],Conne[2])
    end3 = time.time()

    start4 = time.time()
    Path  = find_shortest_path(Mtrx,start_node,end_node)
    end4 = time.time()
    print(Path)

    Tid = plot_points(CordList,Conne[0],Path[0])

    if ans2 =="y":
        print(f"read_coordinate_file: {end1-start1}")
        print(f"construct_fast_graph_connections: {endfast-startfast}")
        print(f"construct_graph: {end3-start3}")
        print(f"find_shortest_path: {end4-start4}")
        print(f"plot_points exkl. plt.show(): {Tid}")
    else:
        print(f"read_coordinate_file: {end1-start1}")
        print(f"construct_graph_connections: {end2-start2}")
        print(f"construct_graph: {end3-start3}")
        print(f"find_shortest_path: {end4-start4}")
        print(f"plot_points exkl. plt.show(): {Tid}")

#Determines that this is a runnable file.
if __name__== '__main__':
    main()