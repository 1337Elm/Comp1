"""Completion of Computer Assignment 1 for course DAT171, Chalmers Institute of Technology. 

Runnable file that calculates
shortest distance from a given startnode to a given endnode in one of the following cities: Sample (S), Hungary (H) or Germany (G). 
There is also a fast(y) or slow(any) version of this program.

Author: Benjamin Elm Jonsson 20011117 (2022) benjamin.elmjonsson@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import time
from scipy import spatial
from matplotlib.collections import LineCollection

def read_coordinate_file(file):
    """ Opens the chosen textfile, strips data in order to make it useful.
        Thereafter mercator projection is used and finally the data is put
        in a Numpy-array that is returned.

        :param file: Chosen File to be analyzed
        :type file: filename - str
    """
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

def construct_graph_connections(coord_list,radius):
    """ Goes through all cities and determining which other cities 
        that can be travelled to. If the connection can be made those
        cities are saved and a list of the possible connections are
        returned aswell as the distance between them in two separate lists.

        :param coord_list: List of the coordinates of each city
        :type coord_list: list, 2D Numpy-array

        :param radius: Maximum range allowed inbetween cities in order to make the connection possible
        :type radius: int
    """
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

def construct_fast_graph_connections(coord_list,radius):
    """Creates a cKDTree in order to determine which cities are
       lying in reach of the given radius of the selected cities.
       These are put into Nunpy-arrays to then be returned aswell as
       distance between the cities.

       :param coord_list: List of the coordinates of each city
       :type coord_list: list, 2D Numpy-array

       :param radius: Maximum range allowed inbetween cities in order to make the connection possible
       :type radius: int
    """
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

def plot_points(coord_list,indices,Path):
    """Plotting the results of the program. 
       Firstly the cities are plotted as a scatterplot in order to see them individually.
       Secondly matplotlibs library is used to plot the routes between the cities.
       Lastly the shortest path is plotted with the matplotlibs library.

       :param coord_list: List of coordinates of each city
       :type coord_list: list, 2D Numpy-array

       :param indices: 2D list of possible routes between cities
       :type indices: list, 2D Numpy-array

       :param Path: List of cities from start_node to end_node
       :type Path: list
    """
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

def construct_graph(indices,distance,N):
    """Constructing the matrix needed in later functions.
       The distance between city i and j is the element in the matrix
       at (i,j). Matrix is of dimension NxN.

       :param indices: 2D list of possible routes between cities
       :type indices: list, 2D Numpy-array
       
       :param distance: list of distances between cities
       :type distance: list, Numpy-array

       :param N: Total ammount of cities
       :type N: int
    """
    Matris = csr_matrix((distance, (indices[1],indices[0])), shape=(N,N)).toarray()
    return Matris

def find_shortest_path(graph,start_node,end_node):
    """Finds the shortest path from start_node to end_node.
       Returns the path taken and its total distance.

       :param graph: Sparse matrix of distances between cities
       :type graph: sparse csr matrix

       :param start_node: The city where the journey starts
       :type start_node: int

       :param end_node: The city where the journey is over
       :type end_node: int
    """
    Path = [end_node]
    D,Pr = shortest_path(graph,directed = False,indices= [start_node],return_predecessors=True)
    
    i = end_node

    while i!=start_node:
        Path.append(Pr[0][i])
        i = Pr[0][i]
    Path.reverse()
    return Path,D[0][end_node]

def main():
    """The main programme which calls the other functions.
       The file is also chosen, furthermore are the timings 
       recorded.
    """
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

if __name__== '__main__':
    """Determines that the file is runnable, initiates the programme.
    """
    main()