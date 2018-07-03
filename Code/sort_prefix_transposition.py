import math
import random
import commands
#import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
#from graphviz import Digraph

def show_stat(D):
    maxi = math.factorial(D[1])
    dict = OrderedDict(sorted(D[0].items()))
    for items in dict.keys():
        dict[items] = round(float(dict[items])/float(maxi)*100.0,2)

    plt.plot(dict.keys(), dict.values())

    print("Pourcentage de reussite : "+str(round(float(D[0][1.0]) / float(math.factorial(D[1]))*100.0,2))+"%")
    maximum = 1.0
    for dist, perms in D[0].items():
        maximum = max(maximum,dist)

    print("Borne sup : "+str(maximum))
    ax=plt.gca()
    ax.set_yticklabels(['0%', '20%','40%','60%','80%','100%'])
    ax.set_xlim([1,2])
    ax.set_ylim([0, 100])
    plt.show()

def show_proportion(n,func):
    maxi = math.factorial(n)
    dict_show = func(n)

    print(dict_show)
    '''for items in dict_show.keys():
        dict_show[items]=round(float(dict_show[items])/float(maxi)*100.0,2)'''
    plt.plot(dict_show.keys(), dict_show.values())
    '''ax = plt.gca()
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_ylim([0, 100])'''
    plt.show()





def show_graph_1(matrice,perm):
    const=0
    F = open("graphe.tek","w+")
    F.write("\documentclass{article}\n\usepackage{tikz}\n\\begin{document}\n\\begin{tikzpicture}\n")
    F.write("\\node (0) at ("+str(0+const)+",0) {0};\n\\node (0+) at ("+str(0.5+const)+",0) {+};\n")
    for i in range(len(perm)):
        F.write("\\node ("+str(perm[i])+"-) at ("+str(float((i+1)*2)-0.5+const)+",0) {-};\n")
        F.write("\\node ("+str(perm[i])+") at ("+str(float((i+1)*2)+const)+",0) {"+str(perm[i])+"};\n")
        F.write("\\node ("+str(perm[i])+"+) at ("+str(float((i+1)*2)+0.5+const)+",0) {+};\n")
    F.write("\\node ("+str(len(perm)+1)+"-) at ("+str((len(perm)+1)*2-0.5+const)+",0) {-};\n")
    F.write("\\node ("+str(len(perm)+1)+") at ("+str((len(perm)+1)*2+const)+",0) {"+str(len(perm)+1)+"};\n")

    F.write("\draw[->, >= latex] ("+str(len(perm)+1)+"-)to [bend left = 30] ("+str(perm[len(perm)-1])+"+);\n")
    for i in range(1,len(perm)):
        F.write("\draw[->, >= latex] ("+ str(perm[len(perm)-i])+"-)to [bend left = 30] ("+ str(perm[len(perm)-i-1])+"+);\n")
    F.write("\draw[->, >= latex] ("+str(perm[0])+"-)to [bend left = 30] (0+);\n")

    F.write("\draw[->, >= latex, color = red] (0+) to[bend left ="+str((perm.index(1)+1)*10+20) +"] (1-);\n")
    for i in range(1,len(perm)):
        if(perm.index(i)<perm.index(i+1)):
            F.write("\draw[->, >= latex, color = red] ("+str(i)+"+) to[bend left ="+str(( math.fabs(perm.index(i)-perm.index(i+1)))*10+20) +" ] ("+str(i+1)+"-);\n")
        else:
            F.write("\draw[->, >= latex, color = red] (" + str(i) + "+) to[bend right = "+str(( math.fabs(perm.index(i)-perm.index(i+1)))*10+20) +"] (" + str(i + 1) + "-);\n")
    F.write(" \draw[->, >= latex, color = red] ("+str(len(perm))+ "+) to[bend left ="+str((len(perm)-perm.index(len(perm)))*10+20) +"] ("+str(len(perm)+1)+"-);\n")
    F.write("\end{tikzpicture}\n\end{document}\n")

    F.close()
    commands.getstatusoutput("pdflatex graphe.tek")
    commands.getstatusoutput("evince graphe.pdf")

    return 0

def show_graph_2(matrice,perm):
    const=0
    F = open("graphe.tek","w+")
    F.write("\documentclass{article}\n\usepackage{tikz}\n\\begin{document}\n\\begin{tikzpicture}\n")
    F.write("\\node (0) at ("+str(0+const)+",0) {0};\n\\node (0+) at ("+str(0.5+const)+",0) {};\n")
    for i in range(len(perm)):
        F.write("\\node ("+str(perm[i])+"-) at ("+str(float((i+1)*2)-0.5+const)+",0) {};\n")
        F.write("\\node ("+str(perm[i])+") at ("+str(float((i+1)*2)+const)+",0) {"+str(perm[i])+"};\n")
        F.write("\\node ("+str(perm[i])+"+) at ("+str(float((i+1)*2)+0.5+const)+",0) {};\n")
    F.write("\\node ("+str(len(perm)+1)+"-) at ("+str((len(perm)+1)*2-0.5+const)+",0) {};\n")
    F.write("\\node ("+str(len(perm)+1)+") at ("+str((len(perm)+1)*2+const)+",0) {"+str(len(perm)+1)+"};\n")

    F.write("\draw[->, >= latex] ("+str(len(perm)+1)+"-.center)to ("+str(perm[len(perm)-1])+"+.center);\n")
    for i in range(1,len(perm)):
        F.write("\draw[->, >= latex] ("+ str(perm[len(perm)-i])+"-.center)to  ("+ str(perm[len(perm)-i-1])+"+.center);\n")
    F.write("\draw[->, >= latex] ("+str(perm[0])+"-.center)to (0+.center);\n")

    F.write("\draw[->, >= latex, color = red] (0+.center) to[bend left =50.0] (1-.center);\n")
    for i in range(1,len(perm)):
        if(perm.index(i)<perm.index(i+1)):
            F.write("\draw[->, >= latex, color = red] ("+str(i)+"+.center) to[bend left =50.0 ] ("+str(i+1)+"-.center);\n")
        else:
            F.write("\draw[->, >= latex, color = red] (" + str(i) + "+.center) to[bend right = 50.0] (" + str(i + 1) + "-.center);\n")
    F.write(" \draw[->, >= latex, color = red] ("+str(len(perm))+ "+.center) to[bend left = 50.0] ("+str(len(perm)+1)+"-.center);\n")
    F.write("\end{tikzpicture}\n\end{document}\n")

    F.close()
    commands.getstatusoutput("pdflatex graphe.tek")
    commands.getstatusoutput("evince graphe.pdf")

    return 0

def calcul_permutation_proportion(n):
    dict_show = {}
    dict = all_distance_prefix(n)
    for i in dict.values():
        if not dict_show.has_key(i[0]):
            dict_show[i[0]] = 1
        else:
            dict_show[i[0]]+=1
    return dict_show

def calcul_optimal_chemin_proportion(n):
    dict_show = {}
    dict = all_distance_prefix(n)
    for i in dict.values():
        if not dict_show.has_key(i[2]):
            dict_show[i[2]] = 1
        else:
            dict_show[i[2]] += 1
    return dict_show



def all_distance_prefix(n):
    dic={}
    perm = [i + 1 for i in range(n)]
    dic[tuple(perm)] = [0, 1, 1]
    obj = math.factorial(n)
    distance = 1
    while len(dic) < obj:
        for perm, value in dic.items():
            if value[0] == distance - 1:
                all_transpositions_prefix(list(perm), dic, distance)
        distance += 1
    return dic

def all_transpositions_prefix(perm, dic, distance):
    for y in range(2, len(perm) + 1):
        for x in range(1, y):
            sol = transposition_prefix(perm,x,y)
            if not dic.has_key(tuple(sol)):
                dic[tuple(sol)] = [distance, bp(sol),dic[tuple(perm)][2]]
            elif dic[tuple(sol)][0] == distance:
                dic[tuple(sol)][2]+=dic[tuple(perm)][2]


def transposition_prefix(perm,x,y):
    sol = perm[x:y]
    sol.extend(perm[:x])
    sol.extend(perm[y:])
    return sol

def bp(perm):
    sol = 1
    for i in range(1, len(perm)):
        if perm[i] - perm[i - 1] != 1:
            sol += 1
    if perm[len(perm) - 1] != len(perm):
        sol += 1
    return sol

def breakpoint_graph(perm):
    matrice = [[0] * (len(perm)+2) for _ in range(len(perm)+2)]

    for i in range(len(perm)+1):
            matrice[i][i+1]+=2

    perm.insert(0,0)
    perm.append(len(perm))

    for i in range(1,len(perm)):
        matrice[i][perm[perm.index(i)-1]]+=1

    perm.remove(0)
    perm.remove(len(perm))
    return matrice


def number_cycles(matrice):
    nb=0
    resultat=[]
    list = [[1]*len(matrice) for i in range(2) ]

    list[0][0]=0
    list[1][len(matrice)-1]=0
    cycle = []

    for i in range(len(matrice)):
        for j in range(2):
            if(list[j][i]==1):
                list[j][i]=2
                cycle.append(i)

                k=0
                while k < len(matrice):

                    if(matrice[i][k]==j+1 or matrice[i][k]==3):
                        j=(j+1)%2
                        i=k

                        if( list[j][i]==2):
                            list[j][i]=0
                            resultat.append(cycle)
                            nb+=1
                            cycle = []
                            break
                        if(list[j][i]==1):

                            list[j][i]=0
                            cycle.append(i)
                            k=-1
                    k+=1

    return (nb,resultat)

def nb_cycle_of_size(perm,n):
    cycles = number_cycles(breakpoint_graph(perm))[1]
    sol=0
    for c in cycles :
        if(len(c) == n*2):
            sol+=1
    return sol

def permutation_to_map(perm):
    perm.insert(0,0)
    perm.append(len(perm))
    print(perm)
    map = []
    not_free=[]

    for i in range(len(perm)-1):
        cycle = []
        if perm[i] not in not_free:
            cycle.append(perm[i])
            j=perm[perm[perm[i]+1]-1]

            while j not in cycle:
                cycle.append(j)
                not_free.append(j)
                j=perm[perm[j+1]-1]
            map.append(cycle)
    return map


def map_to_permutation(map):
    size=0
    for c in map:
        size+=len(c)
    tab=[0]*(size+1)
    print tab
    for c in map:
        for i in range(len(c)):
            tab[(c[i]+1)%len(tab)]=c[(i+1)%len(c)]
    res=[]
    i=len(tab)-1
    while i != 0:
        i=tab[i]
        res.insert(0,i)

    res.pop(0)
    return res


def print_matrice(matrice):
    for line in matrice: print(line)

def div_supp(val, div):
    if int(val/div)*div != val:
        return int(val/div) + 1
    return int(val/div)

def Bk_generateur(n):
    perm = []

    for i in range(n):
        perm.extend([n+i+1])
        perm.extend([n-i])

    return perm

def identite_generateur(n):
    perm = []

    for i in range(n):
        perm.extend([i+1])

    return perm

def reverse_generateur(n):
    perm = []

    for i in range(n):
            perm.extend([n-i])

    return perm

def Mk_generateur(n):
    perm = []

    for i in range(n):
        perm.extend([1+3*i])
        perm.extend([3+3*i])
        perm.extend([2+3*i])

    return perm

def permutation_random_generateur(n):
    perm=identite_generateur(n)
    sol=[]
    for i in range(len(perm)):
          sol.append( perm.pop(random.randint(0, n-i-1)))

    return sol

def algo_sort_reverse(perm):
    n = len(perm)
    if n < 4:
        print("Cette algorithme fonctionne pour les permutation de taille >= 4")
        exit()

    nb_perm=0;
    #print("Size : "+str(n))

    #Phase 0

    for i in range(0,n%4):
        perm = transposition_prefix(perm,1,n-i)
        nb_perm+=1
    n=n-n%4

    print("Phase 0 : "+str(perm))

    #Phase 1

    for i in range(0,n/4-1):
        perm = transposition_prefix(perm,4,n-2*i-1)
        nb_perm+=1
    print("Phase 1.1 : " + str(perm))

    perm = transposition_prefix(perm,2,n/2+1)
    nb_perm+=1

    print("Phase 1.2 : " + str(perm))

    #Phase 2

    for i in range(0,n/4):
        perm = transposition_prefix(perm,2,n-4*i)
        nb_perm+=1

    print("Phase 2 : " + str(perm))

    #Phase 3

    for i in range(0,n/8):
        perm = transposition_prefix(perm,n-5-4*i,n-4*i-1)
        nb_perm+=1

    print("Phase 3.1 : " + str(perm))

    if n%8 == 0 :
        perm = transposition_prefix(perm,2,n/2+1)
        nb_perm+=1
    else:
        perm = transposition_prefix(perm,n/2-1,n/2+1)
        nb_perm+=1


    print("Phase 3.2 : " + str(perm))

    #Phase 4

    for i in range(0, div_supp(n,8.0)-1):
        perm = transposition_prefix(perm,4,4*(n/8)+5+4*i)
        nb_perm+=1


    return (perm,nb_perm)

def algo_sort_Mk(perm,n): #valeur de n = k

    if(n == 0):
        print("Cette algorithme fonctionne pour les permutation de taille >= 1")
        exit()

    for i in range(1,n+1):
        print(perm)
        perm = transposition_prefix(perm,3*(i-1)+1,3*(i-1)+2)
        print(perm)
        perm = transposition_prefix(perm,1,3*(i-1)+3)

    return perm

def algo_sort_Bk(perm,n): #valeur de n = k

    if(n == 0):
        print("Cette algorithme fonctionne pour les permutation de taille >= 1")
        exit()

    for i in range(1,n+1):
        perm = transposition_prefix(perm,2*i-1,2*i)

    return perm

def algo_sort_prefix_v1(perm):
    nb_transposition=0

    if( len(perm) <= 1 ):
        return nb_transposition
    prefix=[]
    i=0
    while i < len(perm)-1:
        ok=True
        prefix.extend(str(perm[i]))
        if( perm[i]+1 != perm[i+1] ):
            for j in range(i+1,len(perm)):
                if(perm[i]+1==perm[j]):
                    perm = transposition_prefix(perm,i+1,j)
                    nb_transposition+=1
                    i=0
                    ok=False
                    break
            if(ok):
                perm = transposition_prefix(perm, i+1, len(perm))
                nb_transposition += 1
                i = 0
        else:
            i=i+1
    return nb_transposition

def algo_sort_prefix_v2(perm):
    nb_transposition = 0
    if( len(perm) <= 1 ):
        return nb_transposition

    while( bp(perm) != 1 ):
        done=False
        memo=False
        for i in range(1,len(perm)):
            if(done):
                break
            if(perm[i]==perm[0]-1):

                if(i==len(perm)-1):

                    for j in range(len(perm)):
                        if(perm[j]==len(perm)):
                            perm = transposition_prefix(perm, j +1, i + 1)
                            nb_transposition += 1
                            done = True
                            break
                else:
                    for j in range(i):
                        if(perm[j]==perm[i+1]-1):
                            perm = transposition_prefix(perm,j+1,i+1)
                            nb_transposition+=1
                            done = True
                            memo = False
                            break
                        if (perm[j] == len(perm) and not memo):
                            memo = True
                            k=j

                    if(memo):
                        perm = transposition_prefix(perm, k + 1, i + 1)
                        nb_transposition += 1
                        done = True

        if(done!=True):

            bande=0
            for i in range(len(perm)-1):
                if(perm[i]+1!=perm[i+1]):
                    bande=i
                    break

            if(perm[bande]==len(perm)):
                perm = transposition_prefix(perm,1,len(perm))
                nb_transposition+=1
            else:
                for i in range(bande,len(perm)):
                    if(perm[bande]+1==perm[i]):
                        perm=transposition_prefix(perm,bande+1,i)
                        nb_transposition+=1
                        break
    return nb_transposition

def algo_sort_prefix_v3(perm):
    nb_transposition=0
    while (bp(perm) != 1):
        done = False
        for i in range(1,len(perm)):
            if(done):
                break
            if((perm[i]+1)%len(perm)==(perm[0])%len(perm)):
                if(i == len(perm)-1):
                    break

                for j in range(1,i):
                    if((perm[j]+1)%len(perm)==(perm[i+1])%len(perm)):
                        perm=transposition_prefix(perm,j+1,i+1)
                        nb_transposition+=1
                        done = True
                        break

        if(not done):
            for i in range(len(perm)-1):
                if(perm[i]+1!=perm[i+1] and not((perm[i+1]==1 and bp(perm)!=3)) ):
                    if(perm[i]==len(perm)):
                        perm = transposition_prefix(perm, i + 1, len(perm))
                        nb_transposition += 1
                        break
                    if(perm[0]==perm[len(perm)-1]+1):
                        for j in range(len(perm)):
                            if(perm[j]==len(perm)):
                                perm = transposition_prefix(perm, j+1, len(perm))
                                nb_transposition += 1
                                done = True
                                break
                    else:
                        for j in range(i,len(perm)):
                            if(perm[j]==perm[i]+1):
                                perm = transposition_prefix(perm, i + 1, j)
                                nb_transposition += 1
                                done = True
                                break

                if(done): break
    return nb_transposition


def algo_sort_prefix_v4(perm):
    nb_transposition = 0

    matrice=breakpoint_graph(perm)

    while(bp(perm)!=1):
        tab = [0] * (len(perm) + 2)
        perm.insert(0, 0)
        perm.append(len(perm))
        for i in range(len(perm)-1):
            if(perm.index(perm[i]+1) < i):
                continue
            for j in range(i+1,perm.index(perm[i]+1)):
                if(perm.index(perm[j]+1) > perm.index(perm[i]+1) or perm.index(perm[j]+1) <= i):
                    tab[i]+=1
                if(perm.index(perm[j]-1) > perm.index(i+1) or perm.index(perm[j]-1) < i):
                    tab[i]+=1
        perm.remove(0)
        perm.remove(len(perm))
        tab.pop(len(perm))
        tab.pop(0)
        done=False
        maximum= max(tab)
        while(done==False):
            for i in range(len(perm)-1,-1,-1):
                if(tab[i]==maximum and ( ((i!= len(perm)-1) and perm[i+1]!=1) or bp(perm)==3)):
                    if(perm[i]+1==len(perm)+1):
                        perm = transposition_prefix(perm, i + 1, len(perm))
                    else:
                        perm=transposition_prefix(perm, i+1,perm.index(perm[i]+1))
                    nb_transposition+=1
                    done = True
                    break
            maximum-=1
    return nb_transposition

def is_double_inverse(perm):

    for i in range(len(perm)-1):
        if(perm[i]!=perm[i+1]+1 and perm[i]!=1):
            return False
    if(perm[-1]==1):
        return False
    return True


def algo_sort_prefix_v5(perm):
    nb_transposition = 0

    if(bp(perm)<=3):
        return bp(perm)/2

    if(len(perm) >5 and is_double_inverse(perm)):
        #maxi=max(perm.index(1)+1,len(perm)-(perm.index(1)+1))
        perm = transposition_prefix(perm, 2,4)
        nb_transposition+=1
    else:
        return algo_sort_prefix_v3(perm)

    while(bp(perm)!=1):
        pile = []
        print(perm)
        for i in range(len(perm)):
            if(perm[i]==perm[0]-1):

                for j in range(i):
                    if(i == len(perm)-1):
                        if(perm[j]==len(perm)):
                            perm = transposition_prefix(perm, j+1, i+1)
                            nb_transposition += 1
                    elif(perm[i+1]==1 and perm[j] == len(perm)):
                        perm = transposition_prefix(perm, j + 1, i + 1)
                        nb_transposition += 1
                    elif(perm[j]+1 ==perm[i+1]):
                        perm = transposition_prefix(perm,j+1,i+1)
                        nb_transposition+=1


        for i in range(len(perm)-1,-1,-1): #pense a len(perm)
            if(perm[i]==len(perm) and i != len(perm)-1 and perm[i+1]!=1):
                perm = transposition_prefix(perm, i+1, len(perm))
                nb_transposition += 1
                break
            if( (perm[i]+1 in pile) and perm[i+1]!=perm[i]+1 and ( perm[i+1]!=1 or bp(perm)==3)):
                perm = transposition_prefix(perm,i+1,perm.index(perm[i]+1))
                nb_transposition+=1
                break
            pile.append(perm[i])

    return nb_transposition

def algo_sort_prefix_v6(perm):
    nb_transposition = 0
    print(perm)
    tab = [[0 for j in range(3)] for i in range(len(perm))]
    tab_coord =[() for i in range(len(perm))]
    print(tab)
    while(bp(perm)!=1):
        done = False
        for i in range(0,len(perm)-1):
            if(perm[i]+1==perm[i+1]):
                print(i)
                tab[i]=[-1,-1,0]
            elif(perm[0]==perm[i]+1):
                for j in range(i):
                    if((i == len(perm)-1 and perm[j]+1 ==len(perm)) or(perm[j]+1 == perm[i+1])):
                        tab[i][1]=2
                        tab_coord[i]=(i,j)
                        done = True
                if not done:
                    for j in range(i):
                        if(perm[j]+1 != perm[j+1]):
                            tab[i][1]=1
                            if(perm[j+1]==1):
                                tab[i][2]-=1
                            if(perm[j+1] < perm[j]):
                                tab[i][0]=-1
                            tab_coord[i] = (i, j)
                            break


        if(perm[-1]==len(perm)):
            tab[-1] = [-1, -1, 0]

        print(tab)


    return nb_transposition


'''def distance_backtracking(perm):
    return distance_backtracking_aux(perm,[],0,False,{})

def distance_backtracking_aux(perm,pile,distance,find,dict):

    #Initialisation des valeurs par defaut
    if (perm[0] == 1):
        val = 0
    else :
        val = 1


    if(dict.has_key(str(perm))):
        return dict[str(perm)]
    # Condition d'arret a la borne inf
    if(bp(perm)==1):
        return distance
    if(bp(perm) == 3):
        if(distance+1 == ((len(perm)+1+number_cycles(breakpoint_graph(perm))[0])/2-nb_cycle_of_size(perm,1)-val )):
            find = True
        return distance+1
    # Born supp
    if( distance >= len(perm)-math.log(len(perm),8)):
        return distance

    minimum = len(perm)
    for y in range(2, len(perm) + 1):
        for x in range(1, y):
            sol = transposition_prefix(perm,x,y)
            if sol not in pile and bp(perm)-bp(sol) >= 0:
                pile.append(sol)

                minimum= min( distance_backtracking_aux(sol,pile,distance+1,find,dict), minimum )
                if(find):
                    return minimum
                pile.remove(sol)
    dict[str(perm)]=minimum
    return minimum
'''

def distance_backtracking(perm):
    return distance_backtracking_aux(perm,[],0)

def distance_backtracking_aux(perm,pile,dist):

    '''if(dict.has_key(str(perm))):
        return dict[str(perm)]'''
    # Condition d'arret a la borne inf
    if(bp(perm)==1):
        if(dist==6):
            print(pile)
        return [0,pile]

    if(dist>6):
        return [6,[]]
    minimum = [len(perm),[]]
    for y in range(2, len(perm) + 1):
        for x in range(1, y):
            sol = transposition_prefix(perm,x,y)

            if  bp(perm)-bp(sol) >= 0 and sol not in pile:
                pile.append(sol)
                tmp = distance_backtracking_aux(sol, pile, dist + 1)
                if(tmp[0]+1 < minimum[0] ):
                    tmp[0]=tmp[0]+1
                    minimum=tmp
                pile.remove(sol)
    return minimum

def all_permutation_of_distance(n,d=None):
    if(d==None):
        d= longest_distance(n)
    dic=all_distance_prefix(n)
    for perm, dist in dic.items():
        if(dist[0] == d ):
            print(str(perm)+" : dp("+str(dist[0])+") , bp("+str(dist[1])+");")

def longest_distance(n):
    return calcul_permutation_proportion(n).keys()[-1]

def print_dic(dic,func,n):
    for perm, dist in dic.items():
        sol = func(list(perm))
        if(sol != 0 and round(float(dist[0])/float(sol)*100.0) <=n):
            print(str(perm)+" : dp("+str(dist[0])+") , bp("+str(dist[1])+"); "+str_func(func)+" : dp("+str(sol)+"); efficacite : "+str(round(float(dist[0])/float(sol)*100.0,2))+"%")

def str_func(func):
    name=str(func)
    return name[10:-19]

def calcul_stat(func, n):
    dic_objectiv = all_distance_prefix(n)
    dic_stat = {}

    for perm, dist in dic_objectiv.items():
        if(dist[0]==0):
            val=1.0
        else:
            val = round(float(func(list(perm)))/float(dist[0]),2)
        if(dic_stat.has_key(val)):
            dic_stat[val]+=1
        else:
            dic_stat[val]=1
    return (dic_stat,n)

'''
Permutation devant faire des transposition ne cassant pas de bp pour etre opt:

(5, 4, 3, 2, 1, 7, 6) : dp(5) , bp(8); distance_backtracking : dp(6); efficacite : 83.33%
(1, 7, 6, 5, 4, 3, 2) : dp(5) , bp(8); distance_backtracking : dp(6); efficacite : 83.33%
(3, 2, 1, 7, 6, 5, 4) : dp(5) , bp(8); distance_backtracking : dp(6); efficacite : 83.33%

[[3, 2, 5, 4, 1, 7, 6], [4, 1, 7, 3, 2, 5, 6], [7, 3, 4, 1, 2, 5, 6], [3, 4, 1, 2, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]

Memo:

(5, 3, 2, 4, 1) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(3, 5, 4, 2, 1) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(2, 5, 1, 4, 3) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(1, 5, 3, 2, 4) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(1, 4, 3, 5, 2) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(3, 2, 5, 1, 4) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(4, 3, 2, 5, 1) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%
(4, 2, 5, 3, 1) : dp(3) , bp(6); algo_sort_prefix_v5 : dp(4); efficacite : 75.0%

print_dic(all_distance_prefix(5),algo_sort_prefix_v5,99.9)
'''

perm=[1,3,2,4,6,5]
map_to_permutation(permutation_to_map(perm))
