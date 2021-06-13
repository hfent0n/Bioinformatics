from anytree import Node, RenderTree
from anytree.exporter import DotExporter
'''
A simple implementation of [1].
[1] - A. V. Aho, Y. Sagiv, T. G. Szymanski, J. D. Ullman. Inferring a tree from lowest common ancestors
with an application to the optimization of relational expressions. SIAM Journal on Computing,
10:405—421, 1981.
'''
def nodeattrfunc(node):
    if ' ' in node.name:
        return "fixedsize=true, width=0.01, height=0.01, shape=point"
    elif ',' in node.name:
        return "fixedsize=true, width=2, height=2, shape=triangle"  
    else:
        return "fixedsize=true, width=0.3, height=0.3"
   
def partition(total, C):
    S = []
    flags = dict()
    for constraint in C:
        (i, j), (k, l) = constraint
        if i in flags and j in flags:
            if flags[i] != flags[j]:
                min_, max_ = min(flags[i], flags[j]), max(flags[i], flags[j])
                for node in S[max_]:
                    flags[node] = min_
                S[min_].extend(S[max_])
                del S[max_]
                for block in S[max_:]:
                    for node in block:
                        flags[node] -= 1   
        elif i in flags:
            flags[j] = flags[i]
            S[flags[i]].append(j)
        elif j in flags:
            flags[i] = flags[j]
            S[flags[j]].append(i)
        else:
            S.append([i, j])
            flags[i] = len(S) - 1
            flags[j] = len(S) - 1
    for constraint in C:
        (i, j), (k, l) = constraint
        if k in flags and l in flags and flags[k] == flags[l]:
            if flags[k] != flags[i]:
                min_, max_ = min(flags[k], flags[i]), max(flags[k], flags[i])
                for node in S[max_]:
                    flags[node] = min_
                S[min_].extend(S[max_])
                del S[max_]
                for block in S[max_:]:
                    for node in block:
                        flags[node] -= 1
    for item in total:
        if item not in flags:
            S.append([item])
    print(S)
    return S
    

def BUILD(S, C):
    global count
    if len(S) == 1:
        return Node(str(S[0]))
    pi = partition(S, C)
    if len(pi) == 1:
        return None
    else:
        T = []
        for block in pi:
            C_m = [x for x in C if x[0][0] in block and x[0][1] in block and x[1][0] in block and  x[1][1] in block]
            T.append(BUILD(block, C_m))
            if T[-1] == None:
                return T[-1]
    
    root = Node(" " * count)
    count+=1
    print(T)
    for node in T:
        node.parent=root
    return root


if __name__ == "__main__":
    count =  1
    C = [[(1, 3), (2, 5)], [(1, 4), (3, 7)], [(2, 6), (4, 8)], [(3, 4), (2, 6)], [(4, 5), (1, 9)], [(7, 8), (2, 10)], [(7, 8), (7, 10)], [(8, 10), (5, 9)]]
    C = [[(1, 3), (2, 5)], [(3, 4), (2, 6)], [(2, 5), (2, 9)], [(2, 6), (2, 9)], [(7, 10), (7, 9)], [(7, 8), (7, 10)]]
    
    root = BUILD([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], C)
    
    DotExporter(root, nodeattrfunc=nodeattrfunc, edgeattrfunc=lambda parent, child: "dir=none").to_picture("root.png")


