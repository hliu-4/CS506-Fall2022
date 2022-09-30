def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += abs(x[i] - y[i])
    return res

def jaccard_dist(x, y):
    if len(x) == 0:
        return 1
    
    res = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            res += 1
    return 1 - res/len(x)

def cosine_sim(x, y):
    if len(x) == 0:
        return 1
    
    dot = 0
    normx = 0
    normy = 0

    for i in range(len(x)):
        dot += x[i] * y[i]
        normx += x[i]**2
        normy += y[i]**2
    
    normx = normx**(1/2)
    normy = normy**(1/2)

    if normx == 0 or normy == 0:
        return 1
    
    return dot/(normx * normy)

# Feel free to add more
