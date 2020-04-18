import numpy as np

def project_onto_simplex(y, normalize=True):
    """
    Project onto Unit simplex, using algorithm: https://arxiv.org/pdf/1101.6081.pdf
    Solves constrained minimization problem:
        argmin 0.5 * ||x - y|| ^ 2
        subject to x >= 0
                   1^T * x == 1

    @param y           Input vector
    @param normalize   Normalize input vector before projecting
    """
    sort_ind = np.argsort(-y) # descending order

    if normalize:
        y_ = y / np.sum(y)
    else:
        y_ = y
    s = y_[sort_ind]

    m = len(y)
    bget = False
    tmpsum = 0

    for i in range(m - 1):
        tmpsum = tmpsum + s[i];
        tmax = (tmpsum - 1) / (i + 1);
        if tmax >= s[i + 1]:
            bget = True
            break
        
    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    x = np.maximum(y_ - tmax, 0)
    return x


def iterative_project_onto_simplex(_y, normalize=True, simplex_size=1.0, eps=0.001):
    if normalize:
        y =  _y / np.sum(_y)
    else:
        y = _y
    r = simplex_size
    paramMu = np.min(y) - r
    objFun  = np.sum(np.maximum(y - paramMu, 0)) - r

    while np.abs(objFun) > eps:
        objFun  = np.sum(np.maximum(y - paramMu, 0)) - r
        df      = np.sum(-(((y - paramMu) > 0) + 0.0))
        paramMu = paramMu - (objFun / df)

    return np.maximum(y - paramMu, 0)


def _l2_solve(X, target_center, lam=0):
    import cvxpy as cp
    # Solve the system
    q = cp.Constant(value=target_center.flatten())
    x_ = cp.Constant(value=X)

    w = cp.Variable(X.shape[0])
    M = np.linalg.norm(X) ** 2 # target_X)
    print("M:", M)
    if lam == 0:
        print("No regularization")
        # cp.norm2(cp.matmul(X, beta) - Y)**2
        objective = cp.Minimize(cp.sum_squares(q.T - w.T @ x_)) # cp.Minimize(cp.sum_squares(q - x_ * w))
    else:
        objective = cp.Minimize(cp.sum_squares(q.T - w.T @ x_) / M + lam * cp.norm2(w)) # + lam * cp.norm2(w))
    constraints = [w >= 0, cp.sum_entries(w) == 1]
    prob = cp.Problem(objective, constraints)

    print("Problem is prepared")

    try:
        result = prob.solve()
    except Exception as ex:
        print("Exception occurred: {}".format(ex))
        print("Using SCS solver")
        result = prob.solve(solver=cp.SCS, verbose=False)
    print("Problem status: {}".format(prob.status))
    weights = w.value.A.flatten()
    print(weights)
    weights[weights < 0] = 0
    weights_sum = np.sum(weights)
    print("Weights sum: {}".format(weights_sum))
    if weights_sum != 1.0: # probably always true
        weights /= weights_sum
    return weights

if __name__ == "__main__":
    # A = np.array([[0.1, 0.5, 0.8, 0.1], [0.9, 1.0, 1.7, 3.9], [1.5, 3.9, 5.0, 1.9]])
    A = np.random.rand(10, 5)
    X = A
    # target_X = A.copy()
    target_X = np.random.rand(1, 5)

    target_center = np.mean(target_X * 2, axis=0) # * X.shape[0]
    # Solve the system
    w = np.linalg.lstsq(X.T, target_center.T, rcond='warn')[0]
    weights = project_onto_simplex(w, normalize=False)
    print(f"Weights sum: {np.sum(weights)}")
    print(f"Weights: {weights}")
    l2_weights = _l2_solve(X, target_center)
    print("L2 weights: {}".format(l2_weights))
    print("Difference: {}".format(np.sum(np.abs(weights - l2_weights))))