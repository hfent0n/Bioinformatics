# The Baum Welch algorithm with log scaling and efficient use of Numpy   
import argparse
import numpy as np
from scipy.special import logsumexp
       
def elnproduct(log_A, log_B):
    mask_A = np.where(log_A == 0)
    mask_B = np.where(log_B == 0)
    prod = log_A + log_B
    try:
        prod[mask_B] == 0
    except IndexError:
        prod[mask_A] == 0
    return prod

def logmatmulexp(log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    try:
        ϴ, R = log_A.shape
        I = log_B.shape[1]
        assert log_B.shape == (R, I)
        log_A_expanded = np.broadcast_to(np.expand_dims(log_A, 2), (ϴ, R, I))
        log_B_expanded = np.broadcast_to(np.expand_dims(log_B, 0), (ϴ, R, I))
        log_pairwise_products = log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)                                                                                                                                                              
        return logsumexp(log_pairwise_products, axis=1)
    except ValueError: # for 1d array
        R = log_A.shape[0]
        I = log_B.shape[1]
        assert log_B.shape == (R, I)
        log_A_expanded = np.broadcast_to(np.expand_dims(log_A, 1), (R, I))
        log_pairwise_products = log_A_expanded + log_B # shape: (R, I)                                                                                                                                                              
        x = logsumexp(log_pairwise_products, axis=1)
        #print("matmult:\n", x)
        return logsumexp(log_pairwise_products, axis=0)

def forward(Y, A, B, pi):
    alpha = np.zeros((N, T))
    alpha[:, 0] = elnproduct(pi,  B[:, Y[0]])
    for t in range(1, T):
        alpha[:, t] = elnproduct(logmatmulexp(alpha[:, t-1], A).T,B[:, Y[t]].T)
    return alpha 

def backward(Y, A, B):
    beta = np.zeros((N, T+1))
    for t in range(T-1, -1, -1):
        beta[:, t] =  logmatmulexp(elnproduct(B[:, Y[t]], beta[:, t+1]), A.T)
    return beta    

def baum_welch(Y, A, B, pi, n_iter):
    ## Caclulate row and column indices for numpy advanced indexing
    #beta
    beta_rows, beta_columns = np.ogrid[:N, :T-1]
    beta_rows = beta_rows[np.newaxis, :, :] - S[:, np.newaxis, np.newaxis] 
    beta_rows[beta_rows < 0] += N
    #A
    A_columns = beta_rows
    A_rows = np.tile(S[:,np.newaxis], (1, T-1))
    #B
    B_rows = beta_rows
    B_columns = Y[1:]
    #a
    a_rows = beta_rows[0]
    a_columns = np.arange(N-1, -1, -1)
    r = S.reshape((-1, 1))
    a_columns = a_columns + r
    a_columns[a_columns >= N] -= N
    a_columns = np.roll(a_columns, 1, 1)
    #gamma
    gamma_rows = S[np.newaxis,:]
    gamma_rows = gamma_rows - r
    gamma_rows[gamma_rows < 0] += N  

    ## Loop the Expectation maximisation
    for _ in range(n_iter):
        alpha   = forward(Y, A, B, pi)[:, :-1]
        beta    = backward(Y, A, B)[:, 2:]
        ## Create 3d alpha and beta tables for the xi calc  
        beta = beta[beta_rows, beta_columns]
        alpha = np.tile(alpha[np.newaxis,:,:,], (N, 1, 1))

        ## Create 3d transition matrix for the xi calc
        A3d = A[A_rows, A_columns]
        
        ## Create 3d emission matrix
        B3d = B[B_rows, B_columns]

        # Calc xi and gamma
        numerator_xi = elnproduct(elnproduct(alpha, beta), elnproduct(A3d, B3d))
        denominator_xi = logsumexp(numerator_xi, axis=(0, 1))
        xi = elnproduct(numerator_xi, -denominator_xi)
        gamma = logsumexp(xi, axis=0)
        
        # Update                
        pi_ = gamma[:, 0]           
        a = elnproduct(logsumexp(xi, axis=2), -logsumexp(gamma, axis=1))
        a = a.T
        a = a[a_rows, a_columns]

        last_gamma = xi[:,:,-1]
        last_gamma = logsumexp(last_gamma[gamma_rows, S], axis=1)  
        gamma = np.hstack((gamma, last_gamma.reshape((-1, 1))))

        b = np.ones((N, K))        
        for i in range(K):
            b[:,i] = logsumexp(gamma[:, Y==i], axis=1)
        b = elnproduct(b, -logsumexp(gamma, axis=1).reshape(-1, 1))

        A = a
        B = b 
        pi = pi_

    return A, B, pi

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("N", type=int)
    cli.add_argument("--sigma", nargs="*")
    cli.add_argument("--Y", nargs="*")
    cli.add_argument("--n_iter", type=int, default=1)
    
    ## Using Wikipedia's variable names.
    opt = cli.parse_args()      
    N       = opt.N             # Number of states
    sigma   = opt.sigma         # Alphabet
    Y       = opt.Y             # Observations
    n_iter  = opt.n_iter        # Epochs to run EM for
    S       = np.arange(N)      # States
    
    K = len(sigma)              # Length of alphabet
    T = len(Y)                  # Length of sequence
    
    # Convert Sequence and alphabet to integers.
    convert = {letter: i for i, letter in enumerate(sigma)}
    Y = np.array([convert[x] for x in Y])
    sigma = np.array([convert[x] for x in sigma])
    
    print(len(Y))
    # Assign initial values for the transition, emission and starting prob matrices.
    np.random.seed(42)
    A  = np.log(np.random.dirichlet(np.ones(N), N))
    B  = np.log(np.random.dirichlet(np.ones(K), N))
    pi = np.log(np.random.dirichlet(np.ones(N), 1))

    
    A, B, pi = baum_welch(Y, A, B, pi, n_iter)
   
    print("----------------Transition-----------------")
    print(np.exp(A))
    print("-----------------Emission-------------------")
    print(np.exp(B))
    print("-----------------Initial-------------------")
    print(np.exp(pi))

