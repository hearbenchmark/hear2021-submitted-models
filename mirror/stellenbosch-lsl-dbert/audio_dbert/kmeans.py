""" By Matthew Baas (rf5.github.io) """

import torch
import torch.nn.functional as F

def kmeans_pp_init(X, k, dist_func, tol=1e-9):
    """ 
    `X` is (d, N) , `k` is int;
    uses kmeanspp init from https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf 
    """
    means = torch.empty(X.shape[0], k, dtype=X.dtype, device=X.device)
    means[:, 0] = X[:, torch.randint(0, X.shape[1], (1, ))[0]]
    for i in range(1, k):
        D = dist_func(X, means[:, :i]).min(dim=-1).values # (N, k)
        D = torch.clamp(D, tol)
        # naive way of doing this
        # probs = D / D.sum(dim=0) 
        # smarter way of doing this to prevent numerical errors
        logp = D.log() - D.sum(dim=0).log()
        pmf = torch.distributions.Categorical(logits=logp, validate_args=False)
        ind = pmf.sample()
        means[:, i] = X[:, ind]
    return means

def euclid_dist(X, means):
    """ `X` is (d, N), `means` is (d, K), returns dist matrix of shape (N, K) """
    dist = ((X[..., None] - means[:, None])**2).sum(dim=0)
    return dist

def cosine_dist(X, means):
    """ `X` is (*_1, d), `means` is (*_2, d), returns dist matrix of shape (N, K) """
    dist = 1 - F.cosine_similarity(X[..., None], means[:, None], dim=0)
    return dist

def smart_euclid_dist(X, means):
    """ `X` is (*_1, d), `means` is (*_2, d), returns dist matrix of shape (*_1, *_2) """
    s1 = X.shape[:-1]
    s2 = means.shape[:-1]
    dim = X.shape[-1]
    means_ = means.view(*([1]*len(s1) + list(s2) + [dim,]))
    X_ = X.view(*(list(s1) + [1]*len(s2) + [dim,]))
    dist = ((X_ - means_)**2).sum(dim=-1)
    return dist

def smart_cosine_dist(X, means):
    """ `X` is (*_1, d), `means` is (*_2, d), returns dist matrix of shape (*_1, *_2) """
    s1 = X.shape[:-1]
    s2 = means.shape[:-1]
    dim = X.shape[-1]
    means_ = means.view(*([1]*len(s1) + list(s2) + [dim,]))
    X_ = X.view(*(list(s1) + [1]*len(s2) + [dim,]))
    dist = 1 - F.cosine_similarity(X_, means_, dim=-1)
    return dist

def smart_cosine_sim(X, means):
    """ `X` is (*_1, d), `means` is (*_2, d), returns dist matrix of shape (*_1, *_2) """
    s1 = X.shape[:-1]
    s2 = means.shape[:-1]
    dim = X.shape[-1]
    means_ = means.view(*([1]*len(s1) + list(s2) + [dim,]))
    X_ = X.view(*(list(s1) + [1]*len(s2) + [dim,]))
    sim = F.cosine_similarity(X_, means_, dim=-1)
    return sim

def k_means(X: torch.Tensor, k: int, tol=1e-9, times=50, dist='euclid', init='kmeanspp', verbose=True):
    """ 
    k-means for `X` (d, N) and `k` classes, where d is vector dimension and N is number of vectors. 
    Tries to fit a kmeans model `times` number of times, returning the results for the best run.
    The kmeans uses `dist` (either 'euclid' or 'cosine') for distance function.
    The kmeans uses `init` cluster initialization (either 'kmeanspp' or 'random').
    Returns (means, cluster assignments, best loss) """
    dist_func = euclid_dist if dist == 'euclid' else cosine_dist
    best_loss = torch.tensor(float('inf'), dtype=torch.float, device=X.device)
    best_means = None
    best_t_jn = None
    for t in range(times):
        if init == 'kmeanspp': means = kmeans_pp_init(X, k, dist_func)
        else: means = X[:, torch.randperm(X.shape[-1], device=X.device)[:k]] # (d, k)
        new_means = 0

        while ((new_means - means)**2).sum() > tol:
            # E step
            new_means = means
            dists = dist_func(X, means) 
            assigned_classes = dists.argmin(dim=-1)
            del dists
            t_jn = torch.zeros((X.shape[-1], k), device=X.device)
            t_jn[torch.arange(t_jn.shape[0], device=X.device), assigned_classes] = 1
            # M step
            for i in range(k):
                class_i_samples = X[:, assigned_classes == i]
                # only update the mean if a sample is assigned to this cluster.
                if class_i_samples.shape[-1] > 0: new_means[:, i] = class_i_samples.mean(dim=-1)

            # class means (d, k)
            loss = (t_jn[None] * dist_func(X, new_means) ).sum() # (d, n, k)
            
        if loss < best_loss:
            if verbose: print(f"Run {t:4d}: found new best loss: {loss:7f}")
            best_loss = loss
            best_means = new_means
            best_t_jn = t_jn
    cluster_assignments = best_t_jn.argmax(dim=-1)
    return best_means, cluster_assignments, best_loss

class SimplePCA:
    def __init__(self, mat, mtype='data'):
        """ mat is either d x N or d x d """
        if mtype == 'data':
            self.mean = torch.mean(mat,dim=1)[:,None]
            self.dstar = mat - self.mean
            self.cov_mat = (1/(mat.shape[-1] - 1)) * (self.dstar@self.dstar.T) # = S
        elif mtype == 'cov':
            self.cov_mat = mat
            self.mean = None
    def fit(self, n_components=None, whiten=False, method='e-decomposition'):
        if n_components is None:
            n_components = self.cov_mat.shape[0]
        self.whiten = whiten
        if method == 'e-decomposition':
            evals, Q = torch.linalg.eigh(self.cov_mat)
            idx = evals.argsort().flip(0)
            self._evals = evals[idx]
            evals = evals[idx][:n_components]
            self.Q = Q[:,idx]
            self.Q = self.Q[:, :n_components]
            self._Q = self.Q.clone()
            self.Lambda = torch.diag(evals)
            if self.whiten: self.Q = self.Q@torch.diag(evals**-0.5)
        elif method == 'SVD':
            u, s, vh = torch.linalg.svd(self.dstar,full_matrices=False)
            evals = (1/(self.dstar.shape[-1] - 1))*(s**2)
            self._evals = evals
            evals = evals[:n_components]
            self.Q = u[:, :n_components]
            self._Q = self.Q.clone()
            self.Lambda = torch.diag(evals)
            if self.whiten: self.Q = self.Q@torch.diag(evals**-0.5)
                
    def transform(self, data):
        """ Transform `data` (d x N') to principal axes (d x r) """
        return self.Q.T@(data - self.mean)
    
    def inverse_transform(self, data):
        """ Transform `data` (r x N) back to original axes """
        if self.whiten: return self.mean + self._Q@(self.Lambda ** 0.5)@data
        else: return self.mean + self.Q@data

if __name__ == '__main__':
    print("Running verification tests")
    N = 1000
    k = 50
    d = 25
    X = torch.rand(d, N).cuda()
    c, assignments, loss = k_means(X, k, times=1, dist='euclid', init='kmeanspp')
    print(c.shape, assignments.shape, loss)