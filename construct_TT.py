import numpy as np
from time import perf_counter as tpc
from numba import jit, njit
import scipy.sparse


class Sparse3D():
    def __init__(self, idx=None, dtype=float, mats=None):
        if mats is None:
            self.shape = idx
            i, j, k = idx
            self.mats = [scipy.sparse.lil_array((i, k), dtype=dtype) for _ in range(j)]
        else:
            j = len(mats)
            i, k = mats[0].shape
            self.shape = i, j, k
            self.mats = mats

        self.dtype = dtype

    def __getitem__(self, idx):

        none_slice = slice(None, None, None)
        mats = self.mats
        try:
            if isinstance(idx[0], slice) and isinstance(idx[2], slice) \
                    and idx[0] == none_slice and idx[2] == none_slice:
                return mats[idx[1]]
        except TypeError:
            pass

        if isinstance(idx, int):
            return mats[idx]

        if len(idx) == 2:
            idx = (idx[0], idx[1], none_slice)

        if len(idx) == 3:
            i, k, j = idx
            try:
                k = int(k)
            except TypeError:
                return Sparse3D(mats=[ mm[i, j] for mm in mats[k] ], dtype=self.dtype)

            return mats[k][i, j]

        raise


    def __setitem__(self, idx, val):
        try:
            if isinstance(idx[0], slice) and isinstance(idx[2], slice):
                idx = (idx[1], )
        except TypeError:
            pass


        if len(idx) == 1:
            idx = idx[0]
            try:
                for mi in range(self.shape[0]):
                    for mj in range(self.shape[-1]):
                        self.mats[idx][mi, mj] = val[mi][mj]
            except Exception as e:
                self.mats[idx] = val

        elif len(idx) == 3:
            i, k, j = idx
            self.mats[k][i, j] = val

    @property
    def full(self):
        res = np.empty(self.shape, dtype=self.mats[0].dtype)
        for i, mat in enumerate(self.mats):
            mat.tolil().todense(out=res[:, i, :])

        return res

    @property
    def T(self):
        return Sparse3D(mats=[ mm.T for mm in self.mats], dtype=self.dtype)


    def to_mat(self, full=False, dtype=None):
        if dtype is None:
            dtype = self.dtype
        r1, n, r2 = self.shape
        res = None
        if r1 == 1:
            res = scipy.sparse.lil_array((n, r2), dtype=dtype)
            for mi, m in enumerate(self.mats):
                mcoo = m.tocoo()
                for i, j in zip(mcoo.col, mcoo.data):
                    res[mi, i] = j
        elif r2 == 1:
            res = scipy.sparse.lil_array((r1, n), dtype=dtype)
            for mi, m in enumerate(self.mats):
                mcoo = m.tocoo()
                for i, j in zip(mcoo.row, mcoo.data):
                    res[i, mi] = j

        else:
            assert False, "No mat!"

        if res is not None and full:
            res = res.todense()

        return res


    def mul(self, other, left=False):
        for mi, m in enumerate(self.mats):
            if left:
                self.mats[mi] = (other @ m).tolil()
            else:
                self.mats[mi] = (m @ other).tolil()

    def set_type(self, dtype):
        self.mats = [m.astype(dtype) for m in self.mats]


    def to_alg(self, l2r=True, cstyle=True, info=None, brakets="()", *, extra_info=True, letters='auo',
               indim_rename=None, outdim_rename=False):


        if len(letters) == 3 and cstyle:
            la, lb, lo = letters
        else:
            la, lb = letters[:2]


        if l2r:
            mats = self.mats
        else:
            mats = [m.T for m in self.mats]

        if info is None:
            info = dict()

        r1, r2 = mats[0].shape
        res  = [[] for _ in range(r2)]
        a_list = []
        out_list = []
        for i, m in enumerate(mats):
            ss = f"{lb}[{i}]*"
            res_cur = dict()
            m = m.tocoo()
            for c, r, val in zip(m.col, m.row, m.data):
                if val == 0:
                    continue

                if val != 1:
                    addstr = f"*{val}"
                else:
                    addstr = ""

                if extra_info:
                    if not r in a_list:
                        a_list.append(r)

                if not indim_rename is None:
                    r = indim_rename[r]
                    if r == -1:
                        continue

                res_cur[c] = res_cur.get(c, []) + [f"{la}[{r}]" + addstr]

            for c, vv in res_cur.items():
                if len(vv) > 1:
                    mn = "(" + " + ".join(vv) + ")"
                else:
                    mn = vv[0]

                res[c].append(ss + mn)
                if c not in out_list:
                    out_list.append(c)

        if extra_info:
            in_dim = max(a_list) + 1 if len(a_list) > 0 else 0
            info["in_dim"] = in_dim
            info["in_dim_missing"] = np.setdiff1d(np.arange(in_dim), a_list)


        out_list_miss =  np.setdiff1d(np.arange(r2), out_list)

        info["out_missing"] = out_list_miss
        info["out_dim"] = r2 - len(out_list_miss)


        if outdim_rename:
            if len(out_list_miss) == 0:
                outdim_rename_idx = None
            else:
                outdim_rename_idx = np.full(r2, -1)
                for i, c in enumerate(out_list):
                    outdim_rename_idx[c] = i

            info["out_rename"] = outdim_rename_idx

        if cstyle:
            txt = ""
            for i, v in enumerate(res):
                ll = " + ".join(v)
                if outdim_rename and outdim_rename_idx is not None:
                    i = outdim_rename_idx[i]
                if i >= 0:
                    txt += f"{lo}[{i}] = {ll};\n"

        else:
            txt = brakets[0]
            for i, v in enumerate(res):
                ll = " + ".join(v)
                if not outdim_rename or len(ll) > 0:
                    txt += ll + ", "

            txt += brakets[1]

        info["muls"] = txt.count("*")

        return txt


    def to_alg_fin(self, l2r=True, cstyle=True, info=None, brakets="()", extra_info=True, letters='abo'):
        """
        out var is the middle var of the core (dim `n`)
        l2r doesnot affect on result, but may affect on number of operation
        """

        if len(letters) == 3 and cstyle:
            la, lb, lo = letters
        else:
            la, lb = letters[:2]

        if l2r:
            mats = self.mats
        else:
            mats = [m.T for m in self.mats]

        if info is None:
            info = dict()
            extra_info = False


        r1, r2 = mats[0].shape
        res  = [[] for _ in range(len(mats))]
        a_list = []
        b_list = []
        for i, m in enumerate(mats):
            res_cur = dict()
            m = m.tocoo()
            for c, r, val in zip(m.col, m.row, m.data):
                if val == 0:
                    continue

                if val != 1:
                    addstr = f"*{val}"
                else:
                    addstr = ""

                if extra_info:
                    if not r in a_list:
                        a_list.append(r)
                    if not c in b_list:
                        b_list.append(c)

                res_cur[c] = res_cur.get(c, []) + [la + f"[{r}]" + addstr]

            res[i] = []
            for c, vv in res_cur.items():
                ss = lb + f"[{c}]*"
                if len(vv) > 1:
                    mn = "(" + " + ".join(vv) + ")"
                else:
                    mn = vv[0]

                res[i].append(ss + mn)

        if extra_info:
            in_dim = max(a_list) + 1
            info["in_dim"] = in_dim
            info["in_dim_missing"] = np.setdiff1d(np.arange(in_dim), a_list)

            in_dim2 = max(b_list) + 1
            info["in_dim2"] = in_dim2
            info["in_dim_missing2"] = np.setdiff1d(np.arange(in_dim2), b_list)



        info["out_dim"] = len(res)
        if cstyle:
            txt = ""
            for i, v in enumerate(res):
                ll = " + ".join(v)
                txt += lo + f"[{i}] = {ll};\n"

        else:
            txt = brakets[0]
            for i, v in enumerate(res):
                ll = " + ".join(v)
                txt += ll + ", "

            txt += brakets[1]

        info["muls"] = txt.count("*")

        return txt

    @property
    def nnz(self):
        return sum([i.nnz for i in self.mats])

    def __repr__(self):
        return f"Sp3D core {self.shape}, {self.nnz} nnz"


def is_perm_mat(mat):
    mcoo = mat.tocoo()
    rd = dict()
    cd = dict()
    cnt = 0
    for r, c, val in zip(mcoo.row, mcoo.row, mcoo.data):
        if val == 0:
            continue

        cnt += 1
        if val != 1:
            return False

        if r in rd:
            return False
        rd[r] = 1

        if c in cd:
            return False

        cd[c] = 1


    return len(rd) == len(cd) == cnt


def G0(n):
    res = np.zeros([n, n], dtype=int)
    for i in range(n):
        res[i, :i+1] = 1
    return res

def main_core(f, n, m):
    return main_core_list([f(i) for i in range(n)], n, m)

def main_core_list(f, n, m):
    """
    Constructs a functional core, it is assumed that
    f: [0, n-1] -> [0, m-1]
    """

    row, col, data = [], [], []

    f0 = f[0]
    row.extend([0]*(f0+1))
    col.extend(list(range(f0+1)))
    data.extend([1]*(f0+1))

    for i in range(1, n):
        f0_prev = f0
        f0 = f[i]
        if f0 > f0_prev:
            d = f0 - f0_prev
            row.extend([i]*d)
            col.extend(list(range(f0_prev+1, f0+1) ))
            data.extend([1]*d)

        if f0 < f0_prev:
            d = f0_prev - f0
            row.extend([i]*d)
            col.extend(list(range(f0+1, f0_prev+1) ))
            data.extend([-1]*d)

    mat = csc_matrix((data, (row, col)), shape=(n, m))
    #return lil_matrix(mat)
    return mat

#@njit
def main_core_list_ex(f, n=None, m=None, res=None, fill=None):
    """
    f: [0, n-1] -> [0, m-1]
    """

    if res is None:
        res = np.zeros((n, m))

    if fill is None:
        fill = [1]*len(f)


    for i, v in enumerate(f):
        if v >= 0:
            res[i, v] = fill[i]
    return res


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==



def next_core(f_list, v_in, v_out=None, to_ret_idx=True, last_core=False):
    """
    if last_core then fill with core with true value
    """

    # vals may contain None, so no numpy array
    if last_core:
        vals = [[(f(v) or 0 ) for v in v_in] for f in f_list]
    else:
        vals = [[f(v) for v in v_in] for f in f_list]

    if v_out is None:
        v_out = set([])
        for v in vals:
            v_out |= set(v)

    v_out = sorted(set(v_out) - set([None]))

    inv_idx = {v: i for i, v in enumerate(v_out)}
    inv_idx[None] = -1

    n, m = len(v_in), len(v_out)
    if last_core:
        m = 1
    core = np.zeros([n, len(f_list), m])
    res = []
    for i, vf in enumerate(vals):
        if last_core:
            print(vf)
            res.append(np.array(vf, dtype=float))
            main_core_list(np.zeros(len(vf), dtype=int), core[:, i, :], fill=res[-1])
        else:
            res.append(np.array([inv_idx[j] for j in vf]))
            main_core_list(res[-1], core[:, i, :])

    if to_ret_idx:
        return core, v_out, res
    else:
        return core, v_out


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==

def add_dim_core(n, m, d):
    res = np.zeros([n, d, m])
    range_l = range(min(n, m))
    res[range_l, :, range_l] = 1

    return res

def insert_dim_core(cores, i, d):

    if i == 0 or i > len(cores) - 1:
        n = 1
    else:
        n = cores[i-1].shape[-1]

    cores = cores[:i] + [add_dim_core(n, n, d)] + cores[i:]

def const_func(x):
    return lambda y: x

def add_func(x):
    return lambda y: y + x

def ind_func(x):
    return lambda y: (0 if x == y else None)

def gt_func(x):
    return lambda y: (0 if y >= x else None)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==




def _reshape(a, shape):
    return np.reshape(a, shape, order='F')

def matrix_svd(M, delta=1E-8, rmax=None, ckeck_zero=True):
    # this function is a modified version from ttpy package, see https://github.com/oseledets/ttpy
    if M.shape[0] <= M.shape[1]:
        cov = M.dot(M.T)
        singular_vectors = 'left'
    else:
        cov = M.T.dot(M)
        singular_vectors = 'right'

    if ckeck_zero and np.linalg.norm(cov) < 1e-14:
    #if np.abs(cov.reshape(-1)).sum() < 1e-14:
        return np.zeros([M.shape[0], 1]), np.zeros([1, M.shape[1]])

    w, v = np.linalg.eigh(cov)
    w[w < 0] = 0
    w = np.sqrt(w)
    svd = [v, w]
    idx = np.argsort(svd[1])[::-1]
    svd[0] = svd[0][:, idx]
    svd[1] = svd[1][idx]
    S = (svd[1]/svd[1][0])**2
    where = np.where(np.cumsum(S[::-1]) <= delta**2)[0]
    if len(where) == 0:
        rank = max(1, min(rmax, len(S)))
    else:
        rank = max(1, min(rmax, len(S) - 1 - where[-1]))

    left = svd[0]
    left = left[:, :rank]

    if singular_vectors == 'left':
        M2 = ((1. / svd[1][:rank])[:, np.newaxis]*left.T).dot(M)
        left = left*svd[1][:rank]
    else:
        M2 = M.dot(left)
        left, M2 = M2, left.T

    return left, M2



def show(N, R):
    # N = [G.shape[1] for G in Y]
    # R = [G.shape[0] for G in Y] + [1]
    l = max(int(np.ceil(np.log10(np.max(R)+1))) + 1, 3)
    form_str = '{:^' + str(l) + '}'
    s0 = ' '*(l//2)
    s1 = s0 + ''.join([form_str.format(n) for n in N])
    s2 = s0 + ''.join([form_str.format('/ \\') for _ in N])
    s3 = ''.join([form_str.format(r) for r in R])
    print(f'{s1}\n{s2}\n{s3}\n')



def full(Y):
    """Returns the tensor in full format."""
    Q = Y[0]
    for y in Y[1:]:
        Q = np.tensordot(Q, y, 1)
    return Q[0, ..., 0]


@njit
def main_core_list_old(f, res, fill=None):
    """
    f: [0, n-1] -> [0, m-1]
    """

    if fill is None:
        fill = np.ones(len(f))

    for i, v in enumerate(f):
        if v >= 0:
            res[i, v] = fill[i]
    return res

def main_core_list_py(f, res, left_to_right=None):
    """
    f: [0, n-1] -> [0, m-1]
    """

    if left_to_right:
        for i, v in enumerate(f):
            if v >= 0:
                res[i, v] = 1
    else:
        for i, v in enumerate(f):
            if v >= 0:
                res[v, i] = 1

    return res

main_core_list = njit(main_core_list_py)

def mid_avr(l):
    return (l[0] + l[-1])/2

@njit
def _next_indices_1(vals_np, eps):
            idx_rank = []
            se = vals_np[0]
            i = 0
            if vals_np[1] - se >= eps:
                idx_rank.append(i)
            len_vals_np_m_1 = len(vals_np) - 1
            #while len(idx_rank) < max_rank:
            while i < len_vals_np_m_1:
                i = min(np.searchsorted(vals_np, se + eps, side='right'), len_vals_np_m_1)
                se = vals_np[i]
                idx_rank.append(i)
                #if i == len_vals_np_m_1:
                #    flag = False
                #    break

            return idx_rank

@njit
def mean_avr_list(vals_np, idx_rank):
    n = len(idx_rank) - 1
    res = np.empty(n)
    for i in range(n):
        i_s = idx_rank[i]
        i_e = idx_rank[i+1]
        if i_e - i_s > 1:
            res[i] = (vals_np[i_s+1] + vals_np[i_e])/2.
        else:
            res[i] = vals_np[i_s+1]

    return res




def next_indices(f_list, v_in, v_out=None, max_rank=None, relative_eps=None):
    """
    if last_core then fill with core with true value
    """

    # vals may contain None, so no numpy array
    vals = [[f(v) for v in v_in] for f in f_list]
    #print(max_rank, vals)
    if max_rank is None and relative_eps is None:

        if v_out is None:
            v_out = set([])
            for v in vals:
                v_out |= set(v)

        v_out = set(v_out) - set([None])
        #v_out = list(v_out)
        try:
            v_out = sorted(v_out)
        except: # Can't sort as it's not a regular type
            pass

        inv_idx = {v: i for i, v in enumerate(v_out)}
        inv_idx[None] = -1

        res = np.array([[inv_idx[j] for j in vf] for vf in vals])

    else:

        #if isinstance(vals[0][0], complex):
        #    dtype=complex
        #else:
        #    dtype=float

        #print(vals, dtype)
        #dtype=float
        vals_np = np.unique(np.array(vals, dtype=float).reshape(-1))
        vals_search = v_out = vals_np = vals_np[:np.searchsorted(vals_np, np.nan)]

        if max_rank is None:
            max_rank = 2**30

        if relative_eps is not None and len(vals_np) > 1:
            eps = relative_eps*(vals_np[-1] - vals_np[0])
            idx_rank = _next_indices_1(vals_np, eps)

        else:
            idx_rank = np.arange(len(vals_np))


        if max_rank < len(idx_rank):
            idx_rank = np.asarray(idx_rank)
            idx_rank = idx_rank[np.linspace(-1, len(idx_rank)-1, max_rank+1).round().astype(int)[1:]]


        if len(vals_np) > len(idx_rank):
            vals_search = vals_np[idx_rank]
            idx_rank = [-1] + list(idx_rank)
            v_out = [mid_avr(vals_np[i_s+1:i_e+1]) for i_s, i_e in zip(idx_rank[:-1], idx_rank[1:])  ]
            #v_out = mean_avr_list(vals_np, idx_rank)


        res = np.array([[np.searchsorted(vals_search, j) if j is not None else -1 for j in vf] for vf in vals])
        #print(res, v_out, len(v_out))


    return v_out, res


def all_sets(ar):
    vals = np.unique(ar)
    N = np.arange(len(ar))

    return [set(N[ar==v]) for v in vals]


def pair_intersects(set1, set2):
    """
    arguments and return  -- list of sets
    """
    res = []
    for s1 in set1:
        for s2 in set2:
            cur_set = s1 & s2
            if cur_set:
                res.append(cur_set)

    return res


def all_intersects(sets):
    set1 = sets[0]
    for set2 in sets[1:]:
        set1 = pair_intersects(set1, set2)

    #return sorted(set1, key=min) # sorting only for convience. Remove when otdebuged
    return set1


def reindex(idxx):

    d = len(idxx)
    res = [None]*d

    idx_cur = idxx[d-1]
    for i in range(d-1, 0, -1):
        s_all = all_intersects([all_sets(v) for v in idx_cur])

        idxx_new = [ [] for _ in  idx_prev]
        for s in s_all:
            for v, v_prev in zip(idxx_new, idx_cur):
                v.append(v_prev[min(s)])

        res[i] = np.array(idxx_new)

        # Alter prv indices
        idx_cur = np.array(idxx[i-1], copy=True, dtype=int) # это уже индескы меньше d-1, поэтому int

        for i, s in enumerate(s_all):
            for v in s:
                #print(v, idx_cur)
                idx_cur[idx_cur==v] = i


    res[0] = idx_cur
    return res


def build_cores_by_indices(idxx, left_to_right=True, sparse=False):
    # global main_core_list
    if not idxx:
        return []

    d = len(idxx)
    cores = [None]*d

    for i, idx in enumerate(idxx):
        n = idx.shape[0]
        r0 = idx.shape[1]
        r1 = max(idx.max() + 1, 1)
        if not left_to_right:
            r1, r0 = r0, r1

        if sparse:
            core = Sparse3D([r0, n, r1], dtype=int)
            f_main_core_list = main_core_list_py
        else:
            core = np.zeros([r0, n, r1])
            f_main_core_list = main_core_list

        for j, vf in enumerate(idx):
            f_main_core_list(vf, core[:, j, :], left_to_right=left_to_right)

        cores[i] = core

    return cores

def build_core_by_vals(func, vals_in_out, sparse=False):
    vals_in, vals_out = vals_in_out
    n = len(func)

    r1 = len(vals_in)
    r2 = len(vals_out)
    if sparse:
        core = Sparse3D([r1, n, r2])
    else:
        core = np.empty([r1, n, r2])

    for k in range(n):
        core[:, k, :] = [[(func[k](i, j) or 0) for j in vals_out] for i in vals_in]

    return core


@njit
def TT_func_mat_vec(vec, idx, res=None, direction=True):
    """
    direction -- forward or backward mul (analog left_to_right)
    """
    num_sum = 0
    if res is None:
        res_len = idx.max() + 1
        res = np.zeros(res_len, dtype=vec.dtype)
    if direction:
        for i, v in enumerate(idx):
            if v >= 0:
                res[v] += vec[i]
                num_sum += 1
    else:
        for i, v in enumerate(idx):
            if v >= 0:
                res[i] += vec[v]
                num_sum += 1

    return res, num_sum

def TT_func_mat_mul(mat, idx, res=None, direction=True):
    if res is None:
        if direction:
            res = np.zeros((mat.shape[0], idx.max() + 1))
        else:
            res = np.zeros((idx.shape[1], mat.shape[1]))

    return _TT_func_mat_mul(mat, idx, res, direction)

@njit
def _TT_func_mat_mul(mat, idx, res, direction=True):
    """
    direction -- forward or backward mul (analog left_to_right)
    """

    if direction:
        for i, v in enumerate(idx):
            if v >= 0:
                res[:, v] += mat[:, i]
    else:
        for i, v in enumerate(idx):
            if v >= 0:
                res[i, :] += mat[v, :]

    return res


# -=-=-=-=-=-

def make_two_arg(func):
    return lambda x, y: func(x)

def reindex_None(idx):
    res = []
    for i, col in enumerate(idx.T):
        nqu = np.unique(col)
        if len(nqu) == 1 and nqu[0] == -1:
            res.append(i)

    n = idx.shape[1]
    res = np.array(res)
    return res, idx[:, np.setdiff1d(np.arange(n), res)]


def reindex_None_all(idxs):
    d = len(idxs)
    for i in range(d-1, -1, -1):
        bad_idx, idxs[i] = reindex_None(idxs[i])
        if i > 0:
            idx_prev = idxs[i - 1]
            for bi in sorted(bad_idx)[::-1]:
                idx_prev[idx_prev == bi] = -1
                idx_prev[idx_prev > bi] -= 1

            idxs[i - 1] = idx_prev

    resort_first_idx(idxs)
    resort_last_idx(idxs)


def invert_idx(idx):
    idx_invs = np.empty_like(idx)
    idx_invs[idx] = np.arange(idx.size)
    return idx_invs



def resort_first_idx(idxs):
    if len(idxs) < 2:
        return

    idx_minus = np.where(idxs[0][:, 0] < 0)[0]
    N = idxs[0].shape[0]

    if len(idx_minus) == 0:
        idx = idx_srt_internal = np.argsort(idxs[0][:, 0])

    else:
        idx_norm  = np.where(idxs[0][:, 0] >= 0)[0]
        idx_srt_internal = np.argsort(idxs[0][idx_norm, 0])
        idx = np.arange(N)
        idx[idx_norm] = idx[idx_norm][idx_srt_internal]

        print(f"Warning, indices of the first core ({idx_minus}) are not in use. Consider to redefine tensor.")

    idxs[0] = idxs[0][idx]
    idxs[1] = idxs[1][:, invert_idx(idx_srt_internal)]



def resort_last_idx(idxs):
    if len(idxs) < 2:
        return

    arg = []
    for ii in idxs[-1].T:
        idx_w = np.where(ii >= 0)[0]
        if len(idx_w) > 0:
            arg.append(idx_w[0])
        else:
            arg.append(len(ii))

    idx = np.argsort(arg)
    idxs[-1] = idxs[-1][:, idx]

    mask = idxs[-2] >= 0
    idxs[-2][mask] = invert_idx(idx)[idxs[-2][mask]]


class tens(object):


    @property
    def full(self):
        return full(self.cores)

    def mat_mul(self, n, i, mat, direction=True, res=None):
        """
        matmul silece $i$ of core $n$ on mat (left or right)
        """
        if n == self.pos_val_core or self._cores:
            if direction:
                if res is not None:
                    res += mat @ self.core(n)[:, i, :]
                    return res
                else:
                    return mat @ self.core(n)[:, i, :]
            else:
                if res is not None:
                    res += self.core(n)[:, i, :] @ mat
                    return res
                else:
                    return self.core(n)[:, i, :] @ mat

        if n < self.pos_val_core:
            return TT_func_mat_mul(mat, self.indices[0][n][i], res=res, direction=direction)
        if n > self.pos_val_core:
            return TT_func_mat_mul(mat.T, self.indices[1][self.d-1 - n][i], res=res.T, direction=not direction)

    def test_mat_mul(self):
        mat = np.array([[1]])
        for n in range(self.pos_val_core):
            #mat = sum(self.mat_mul(n, i, mat, direction=True) for i in range(len(self.indices[0][n])))
            res = np.zeros([mat.shape[0], self.indices[0][n].max()+1])
            for i in range(len(self.indices[0][n])):
                self.mat_mul(n, i, mat, direction=True, res=res)
            mat = res


        mat = sum(self.mat_mul(self.pos_val_core, i, mat, direction=True) for i in range(len(self.funcs_vals)))

        for n in range(self.pos_val_core+1, self.d):
            #mat = sum(self.mat_mul(n, i, mat, direction=True) for i in range(len(self.indices[1][self.d-1 - n])))
            res = np.zeros([mat.shape[0], self.indices[1][self.d-1 - n].shape[1]])
            for i in range(len(self.indices[1][self.d-1 - n])):
                self.mat_mul(n, i, mat, direction=True, res=res)

            mat = res


        return mat.item()

    def convolve(self, t, or1='C', or2='F'):
        """
        convolve two TT-tensors, calculationg tensor product throu vectorization
        """
        shapes = self.shapes
        assert (shapes == t.shapes).all()

        mat = np.array([[1]])

        for n in range(self.d):
            res = np.zeros(self.cores_shape(n)[1]*t.cores_shape(n)[1])
            mat = mat.reshape(-1, self.cores_shape(n)[0], order=or1)
            for i in range(shapes[n]):
                tmp = self.mat_mul(n, i, mat, direction=True)
                tmp2 = t.mat_mul(n, i, tmp.T, direction=True)
                #print(tmp.shape, tmp2.shape, res.shape)
                res += tmp2.reshape(-1, order=or2)

            mat = res

        return mat.item()

    def cores_shape(self, n):
        if self._cores_shapes[n] is not None:
            return self._cores_shapes[n]

        if n < self.pos_val_core:
            idx = self.indices[0][n]
            cs = (idx.shape[1], idx.max() + 1)

        if n > self.pos_val_core:
            idx = self.indices[1][self.d-1 - n]
            cs = (idx.max() + 1, idx.shape[1])

        if n == self.pos_val_core:
            cs = self.core(n).shape
            cs = (cs[0], cs[-1])

        self._cores_shapes[n] = cs
        return cs


    def __init__(self, funcs=None, *, indices=None, indicator=False,
                 do_reverse=False, do_truncate=False, do_None_clean=False,
                 v_in=None, debug=True, relative_eps=None, max_rank=None):

        self.indicator = False # no need in mid tensor, use only indices
        if type(funcs[0][0]) == list: # new
            self.funcs_left  = funcs[0]
            self.funcs_right = funcs[1]
            self.funcs_vals  = funcs[2]
        else:
            self.indicator = indicator # no need in mid tensor, use only indices

            self.funcs_left  = funcs if indicator else funcs[:-1]
            self.funcs_right = []
            self.funcs_vals = []
            _="""
            for i, fi in enumerate(funcs[-1]):
                #f =
                #f(0, 0)
                #self.funcs_vals.append(lambda x, y: funcs[-1][i](x))
                self.funcs_vals.append(make_two_arg(funcs[-1][i]))
                self.funcs_vals[-1](0, 0)

            self.funcs_vals[0](0, 0)
            self.funcs_vals[1](0, 0)
            """
            self.funcs_vals = [make_two_arg(i) for i in funcs[-1]]



        #self.funcs = funcs
        self.d = len(self.funcs_left) + len(self.funcs_right) + (not self.indicator)
        self.pos_val_core = len(self.funcs_left)
        self.do_reverse = do_reverse
        self.do_None_clean = do_None_clean
        self.do_truncate = do_truncate and not self.funcs_right
        self._cores = None
        self._cores_sparse = None
        self._indices = indices
        self.debug = debug
        self._cores_shapes = [None]*self.d
        if v_in is None:
            self.v_in = 0
        else:
            self.v_in = v_in

        self.relative_eps = relative_eps
        self.max_rank = max_rank


    def p(self, mes):
        if self.debug:
            print(mes)

    @property
    def indices(self):
        if  self._indices is not None:
            return  self._indices

        self._indices = []

        def build_i(funcs):
            v_in = [self.v_in]
            idxx_a = []
            for func in funcs:
                v_in, idxx = next_indices(func, v_in, max_rank=self.max_rank, relative_eps=self.relative_eps)
                idxx_a.append(idxx)

            if self.do_None_clean:
                assert  len(idxx_a[0]) > 0, 'Derivative functions gives whole zero tensor. Not compatible with argument "do_None_clean=True"'
                reindex_None_all(idxx_a)
            v_out_left = v_in
            return idxx_a, v_in


        idxx_a, v_out_left = build_i(self.funcs_left)
        # self._indices.append(idxx_a)

        if self.indicator:
            idx_mid = np.copy(idxx_a[-1])
            for i, val in enumerate(v_out_left):
                # assert val in [0, 1]
                idxx_a[-1][idx_mid==i] = val - 1

            self._indices.extend([idxx_a, [], []])
        else:
            self._indices.append(idxx_a)

            idxx_a, v_out_right = build_i(self.funcs_right)
            self._indices.append(idxx_a)

            self._indices.append([v_out_left,  v_out_right])

        return  self._indices

    @indices.setter
    def indices(self, indices):
        #print("Don't bother me!")
        self._indices = indices


    def get(self, idx, use_mid=None, use_cores=False):
        if use_mid is None:
            use_mid = use_cores

        assert len(idx) == self.d, f"Wrong idx length: {len(idx)} != {self.d}"

        if use_cores:
            cores = self.cores
            G = cores[0][0, idx[0]]
            for Y, ii in zip(cores[1:], idx[1:]):
                G = G @ Y[:, ii, :]

            return G.item()

        else: # use indeces
            def get_val(indeces, idx):
                val = 0
                for ind, ii in zip(indeces, idx):
                    val = ind[ii][val]
                    if val < 0:
                        return -1
                return val

            val_l = get_val(self.indices[0], idx)
            if val_l < 0:
                return 0
            if self.indicator:
                return val_l + 1 # 0 or 1 or more

            val_r = get_val(self.indices[1], idx[::-1])
            if val_r < 0:
                return 0


            pos = self.pos_val_core
            k = idx[pos]
            if use_mid:
                mid_core = self.core(pos, skip_build=True)
                return mid_core[val_l, k,  val_r]
            else: # get value from the func
                vals_in, vals_out = self.indices[2]
                return self.funcs_vals[k](vals_in[val_l], vals_in[val_r]) or 0



    def make_func_for_convolv(self, mid=-1, use_numba=False, info=None):
        cores = self.make_tails_identity()

        d = len(cores)
        if mid < 0:
            mid += d

        if info is None:
            info = dict()

        assert 0 < mid < d, "Bad middle core"

        mults = 0
        lfs = []
        rfs = []
        for gen, llist, l2r in zip([range(1, mid), range(d-2, mid-1, -1)], [lfs, rfs], [True, False]):
            for i in gen:
                cinfo = dict()
                txt = cores[i].to_alg(l2r=l2r, cstyle=False, brakets="[]", info=cinfo,  extra_info=False)
                if use_numba:
                    txt = f"np.array({txt})"

                lf = eval(f'lambda a, u: {txt}')
                if use_numba:
                    lf = njit(lf)

                llist.append(lf)

                mults += cinfo['muls']
                out_dim =  cinfo["out_dim"]

        info["mults"] = mults + out_dim

        def outF(H):
            if len(lfs) > 0:
                v1 = lfs[0](H[0], H[1])
                for i, f in enumerate(lfs[1:], start=2):
                    v1 = f(v1, H[i])
            else:
                v1 = np.asarray(H[0])


            if len(rfs) > 0:
                v2 = rfs[0](H[d-1], H[d-2])
                for i, f in enumerate(rfs[1:], start=3):
                    v2 = f(v2, H[d-i])
            else:
                v2 = np.asarray(H[d-1])


            return  np.asarray(v1) @ np.asarray(v2)

        return outF



    def make_func_for_convolv_vec(self, mid=0, mid_l2r=True, use_numba=False, info=None):
        cores = self.make_tails_identity()

        d = len(cores)
        if mid < 0:
            mid += d

        if info is None:
            info = dict()

        assert 0 <= mid < d , "Bad middle core"


        mults = 0
        lfs = []
        rfs = []
        for gen, llist, l2r in zip([range(1, mid), range(d-2, max(mid, 0), -1)], [lfs, rfs], [True, False]):
            for i in gen:
                cinfo = dict()
                txt = cores[i].to_alg(l2r=l2r, cstyle=False, brakets="[]", info=cinfo,  extra_info=False)
                if use_numba:
                    txt = f"np.array({txt})"

                lf = eval(f'lambda a, u: {txt}')
                if use_numba:
                    lf = njit(lf)

                llist.append(lf)

                mults += cinfo['muls']

        info["mults"] = mults


        if mid == d-1:
            def outF(H):
                v1 = lfs[0](H[0], H[1])
                for i, f in enumerate(lfs[1:], start=2):
                    v1 = f(v1, H[i])

                return  np.asarray(v1)

        elif mid == 0:
            def outF(H):
                v2 = rfs[0](H[d-2], H[d-3])
                for i, f in enumerate(rfs[1:], start=4):
                    v2 = f(v2, H[d-i])

                return  np.asarray(v2)

        else:

            txt = cores[mid].to_alg_fin(l2r=mid_l2r, cstyle=False, brakets="[]", info=cinfo,  extra_info=False)
            if use_numba:
                txt = f"np.array({txt})"

            mf = eval(f'lambda a, b: {txt}')
            if use_numba:
                mf = njit(mf)


            info["mults"] += cinfo['muls']


            def outF(H): # NOTE! Input H is of dimension d-2
                if len(lfs) > 0:
                    v1 = lfs[0](H[0], H[1])
                    for i, f in enumerate(lfs[1:], start=2):
                        v1 = f(v1, H[i])
                else:
                    v1 = np.array(H[0])


                if len(rfs) > 0:
                    v2 = rfs[0](H[d-2], H[d-3])
                    for i, f in enumerate(rfs[1:], start=4):
                        v2 = f(v2, H[d-i])
                else:
                    v2 = np.array(H[d-2])

                if mid_l2r:
                    res = mf(np.asarray(v1), np.asarray(v2))
                else:
                    res = mf(np.asarray(v2), np.asarray(v1))


                return  res

        return outF




    def make_tails_identity(self, sparse=True):
        if sparse:
            G0 = self.cores_sparse[0].to_mat() # (dtype=int)
            if is_perm_mat(G0):
                G0_inv = G0.T
            else:
                G0_inv = scipy.sparse.linalg.inv(G0.tocsc())

            self.cores_sparse[1].mul(G0_inv.T, left=True)
            self.cores_sparse[0].mul(G0_inv, left=False)

            G1 = self.cores_sparse[-1].to_mat() #dtype=int)
            if is_perm_mat(G1):
                G1_inv = G1.T
            else:
                G1_inv = scipy.sparse.linalg.inv(G1.tocsc())

            self.cores_sparse[-2].mul(G1_inv.T, left=False)
            self.cores_sparse[-1].mul(G1_inv, left=True)

            self.cores_sparse[0].set_type(int)
            self.cores_sparse[-1].set_type(int)

            return self.cores_sparse



    @property
    def cores_sparse(self):
        if self._cores_sparse is None:

            cores_left  = build_cores_by_indices(self.indices[0], left_to_right=True, sparse=True)
            self._cores_sparse = cores_left
            if not self.indicator:
                cores_right = build_cores_by_indices(self.indices[1], left_to_right=False, sparse=True)
                core_val = build_core_by_vals(self.funcs_vals, self.indices[2], sparse=True)
                self._cores_sparse = cores_left + [core_val] + cores_right[::-1]

        return self._cores_sparse


    @property
    def cores(self):
        if self._cores is None:

            cores_left  = build_cores_by_indices(self.indices[0], left_to_right=True)
            self._cores = cores_left
            if not self.indicator:

                cores_right = build_cores_by_indices(self.indices[1], left_to_right=False)
                try:
                    core_val = self.mid_core
                except:
                    core_val = build_core_by_vals(self.funcs_vals, self.indices[2])
                self._cores = cores_left + [core_val] + cores_right[::-1]

            if self.do_truncate:
                self.truncate()

        return self._cores

    def core(self, n, skip_build=False):
        if self._cores is None or skip_build:
            d = self.d
            if   n < self.pos_val_core:
                return build_cores_by_indices([self.indices[0][n]], left_to_right=True)[0]
            elif n > self.pos_val_core:
                return build_cores_by_indices([self.indices[1][d-1 - n]], left_to_right=False)[0]
            else:
                try:
                    return self.mid_core # mid_core does not midifyed during rounding
                except:
                    if self.indicator:
                        self.mid_core = build_cores_by_indices([self.indices[0][d-1]], left_to_right=True)[0]
                    else:
                        self.mid_core = build_core_by_vals(self.funcs_vals, self.indices[2])
                    return self.mid_core

        else:
            return self._cores[n]

    @cores.setter
    def cores(self, cores):
        if self._cores is not None:
            print("Warning: cores are already set")
        self._cores = cores

    def index_revrse(self):
        """
        Heavy procedure
        """

        self._indices = reindex(self._indices)


    def truncate_indices(self, show_half=False):
        self.indices[0] = reverse_idxs(reverse_idxs(self.indices[0], show=show_half))


    def shapes(self, func_shape=False):
        if func_shape:
            if self.indicator:
                return np.array([i.shape[0] for i in self.indices[0]])
            else:
                return np.array([i.shape[0] for i in self.indices[0]] + [ len(self.funcs_vals) ] + [i.shape[0] for i in self.indices[1]][::-1])
        else:
            return np.array([i.shape[1] for i in self.cores])


    def ranks(self, func_shape=False):
        if func_shape:
            if self.indicator:
                return np.array( [1] + [i.max() + 1 for i in self.indices[0]] )
            else:
                return np.array( [1] + [i.max() + 1 for i in self.indices[0]] + [i.max() + 1 for i in self.indices[1]][::-1] + [1] )
        else:
            return np.array([1] + [G.shape[-1] for G in self.cores])


    @property
    def erank(self):
        """Compute effective rank of the TT-tensor."""
        N = self.shapes(True)
        R = self.ranks(True)
        sz = np.dot(N * R[:-1], R[1:])
        b = N[0] + N[-1]
        a = np.sum(N[1:-1])
        return (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)

    def show(self):
        # show(self.cores)
        N = self.shapes(func_shape=True)
        R = self.ranks(True)
        show(N, R)

    def show_TeX(self, delim="---"):
        rnks = [i.shape[0] for i in self.cores] + [1]
        print(delim.join([str(j) for j in rnks]))


    def truncate(self, delta=1E-10, r=np.iinfo(np.int32).max):
        N = self.shapes
        Z = self.cores
        # We don't need to othogonolize cores here. But delta might not adcuate
        for k in range(self.d-1, 0, -1):
            M = _reshape(Z[k], [Z[k].shape[0], -1])
            L, M = matrix_svd(M, delta, r, ckeck_zero=False)
            Z[k] = _reshape(M, [-1, N[k], Z[k].shape[2]])
            Z[k-1] = np.einsum('ijk,kl', Z[k-1], L, optimize=True)

        self._cores = Z

    def reindex_None(self, num=None):
        if num is None:
            num = [0, 1]
        if isinstance(num, int):
            num = [num]
        for n in num:
            idxss = self.indices[n]
            if idxss:
               reindex_None_all(idxss)



    def simple_mean(self):
        Y = self.cores
        G0 = Y[0]
        G0 = G0.reshape(-1, G0.shape[-1])
        G0 = np.sum(G0, axis=0)
        for i, G in enumerate(Y[1:], start=1):
            G0 = G0 @ np.sum(G, axis=1)

        return G0.item()


    def simple_mean_func(self):
        k = self.pos_val_core
        d = self.d
        #print(d)

        num_op_sum = 0
        def build_vec(inds, G, head=True):
            num_op_sum = 0
            if head:
                G = G.reshape(-1, G.shape[-1])
                #num_op_sum += (G != 0).sum() # Actually, all 1 in G on deiffernet places, thus no sum
                G = np.sum(G, axis=0)
            else:
                G = G.reshape(G.shape[0], -1)
                #num_op_sum += (G != 0).sum()
                G = np.sum(G, axis=1)

            #num_op_mult = num_op_sum

            #print(G)
            for idxx in inds:
                res_l = idxx.max() + 1
                res = np.zeros(res_l, dtype=int)
                for idx in idxx:
                    _, num_sum_cur = TT_func_mat_vec(G, idx, res)
                    num_op_sum += 2*num_sum_cur  # 2* because '+' and '*' there
                #print(res)
                G = res
            return G, num_op_sum

        t0 = []
        t0.append(tpc())
        # G0 = self.core(0, skip_build=True)
        G0 = np.array([[1]], dtype=int)
        t0.append(tpc()) #1
        G0, num_op_sum_cur = build_vec(self.indices[0], G0) if k > 0 else (np.array([[1]]), 0)
        t0.append(tpc()) #2
        num_op_sum += num_op_sum_cur
        self.num_op = num_op_sum
        if self.indicator:
            return G0.item()
        # G1 = self.core(d - 1, skip_build=True)
        G1 = np.array([[1]], dtype=int)
        t0.append(tpc()) #3
        G1, num_op_sum_cur  = build_vec(self.indices[1], G1, False) if k < d - 1 else (np.array([[1]]), 0)
        num_op_sum += num_op_sum_cur
        t0.append(tpc()) #4

        # l0 = G0.size()
        #l1 = G1.size()
        mid_core = self.core(k, skip_build=True)
        t0.append(tpc()) #5
        num_op_sum += (mid_core != 0).sum()
        core_k = np.sum(mid_core, axis=1)
        t0.append(tpc()) #6
        #print(G0, core_k, G1)
        #print(G0.shape, self.core(k).shape, G1.shape)
        #print(G0, G1)
        n, m = core_k.shape

        num_op_sum += n*m + min(m, n) # mults
        num_op_sum += n*m + min(m, n) # sums
        self.num_op = num_op_sum
        times = np.array(t0)
        self._times = times[1:] - times[:-1]
        return (G0 @ core_k @ G1).item()

    def show_n_core(self, n):
        c = self.cores[n]
        for i in range(c.shape[1]):
            print(c[:, i, :])


    def mul(self, other):
        return mul(self, other)

    def mul_full(self, other, *a, **k):
        return mul_full(self, other, *a, **k)


def mul(t1, t2):
    """
    Hadamard multiply of two indices TT using only indices
    Output shapes will be multipilication of the input tensor shapes
    """

    assert t1.indicator and t2.indicator, "Use kron of cores to multiply common tensors"

    ranks1 = t1.ranks(True)[:-1]
    ranks2 = t2.ranks(True)
    shapes = t1.shapes(True)

    res = []
    for r1, r2, n in zip(ranks1, ranks2[:-1], shapes):
        res.append(np.full( [n, r1*r2], -1 ))

    for idx_res, idx1, idx2, r in zip(res, t1.indices[0], t2.indices[0], ranks2[1:]):
        for out, i1, i2 in zip(idx_res, idx1, idx2):
            kron2idx(i1, i2, r, out)

    full_clean_idx(res)

    return tens(gen_f_idxs(res), indices=[res, [], []], indicator=True, v_in=(1, ), debug=t1.debug, max_rank=t1.max_rank)


def mul_full(t1, t2, order="C", seq=False):
    """
    Hadamard multiply of two indices TT using only indices.
    Output shapes will be multipilication of the input tensor shapes
    """

    assert t1.indicator and t2.indicator, "Use kron of cores to multiply common tensors"

    ranks1 = t1.ranks(True)
    ranks2 = t2.ranks(True)
    shapes1 = t1.shapes(True)
    shapes2 = t2.shapes(True)

    res = []
    for r1, r2, n1, n2 in zip(ranks1[:-1], ranks2[:-1], shapes1, shapes2):
        res.append(np.full( [n1*n2, r1*r2], -1 ))

    for idx_res, idx1, idx2, r2, r1, n1, n2 in zip(res, t1.indices[0], t2.indices[0], ranks2[1:], ranks1[1:], shapes1, shapes2):
        #for out, i1, i2 in zip(idx_res, idx1, idx2):
        for k, out in enumerate(idx_res):
            if order == 'C':
                i1, i2 = divmod(k, n2)
            else:
                i2, i1 = divmod(k, n1)
            if seq:
                kron2idx(idx1[i1], idx2[i2], r2, out)
            else:
                kron2idx(idx2[i2], idx1[i1], r1, out)

    full_clean_idx(res)

    return tens(gen_f_idxs(res), indices=[res, [], []], indicator=True, v_in=(1, ), debug=t1.debug, max_rank=t1.max_rank, do_None_clean=True)



def mult_and_mean(Y1, Y2):
    G0 = Y1[0][:, None, :, :, None] * Y2[0][None, :, :, None, :]
    G0 = G0.reshape(-1, G0.shape[-1])
    G0 = np.sum(G0, axis=0)
    for G1, G2 in zip(Y1[1:], Y2[1:]):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)
        print(G0.shape, G.shape)
        G0 = G0 @ G

    return G0.item()

def mult_and_mean(Y1, Y2):
    G0 = np.array([[1]])
    for G1, G2 in zip(Y1, Y2):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)
        #print(G0.shape, G.shape)
        G0 = G0 @ G

    return G0.item()



def partial_mean(Y):
        G0 = Y[0]
        G0 = np.sum(G0, axis=1)
        for G in Y[1:]:
            G0 = G0 @ np.sum(G, axis=1)

        return G0


def gen_f_idxs(idxs, dtype=int):
    """
    Generate funcs that can generate the given idxs in reverse order
    """
    d = len(idxs)
    ranks = [1] + [i.max() + 1 for i in idxs]
    def gen_f(i, pos, last):
        idx = idxs[pos][i]
        r = ranks[pos]
        def f(x):
            res = np.zeros(r, dtype=dtype)
            for i, elem in enumerate(idx):
                if elem < 0:
                    continue

                res[i] += x[elem]
            if last:
                assert len(res) == 1
                return res[0]

            if (res == 0).all():
                return None
            return tuple(res.tolist())


        return f


    return [[gen_f(i, pos, pos==0)
           for i, _ in enumerate(idxs[pos])]
             for pos in range(d-1, -1, -1)]


def reverse_idxs(idxs, dtype=int, show=False, out_full_tensor=False):
    """
    reverse flow, expecting  truncation of thi
    """
    ff = gen_f_idxs(idxs, dtype=dtype)
    tt = tens(ff, v_in=(1, ), indicator=True)
    if show:
        tt.show()

    if out_full_tensor:
        return tt
    else:
        return tt.indices[0]


def full_clean_idx(res):
    """
    Sort of truncation in indices format
    """
    prev = np.array([0], dtype=int)
    for ir, rr in enumerate(res):

        add = -1 if -1 in prev else 0
        if ir > 0:
            res[ir-1] = np.searchsorted(prev, res[ir-1], side='left') + add

        if add < 0:
            prev = prev[prev >= 0]

        res[ir] = rr = rr[:, prev]


        rr_f = rr.reshape(-1)
        prev = np.unique(rr_f)

def kron2idx(idx1, idx2, r, out):
    """
    Kronecker product of cores in indices format
    """
    r1, r2 = len(idx1), len(idx2)
    mask = idx2 >=0
    for pos, i1 in zip(range(0, r1*r2, r2), idx1):
        if i1 == -1:
            continue

        idx2n = out[pos:pos + r2]
        idx2n[:] = idx2
        idx2n[mask] += i1*r

