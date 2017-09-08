from __future__ import division
from builtins import *
import numpy as np
import collections
import itertools



def weighted_var(x, w, axis=None, axis_sum=None, unbiased=True, **kwargs):
    """
    return the weighted variance over x with weight w

    v = sum(w)**2/(sum(w)**2 - sum(w**2)) * sum(w * (x-mu)**2 )

    Parameters
    ----------
    x : array
        values to consider

    w : array
        weights 

    axis : axis to average over

    axis_sum : axis to sum over for  w,w**2

    unbiased : bool (default True)
    if True, then apply unbiased norm (like ddof=1)
    else, apply biased norm (like ddof=0)


    **kwargs : arguments to np.average

    Returns
    -------
    Ave : weighted average
        shape x with `axis` removed

    Var : weighted variance 
        shape x with `axis` removed
    """

    if axis_sum is None:
        axis_sum = axis

    m1 = np.average(x, weights=w, axis=axis, **kwargs)
    m2 = np.average((x - m1)**2, weights=w, axis=axis, **kwargs)

    if unbiased:
        w1 = w.sum(axis=axis_sum)
        w2 = (w * w).sum(axis=axis_sum)
        m2 *= w1 * w1 / (w1 * w1 - w2)

    return m1, m2


def cov_nd(x,axis=0,ddof=0,**kwargs):
    """
    compute to covariance of x over axis with arbitrary dimensions
    """

    #roll axis to last
    X = np.rollaxis(x,axis,x.ndim)
    X2D = X.reshape(-1,X.shape[-1])

    Cov2D = np.cov(X2D, ddof=ddof, **kwargs)
    return Cov2D.reshape(X.shape[:-1]*2)


def _diag_nd(x):
    """
    return 'diagonal' of nd matrix
    """
    assert x.ndim %2 == 0
    if x.ndim == 0:
        return x
    else:
        shape_h = x.shape[:x.ndim//2]
        shape_r = (np.prod(shape_h),)*2
        return np.diag(x.reshape(*shape_r)).reshape(*shape_h)




def _splice_stats(Ave, weights, dtype, Var=None, Cov=None):
    """
    splice stats

    in this, assume Ave, Var, Cov is already rolled so
    axis to splice over is always zero

    intent is to call this from separate function
    """

    S = np.zeros(Ave.shape[1:], dtype=dtype)
    W = 0.0

    if Var is not None:
        V = np.zeros(Ave.shape[1:], dtype=dtype)
    else:
        V = None

    if Cov is not None:
        C = np.zeros(Cov.shape[1:], dtype=dtype)
    else:
        C = None

    for i in range(Ave.shape[0]):
        w = weights[i]

        W_last = W
        W += w

        f = w / W
        delta = Ave[i,...] - S
        deltaM = np.multiply.outer(delta, delta)

        delta *= f
        deltaM *= f * W_last

        S += delta
        if Var is not None:
            V += Var[i,...] * w + _diag_nd(deltaM)

        if Cov is not None:
            C += Cov[i,...] * w + deltaM


    if Var is not None:
        V /= W

    if Cov is not None:
        C /= W
    return W, S, V, C


def _reshape(x,axis):
    """
    roll axis to first position
    """
    if x.ndim == 1:
        return x.reshape(-1,1)
    else:
        return np.rollaxis(x,axis,0)

def _set_def_var(x, shape, axis, dtype):
    """
    create a default variable (Var/Cov below)

    x : array or None or True
        if None, return None.  If true, make zero array of shape and dtype.
        else, reshape array x with `axis`
    """

    if x is not None:
        if type(x) is bool and x:
            out = np.zeros(shape, dtype=dtype)
        else:
            out = _reshape(x, axis).astype(dtype)
    else:
        out = None
    return out


def _setup_Weight_Ave_Var_Cov(ave, weights, var, cov, axis, dtype):
    """
    setup variables depending on input
    """
    # average reshape
    Ave = _reshape(ave,axis)

    # variance
    Var = _set_def_var(var, shape=Ave.shape, dtype=dtype, axis=axis)
    if Var is not None:
        if type(var) is not bool:
            assert var.shape == ave.shape
        assert Var.shape == Ave.shape

    # cov
    Cov = _set_def_var(cov, shape=(Ave.shape[0],) + Ave.shape[1:]*2,
                       dtype=dtype, axis=axis)
    if Cov is not None:
        assert Cov.shape[0] == Ave.shape[0]
        assert Cov.shape[1:] == Ave.shape[1:]*2
        
    # weights
    if weights is None:
        weights = np.ones(Ave.shape[0], dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)

    return weights, Ave, Var, Cov


def _squeeze_axis(x, axis=0):
    if x.shape[axis]==1:
        # last part is to acces zero shape arrays
        x = np.squeeze(x,axis=axis)[()]
    return x



def spliced_stats(ave, var=None, weights=None, cov=None,
                  axis=0, dtype=None, ret_all=True,
                  squeeze_axis=True):
    """
    compute spliced stats 

    Parameters
    ----------
    w : ndarray (Default None)
        array of weights of shape (nsample,)
        if None, assume weights are equal.
    ave : 1 or 2 d array (Default None)
        array of averages of shape (nsample,) or (nvar,nsample) where nvar
        is the number of variables to consider
    var : 1 or 2d array (Default None)
        array of variances

    cov : 3d array of covariances (Default None)
    (nvar,nvar,nsample)

    ret_all : bool (default False)
        if True, return all (W,Ave,Var,Cov) regardless of input

    dtype : numpy dtype (default None)
        dtype of output.  if None, use ave.dtype

    squeeze_axis : bool (Default True)
    if True, and 

    Returns
    -------
    W : spliced weight
    Ave : spliced ave (if ave is not None)
    Var : spliced Var (if var is not None)
    Cov : spliced Cov (if cov is not None)
    """

    #parameters
    if dtype is None:
        dtype = ave.dtype

    weights, Ave, Var, Cov = _setup_Weight_Ave_Var_Cov(ave, weights,
                                                       var,
                                                       cov,
                                                       axis=axis,
                                                       dtype=dtype)

    W, S, V, C = _splice_stats(Ave, weights, dtype, Var=Var, Cov=Cov)

    # output
    def _squeeze(x):
        if x is None:
            pass
        elif squeeze_axis:
            x = _squeeze_axis(x, axis)
        return x

    S = _squeeze(S)
    ret = [W, S]

    if V is not None or ret_all:
        V = _squeeze(V)
        ret.append(V)

    if cov is not None or ret_all:
        C =  _squeeze(C)
        ret.append(C)
    return tuple(ret)


def spliced_stats_block(ave, weights=None, var=None, cov=None,
                          axis=0, dtype=None,
                          squeeze_axis=True,
                          block_size=None, last_flag='include',
                          squeeze_block=True):

    if dtype is None:
        dtype = ave.dtype
    if block_size is None:
        block_size = ave.shape[axis]

    weights, Ave, Var, Cov = _setup_Weight_Ave_Var_Cov(ave, weights,
                                                       var,
                                                       cov,
                                                       axis=axis,
                                                       dtype=dtype)

    W_L = []
    S_L = []
    V_L = []
    C_L = []

    nsamp = Ave.shape[0]
    i0 = 0
    go = True
    while go and i0 < nsamp:
        i1 = i0 + block_size
        if i1 > nsamp:
            if last_flag == 'include':
                i1 == nsamp
                # do last, and this is the last iter
                go = False
            else:
                # don't do last and get out
                break

        if Var is None:
            _Var = None
        else:
            _Var = Var[i0:i1]

        if Cov is None:
            _Cov = None
        else:
            _Cov = Cov[i0:i1]


        W, S, V, C = _splice_stats(Ave[i0:i1], weights[i0:i1],
                                   dtype, Var=_Var, Cov=_Cov)

        i0 = i1
        W_L.append(W)
        S_L.append(S)
        V_L.append(V)
        C_L.append(C)

    # output
    W_L = np.array(W_L)
    if squeeze_block:
        W_L = _squeeze_axis(W_L,axis=0)

    def _squeeze(x):
        x = np.array(x)
        if squeeze_axis:
            x = _squeeze_axis(x, axis=-1)
        if squeeze_block:
            x = _squeeze_axis(x, axis=0)
        return x

    S_L = _squeeze(S_L)

    ret = [W_L, S_L]

    if Var is not None:
        V_L = _squeeze(V_L)
        ret.append(V_L)

    if Cov is not None:
        C_L = _squeeze(C_L)
        ret.append(C_L)

    return tuple(ret)



def spliced_stats_block_2(ave, var=None, weights=None, cov=None, axis=0, dtype=None, block_size=None,
                  include_last=True, squeeze_axis=True, squeeze_block=True):
    """
    compute spliced stats 

    Parameters
    ----------
    w : ndarray (Default None)
        array of weights of shape (nsample,)
        if None, assume weights are equal.
    ave : 1 or 2 d array (Default None)
        array of averages of shape (nsample,) or (nvar,nsample) where nvar
        is the number of variables to consider
    var : 1 or 2d array (Default None)
        array of variances

    cov : 3d array of covariances (Default None)
    (nvar,nvar,nsample)

    ret_all : bool (default False)
        if True, return all (W,Ave,Var,Cov) regardless of input

    dtype : numpy dtype (default None)
        dtype of output.  if None, use ave.dtype

    block_size : int or None
    size of blocks to consider.  If none, block_size=ave.shape[axis],
    i.e., number of samples

    Returns
    -------
    W : spliced weight
    Ave : spliced ave (if ave is not None)
    Var : spliced Var (if var is not None)
    Cov : spliced Cov (if cov is not None)
    """

    #parameters
    if dtype is None:
        dtype = ave.dtype

    if block_size is None:
        block_size = ave.shape[axis]


    # roll axis to first position
    def _reshape(x,axis):
        if x.ndim == 1:
            return x.reshape(-1,1)
        else:
            return np.rollaxis(x,axis,0)

    # average reshape
    Ave = _reshape(ave,axis)
    S = np.zeros(Ave.shape[1:], dtype=dtype)
    S_L = []

    # variance
    if var is not None:
        if type(var) is bool and var:
            Var = np.zeros_like(Ave, dtype=dtype)
        else:
            Var = _reshape(var,axis)
        assert Var.shape == Ave.shape
        V = np.zeros(Ave.shape[1:],dtype=dtype)
        V_L = []

    # cov
    if cov is not None:
        if type(cov) is bool and cov:
            Cov = np.zeros((Ave.shape[0],) + Ave.shape[1:]*2, dtype=dtype)
        else:
            Cov = _reshape(cov,axis)
        assert Cov.shape[0] == Ave.shape[0]
        assert Cov.shape[1:] == Ave.shape[1:]*2
        C = np.zeros(Cov.shape[1:], dtype=dtype)
        C_L = []


    #weights
    if weights is None:
        weights = np.ones(Ave.shape[0], dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)
    W = 0.0
    W_L = []


    for i in range(Ave.shape[0]):
        w = weights[i]

        W_last = W
        W += w

        f = w / W
        delta = Ave[i,...] - S
        deltaM = np.multiply.outer(delta, delta)

        delta *= f
        deltaM *= f * W_last

        S += delta

        if var is not None:
            V += Var[i,...] * w + _diag_nd(deltaM)

        if cov is not None:
            C += Cov[i,...] * w + deltaM

        # take block average?
        if (i+1)%block_size == 0 or \
           (i==Ave.shape[0]-1 and include_last):
            W_L.append(W)
            S_L.append(S.copy())
            S[:] = 0.0

            if var is not None:
                V_L.append(V / W)
                V[:] = 0.0

            if cov is not None:
                C_L.append(C / W)
                C[:] = 0.0
            W = 0.0


    # output
    W_L = np.array(W_L)
    if squeeze_block and W_L.shape[0]==1:
        W_L = np.squeeze(W_L,axis=0)

    def _squeeze(x):
        x = np.asarray(x)
        if squeeze_axis and x.shape[-1]==1:
            x = np.squeeze(x,axis=-1)
        if squeeze_block and x.shape[0]==1:
            x = np.squeeze(x,axis=0)
        return x
    
    S_L = _squeeze(S_L)

    ret = [W_L, S_L]

    if var is not None:
        V_L = _squeeze(V_L)
        ret.append(V_L)

    if cov is not None:
        C_L = _squeeze(C_L)
        ret.append(C_L)

    return tuple(ret)




def cumave(ave, w, axis=0, var=None):
    """
    take a sequence of averages and create a running ave and var

    Parameters
    ----------
    ave : array-like
    
    var : array-like (or None)
    
    w : 1d-array or scalar
    weights

    axis : int
    axis to cum overlap

    Returns
    -------
    Rave : running ave

    Rvar : runnin variance (if var is not None)

    """

    if var is None:
        _var = np.zeros_like(ave)
    else:
        _var = var

    if isinstance(w, collections.Iterable):
        _w = w
    else:
        _w = np.ones(ave.shape[axis], dtype=np.float) * w

    Rave = np.zeros_like(ave)
    Rvar = np.zeros_like(_var)
    Rw = np.zeros_like(_w)

    W = 0
    shape = list(ave.shape)
    shape.pop(axis)
    shape = tuple(shape)
    S0 = np.zeros(shape, dtype=ave.dtype)
    S1 = np.zeros(shape, dtype=ave.dtype)

    slc = [slice(None)] * ave.ndim

    for i in range(ave.shape[axis]):
        W_last = W
        W += _w[i]

        f0 = _w[i] / W
        f1 = W_last * f0

        slc[axis] = i
        delta0 = ave[slc] - S0
        delta1 = delta0**2

        S0 += delta0 * f0
        S1 += _var[slc] * _w[i] + delta1 * f1

        Rave[slc] = S0
        Rvar[slc] = S1 / W
        Rw[i] = W

    if var is None:
        return Rave, Rw
    else:
        return Rave, Rvar, Rw


class _RunningStats(object):
    """
    object to keep running stats of vector object
    """

    def __init__(self, shape, dtype=np.float):
        """
        running stats with covariance object

        Parameters
        ----------
        shape : tuple or int

        dtype : data type
        """
        self.dtype = dtype
        self.shape = shape
        self.Zero()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self,val):
        if isinstance(val, int):
            val = (val,)
        if type(val) is not tuple:
            raise ValueError('shape must be tuple or int')
        self._shape = val

    @property
    def var_shape(self):
        """property of variance/covariance shape"""
        raise ValueError('implement in subclass')

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self,val):
        self._dtype = val

    def Zero(self):
        #spliced statistics
        #S -> mean
        #V -> var/cov
        self._S = np.zeros(self.shape, dtype=self.dtype)
        self._V = np.zeros(self.var_shape, dtype=self.dtype)
        self._W = np.zeros(self.shape, dtype=self.dtype)

    def zero_like(self):
        """easy way to create zero object like self"""
        return self.__class__(shape=self.shape, dtype=self.dtype)

    def copy(self):
        new = self.__class__(shape=self.shape, dtype=self.dtype)
        new._S[...] = self._S
        new._V[...] = self._V
        new._W[...] = self._W
        return new

    # --------------------------------------------------
    # pushing
    # --------------------------------------------------
    def push_stat(self, w, a, v):
        """
        push single set of stats 

        Parameters
        ----------
        w : int or 
            weight of sample

        a : array shape == self.shape
            array of averages

        v:  array shape == self.var_shape
            array of variance/covariance
        """
        raise ValueError('implement in child')


    def push_stats(self, w, a, v, axis=0):
        """
        push multiple stats to self along axis
        """
        nsamp = a.shape[axis]
        assert w.shape[axis] == nsamp
        assert v.shape[axis] == nsamp

        for i in range(nsamp):
            ww = np.take(w,i,axis=axis)
            aa = np.take(a,i,axis=axis)
            vv = np.take(v,i,axis=axis)
            self.push_stat(w=ww, a=aa, v=vv)

    def push_vals(self, x, axis=-1, ddof=0, w=None):
        """
        push an array of values to stats

        Parameters
        ----------
        x : array
            values to accumulate
        axis : int (Default -1)
            axis to average along
        ddof : int (Default 0)
            degree of freedom to use in var/cov calculation
        w : array (Default None)
        if None, use x.shape 
        """
        raise ValueError('implement in child')

    def push_val(self, x, w=1.0, v=0.0):
        """push a single value to stats"""
        self.push_stat(w=w, a=x, v=v)

    # --------------------------------------------------
    # +/- +=
    # --------------------------------------------------
    def __add__(self, b):
        assert type(self) == type(b)
        new = self.copy()
        new.push_stat(w=b._W, a=b.mean(), v=b.class_var())
        return new

        # if type(self) == type(b):
        #     new = self.copy()
        #     new.push_stat(w=b._W, a=b.mean(), v=b.class_var())
        #     return new
        # else:
        #     new = self.copy()
        #     new.push_val(x=b)
        #     return new

    # def __radd__(self,b):
    #     return self.__add__(b)

    def __sub__(self, b):
        assert (type(self) == type(b))
        assert np.all(self._W > b._W)
        new = self.copy()
        new.push_stat(w=-b._W, a=b.mean(), v=b.class_var())
        return new

    def __iadd__(self, b):
        assert type(self) == type(b)
        self.push_stat(w=b._W, a=b.mean(), v=b.class_var())
        return self

        # if type(self) == type(b):
        #     self.push_stat(w=b._W, a=b.mean(), v=b.class_var())
        #     return self
        # else:
        #     self.push_val(x=b)
        #     return self

    # --------------------------------------------------
    # stats
    # --------------------------------------------------
    #TODO : always acces [()] incase 0d arrays?
    def class_var(self):
        """returns var or cov"""
        raise ValueError('implement in child return e.g. self.var() or self.cov()')
    
    def weight(self):
        """total weight"""
        return self._W

    def mean(self):
        """mean over samples"""
        return self._S

    def var(self, ddof=0.0):
        """
        variance over samples, with specified degree of freedom
        """
        raise ValueError('implement in child')

    def std(self, ddof=0.0):
        """standard deviation, with degree of freedom"""
        return np.sqrt(self.var(ddof))


    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_stat(cls, w, a, v):
        """
        object from single weight, average, variance/covariance
        """
        new = cls(shape=a.shape, dtype=a.dtype)
        new.push_stat(w, a, v)
        return new

    @classmethod
    def from_stats(cls, w, a, v, axis=0):
        """
        object from several weights, averages, variances/covarainces along axis
        """
        #get shape
        shape = list(a.shape)
        shape.pop(axis)
        shape = tuple(shape)

        new = cls(shape=shape, dtype=a.dtype)
        new.push_stats(w=w, a=a, v=v, axis=axis)
        return new

    @classmethod
    def from_vals(cls, x, axis=0, ddof=0, w=None, dtype=None):

        #get shape
        shape = list(x.shape)
        shape.pop(axis)
        shape = tuple(shape)

        if dtype is None:
            dtype = x.dtype

        new = cls(shape=shape,dtype=dtype)
        new.push_vals(x, axis=axis, ddof=ddof, w=w)
        return new




class RunningStatsVecCov(_RunningStats):
    """
    vector covariance object
    """

    @property
    def var_shape(self):
        return self.shape*2


    def push_stat(self, w, a, v):
        """
        accumulate statistics

        Parameters
        ----------
        w : scalar or array broadcastable to a.shape
            weight of sample.
            if `None`, use w=1.0

        a : array shape == self.shape
            averages
        v : array shape == self.var_shape
        """

        W0_last = self._W.copy()
        self._W += w

        f = w / self._W
        delta = a - self._S
        deltaM = np.multiply.outer(delta, delta)

        delta *= f
        deltaM *= f * W0_last

        #spliced stats
        self._S += delta
        self._V += v * w + deltaM

    def push_vals(self, x, axis=-1, ddof=0, w=None):
        """
        push an array of values to stats

        Parameters
        ----------
        x : array
            values to accumulate
        axis : int (Default -1)
            axis to average along
        ddof : int (Default 0)
            degree of freedom to use in var/cov calculation

        w : array (Default None)
            if None, use x.shape[axis] 
        """
        if w is None:
            w = x.shape[axis]
        self.push_stat(w = w,
                        a=np.mean(x, axis=axis),
                        v=cov_nd(x, axis=axis, ddof=ddof))

    def class_var(self):
        """for this class "var" is actually covariance"""
        v = self.cov()
        v[self.weight()==0] = 0.0
        return v

    def var(self, ddof=0.0):
        return _diag_nd(self._V) / (self._W - ddof)

    def cov(self, ddof=0.0):
        return self._V / (self._W - ddof)



class RunningStatsVec(_RunningStats):
    """
    Vector ave/variance object
    """
    @property
    def var_shape(self):
        return self.shape

    def push_stat(self, w, a, v):
        """
        accumulate statistics

        Parameters
        ----------
        w : int or array broadcastable to a.shape
            weight of sample

        a : array shape == self.shape
            averages
        v : array shape == self.var_shape
        """
        W0_last = self._W.copy()
        self._W += w

        f = w / self._W
        delta = a - self._S
        deltaM = delta * delta

        delta *= f
        deltaM *= f * W0_last

        #spliced stats
        self._S += delta
        self._V += v * w + deltaM

    def push_vals(self, x, axis=-1, ddof=0, w=None):
        """
        push an array of values to stats

        Parameters
        ----------
        x : array
            values to accumulate
        axis : int (Default -1)
            axis to average along
        ddof : int (Default 0)
            degree of freedom to use in var/cov calculation

        w : array (Default None)
            if None, use x.shape[axis]
        """
        if w is None:
            w = x.shape[axis]
        self.push_stat(w = w,
                        a=np.mean(x, axis=axis),
                        v=np.var(x, axis=axis, ddof=ddof))


    def class_var(self):
        v=self.var()
        v[self.weight()==0] = 0.0
        return v

    def var(self, ddof=0.0):
        return np.asarray(self._V / (self._W - ddof))

class RunningStats(RunningStatsVec):
    def __init__(self, shape=(), dtype=np.float):
        super(self.__class__,self).__init__(shape=(),dtype=dtype)



class RunningStatsList(object):
    """
    collection of running stats for grouping, etc
    """
    def __init__(self, data=None, *args, **kwargs):
        if data is None:
            data = []
        self._list = data

    @property
    def shape(self):
        return (len(self),)


    def copy(self):
        data = [x.copy() for x in self]
        return self.__class__(data=data)

    # --------------------------------------------------
    # list access
    # --------------------------------------------------
    def __len__(self):
        return len(self._list)

    def __getitem__(self,idx):
        if isinstance(idx, slice):
            L = self._list[idx]

        elif isinstance(idx,(list, np.ndarray)):
            idx = np.asarray(i)
            if np.issubdtype(idx.dtype,np.integer):
                L = [self._list[j] for j in idx]
            elif np.issubdtype(idx.dtype,np.bool):
                assert idx.shape == self.shape
                L = [xx for xx,mm in zip(self._list,idx) if mm]
        else:
            return self._list[idx]

        return self.__class__(data=L)

    def __setitem__(self,idx,val):
        self._list[idx] = val


    # --------------------------------------------------
    # stats
    # --------------------------------------------------
    def mean(self):
        return np.array([x.mean() for x in self._list])

    def var(self,ddof=0.0):
        return np.array([x.var(ddof=ddof) for x in self._list])

    def cov(self,ddof=0.0):
        return np.array([x.cov(ddof=ddof) for x in self._list])

    def weight(self):
        return np.array([x.weight() for x in self._list])

    def val_SEM(self, x, weighted, unbiased, norm):
        """
        find the standard error in the mean (SEM) of a value

        Parameters
        ----------
        x : array
            array (self.mean(), etc) to consider

        weighted : bool
            if True, use `weighted_var`
            if False, use `np.var`

        unbiased : bool
            if True, use unbiased stats (e.g., ddof=1 for np.var)
            if False, use biased stats (e.g., ddof=0 for np.var)

        norm : bool
            if True, scale var by x.shape[0], i.e., number of samples

        Returns
        -------
        sem : standard error in mean 
        """
        if weighted:
            v = weighted_var(x, w=self.weight(), axis=0, unbiased=unbiased)[-1]
        else:
            if unbiased:
                ddof = 1
            else:
                ddof = 0

            v = np.var(x,ddof=ddof,axis=0)
        if norm:
            v = v / x.shape[0]

        return np.sqrt(v)

    def mean_SEM(self, weighted=True, unbiased=True, norm=True):
        """self.val_SEM with x=self.mean()"""
        return self.val_SEM(self.mean(), weighted, unbiased, norm)

    def var_SEM(self, weighted=True, unbiased=True, norm=True, ddof=0.0):
        """self.val_SEM with x=self.var()"""
        return self.val_SEM(self.var(ddof=ddof), weighted, unbiased, norm)

    def cov_SEM(self, weighted=True, unbiased=True, norm=True, ddof=0.0):
        """self.val_SEM with x=self.cov()"""
        return self.val_SEM(self.cov(ddof=ddof), weighted, unbiased, norm)


    # --------------------------------------------------
    # combine for block averages
    # --------------------------------------------------
    def combine(self, block_size=2, min_len=None, inplace=False):
        """
        perform block averaging of self

        [sum(self[i:i+block_size) for i in range(0,len(self),block_size)]

        Parameters
        ----------
        block_size : int (Default 2)
            size of block.  If `None`, block_size = len(self), i.e., one block

        min_len : int (Default None)
            if last block is of size >= min_len, then include it in output.
            Otherwise, discard this block.
            If `None`, min_len=block_size

        inplace : bool (Default False)
            if True, perform inplace modification

        Returns
        -------
        output : RunningStatsList
            RunningStatsList of block averaged data
        """
        if inplace:
            data = self._list
        else:
            data = self.copy()._list

        if block_size is None:
            block_size = len(self)

        if block_size > len(self):
            block_size = len(self)

        if min_len is None:
            min_len = block_size

        L = []
        s = 0
        while True:
            e = s + block_size
            if e >= len(self):
                e = len(self)
                if e-s < min_len:
                    break

            x = data[s]
            for y in data[s+1:e]:
                x += y
            L.append(x)
            if e >= len(self):
                break
            s += block_size

        if inplace:
            self._list = L
        else:
            return self.__class__(data=L)


    
    def _combine_0(self,block_size=2,min_len=None):
        if min_len is None:
            min_len = block_size

        L = []
        s = 0
        while True:
            e = s + block_size
            if e >= len(self):
                e = len(self)
                if e-s < min_len:
                    break

            L.append(sum(self[s:e],self[0].zero_like()))
            if e >= len(self):
                break
            s += block_size

        return self.__class__(data=L)


    # --------------------------------------------------
    # constructurs
    # --------------------------------------------------
    @classmethod
    def from_stats_gen(cls, rs, w, a, v,
                       axis=0, **kwargs):
        """
        create a RunningStatsList from data

        Parameters
        ----------
        rs : RunningStats class
            class for each element of list

        w : array-like
            weights for each sample

        a : array-like
            mean for each sample

        v : array-like
            var/cov for each sample

        axis : int
            axis corresponding to sample

        **kwargs : extra args to cls

        Returns
        -------
        out : RunningStatsList object
        """
        assert w.shape[axis] == a.shape[axis]
        assert v.shape[axis] == a.shape[axis]

        data = []
        for i in range(a.shape[axis]):
            ww = np.take(w,i,axis=axis)
            aa = np.take(a,i,axis=axis)
            vv = np.take(v,i,axis=axis)
            data.append(rs.from_stat(w=ww,a=aa,v=vv))
        return cls(data=data,**kwargs)

    @classmethod
    def from_stats(cls, w, a, var=None, cov=None, axis=0, **kwargs):
        """
        create Running StatsList from data

        must specify either var or cov
        """

        if cov is not None:
            rs = RunningStatsVecCov
            v = cov
        elif var is not None:
            rs = RunningStatsVec
            v = var
        else:
            raise ValueError('must specify var or cov')

        return cls.from_stats_gen(rs, w, a, v, axis=axis, **kwargs)

