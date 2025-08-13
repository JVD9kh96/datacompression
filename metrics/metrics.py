"""
metrics.py
----------
Utilities to reproduce the information-theoretic measurements from:

    "Scalable Image Coding for Humans and Machines"
    H. Choi, I. V. Bajic, (paper uploaded by user)

Implements:
 - H(Y) ≈ E[-log2 p(ŷ)] rate estimates from model likelihoods (Fig.7, Eq.(7))
 - percentage H(Y1)/H(Y) (Fig.7, Eq.(8))
 - Mutual information estimate I(Ỹ ; F1) using KDE + clustering (Fig.9, Eq.(9))

Notes:
 - For KDE we use sklearn.neighbors.KernelDensity (Gaussian kernel).
 - KDE in high-dim may be unreliable. The user should subsample or reduce
   dimensionality if needed (PCA etc.). The paper uses KDE as in [44].
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

LN2 = math.log(2.0)


# --------------------------
# Rate / Entropy helpers
# --------------------------
# metrics.py
import math
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf

# stable epsilon for probabilities
_EPS = 1e-12
_SQRT2 = math.sqrt(2.0)


def clip_probs(p: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Clip probabilities to [eps, 1]. Works on numpy arrays."""
    p = np.asarray(p)
    p = np.nan_to_num(p, nan=eps, posinf=1.0, neginf=eps)
    p = np.clip(p, eps, 1.0)
    return p


def discrete_gaussian_likelihood_tf(x: tf.Tensor,
                                   mu: tf.Tensor,
                                   sigma: tf.Tensor,
                                   eps: float = _EPS) -> tf.Tensor:
    """
    Compute discrete Gaussian probability P(X = round(x)) ≈ Phi(x+0.5) - Phi(x-0.5).
    All arguments are tf.Tensors of broadcastable shapes (e.g. [B,H,W,C]).
    Returns probabilities (not log).
    """
    # ensure sigma positive
    sigma = tf.maximum(sigma, 1e-6)
    upper = 0.5 * (1.0 + tf.math.erf((x + 0.5 - mu) / (sigma * _SQRT2)))
    lower = 0.5 * (1.0 + tf.math.erf((x - 0.5 - mu) / (sigma * _SQRT2)))
    p = upper - lower
    p = tf.clip_by_value(p, eps, 1.0)
    return p


def discrete_gaussian_mixture_likelihood_tf(x: tf.Tensor,
                                           mus: tf.Tensor,
                                           sigmas: tf.Tensor,
                                           logits: tf.Tensor,
                                           eps: float = _EPS) -> tf.Tensor:
    """
    Numerically-stable mixture likelihood:
      p(x) = sum_k softmax(logits)_k * (Phi(x+0.5; mu_k, sigma_k) - Phi(x-0.5; mu_k, sigma_k))
    Shapes:
      x: [..., C]
      mus, sigmas, logits: [..., C, K]
    Returns: p: [..., C] (probabilities)
    """
    # compute per-component cdf diffs => shape [..., C, K]
    x_exp = tf.expand_dims(x, axis=-1)  # [..., C, 1]
    upper = 0.5 * (1.0 + tf.math.erf((x_exp + 0.5 - mus) / (sigmas * _SQRT2)))
    lower = 0.5 * (1.0 + tf.math.erf((x_exp - 0.5 - mus) / (sigmas * _SQRT2)))
    comp_p = upper - lower  # [..., C, K]
    # mixture weights
    w = tf.nn.softmax(logits, axis=-1)  # [..., C, K]
    # weighted sum
    p = tf.reduce_sum(w * comp_p, axis=-1)
    p = tf.clip_by_value(p, eps, 1.0)
    return p


def bits_from_likelihoods_np(lik: np.ndarray, eps: float = _EPS) -> float:
    """
    Convert numpy array of per-element probabilities to total bits.
    Returns scalar bits (sum over all elements).
    """
    lik = clip_probs(lik, eps=eps)
    # sum of -log2(p) over all elements
    nats = -np.log(lik).sum()
    bits = float(nats / math.log(2.0))
    return bits


def rate_from_likelihoods(y_likelihoods: np.ndarray, divide_by_pixels: Optional[int] = None) -> Dict[str, float]:
    """
    y_likelihoods: numpy array [B, Hy, Wx, C] or any shape of probabilities
    divide_by_pixels: integer number of input pixels (B * H_in * W_in). If None, function returns bits only.
    Returns {'bits': total_bits, 'bpp': bits_per_pixel (or None)}
    """
    bits = bits_from_likelihoods_np(y_likelihoods)
    if divide_by_pixels is None or divide_by_pixels == 0:
        bpp = None
    else:
        bpp = bits / float(divide_by_pixels)
    return {'bits': bits, 'bpp': bpp}


def rate_fraction_base(y1_like: np.ndarray, y_like: np.ndarray) -> float:
    """
    Compute percentage 100 * H(Y1) / H(Y)
    y1_like, y_like are numpy probability arrays (must correspond to same batch/pixels).
    """
    bits_y1 = bits_from_likelihoods_np(y1_like)
    bits_y = bits_from_likelihoods_np(y_like)
    if bits_y <= 0:
        return 0.0
    return 100.0 * (bits_y1 / bits_y)


# -------------------------------
# A diagnostic utility to print suspicious cases
# -------------------------------
def print_likelihood_stats_np(lik: np.ndarray, name: str = 'lik'):
    lik = np.asarray(lik)
    print(f"{name}.shape = {lik.shape}")
    print(f"{name}.min, max, mean, zeros = {lik.min()}, {lik.max()}, {lik.mean()}, {(lik <= 0).sum()}")



# --------------------------
# Fiber extraction helpers
# --------------------------
def extract_fibers_Y(Y: np.ndarray) -> np.ndarray:
    """
    Convert sub-latent Y (B,H,W,Cy) into fibers of shape (N_fibers, Cy),
    where each fiber is the vector at one spatial location.

    Args:
      Y: numpy array shape (B, H, W, C_y)

    Returns:
      2D array (N, C_y) where N = B * H * W
    """
    Y = np.asarray(Y)
    if Y.ndim != 4:
        raise ValueError("Y must be 4D [B,H,W,Cy]. Got shape: %s" % (Y.shape,))
    B, H, W, Cy = Y.shape
    fibers = Y.reshape((-1, Cy))
    return fibers


def extract_corresponding_F_fibers(F: np.ndarray, Y_shape: Tuple[int, int], patch_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Extract fibers from feature map F corresponding to each spatial location of Y.

    Suppose Y has spatial (Ny, Nx). F has spatial (Nfy, Nfx).
    We assume Nfy = Ny * ph and Nfx = Nx * pw (integer mapping). For each Y location (i,j)
    we extract the patch F[ph*i:ph*i+ph, pw*j:pw*j+pw, :] and flatten it.

    Args:
      F: numpy array shape (B, Hf, Wf, Cf)
      Y_shape: tuple (Ny, Nx) spatial dims of Y
      patch_size: (ph, pw) patch size to extract in F per Y location (default (2,2))

    Returns:
      2D array (N_fibers, Cf * ph * pw)
    """
    F = np.asarray(F)
    if F.ndim != 4:
        raise ValueError("F must be 4D [B,Hf,Wf,Cf]. Got shape: %s" % (F.shape,))
    ph, pw = patch_size
    B, Hf, Wf, Cf = F.shape
    Ny, Nx = Y_shape

    # basic checks
    if (Hf % Ny) != 0 or (Wf % Nx) != 0:
        # try to infer mapping via integer factor
        # but require exact mapping
        raise ValueError("F spatial dims must be integer multiples of Y dims. F shape: %s, Y shape: %s" % ((Hf, Wf), (Ny, Nx)))
    # expected patch sizes derived from factor
    factor_h = Hf // Ny
    factor_w = Wf // Nx
    # If user passed patch_size not equal to factor, we still support extracting patch_size region from the factor grid
    # but default behavior is patch_size == (factor_h, factor_w)
    if patch_size != (factor_h, factor_w):
        # warn but proceed (extract smaller/larger patches if possible)
        pass

    # We'll extract patch of size (ph, pw) starting at top-left of each factor-block.
    fibers = []
    for b in range(B):
        for iy in range(Ny):
            start_h = iy * factor_h
            for ix in range(Nx):
                start_w = ix * factor_w
                # clamp to image
                end_h = start_h + ph
                end_w = start_w + pw
                if end_h > Hf or end_w > Wf:
                    # skip if patch doesn't fit (shouldn't happen)
                    continue
                patch = F[b, start_h:end_h, start_w:end_w, :]
                fibers.append(patch.reshape(-1))
    fibers = np.stack(fibers, axis=0)  # (B*Ny*Nx, ph*pw*Cf)
    return fibers


# --------------------------
# KDE and MI estimation
# --------------------------
def _silverman_bandwidth(data: np.ndarray) -> float:
    """
    Silverman rule-of-thumb generalized for multivariate data (approx).
    h = std * n^{-1/(d+4)} times constant factor
    For multivariate we take average std across dims.
    This is heuristic; user may override with bandwidth param.
    """
    n, d = data.shape
    std = np.std(data, axis=0)
    avg_std = float(np.mean(std))
    if avg_std <= 0:
        avg_std = 1.0
    # Silverman's constant for gaussian kernel (univariate): ( (4/(d+2))^{1/(d+4)} )
    const = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    h = const * avg_std * (n ** (-1.0 / (d + 4.0)))
    # avoid too tiny bandwidth
    h = max(h, 1e-6)
    return h


def estimate_entropy_kde(samples: np.ndarray, bandwidth: Optional[float] = None, kernel: str = "gaussian") -> float:
    """
    Estimate differential entropy H(X) (in bits) from samples using KDE plug-in estimator:
      H ≈ - mean(log p_hat(x_i)) / ln(2)

    Args:
      samples: (N, d) numpy array
      bandwidth: if None, heuristic (Silverman) is used
      kernel: kernel type for sklearn.KernelDensity

    Returns:
      entropy_in_bits (float)
    """
    samples = np.asarray(samples, dtype=np.float64)
    n, d = samples.shape
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0

    if bandwidth is None:
        bandwidth = _silverman_bandwidth(samples)

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(samples)
    logp = kde.score_samples(samples)  # natural log density p_hat(x)
    # -mean(log p_hat)
    H_nats = -np.mean(logp)
    H_bits = H_nats / LN2
    return float(H_bits)


def estimate_mi_kde(
    Y_sub: np.ndarray,
    F_feat: np.ndarray,
    Y_spatial_shape: Tuple[int, int],
    patch_size: Tuple[int, int] = (2, 2),
    n_clusters: int = 16,
    sample_limit: Optional[int] = 50000,
    bandwidth: Optional[float] = None,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Estimate mutual information I(Y_sub_fiber ; F_feature_cluster_index) using KDE and clustering.
    This implements the paper's procedure (Section III-E and Eq.9) approximately.

    Steps:
      1) Extract Y-fibers: from Y_sub (B,Ny,Nx,Lj) -> (N, Lj)
      2) Extract corresponding F-fibers: from F_feat (B, Hf, Wf, Cf) mapping each Y location to a patch
         of size patch_size in F and flatten -> (N, Df)
      3) Cluster F-fibers into K clusters (KMeans)
      4) Compute H(Y) via KDE
      5) For each cluster k, compute H(Y | F1=k) via KDE on Y-fibers assigned to cluster k
      6) I = H(Y) - sum_k p_k * H(Y | F1=k)

    Args:
      Y_sub: numpy array shape (B, Ny, Nx, Lj)
      F_feat: numpy array shape (B, Hf, Wf, Cf)
      Y_spatial_shape: (Ny, Nx)
      patch_size: patch size inside F corresponding to one Y location (default (2,2))
      n_clusters: K (default 16)
      sample_limit: max number of fibers to use in total (for speed). None => use all
      bandwidth: KDE bandwidth (float) or None to use heuristic. If too small/large, results change.
      random_state: random state for KMeans and any subsampling
      verbose: show progress bars

    Returns:
      dict with keys:
        - 'I_bits_per_fiber' : estimated mutual information (bits per fiber)
        - 'H_Y' : entropy estimate of Y (bits per fiber)
        - 'H_cond' : weighted conditional entropy sum (bits per fiber)
        - 'cluster_sizes' : list of sizes per cluster
        - 'n_used' : number of fibers used
        - 'bandwidth' : bandwidth used
    """
    # Validate shapes
    Y_sub = np.asarray(Y_sub)
    F_feat = np.asarray(F_feat)
    if Y_sub.ndim != 4:
        raise ValueError("Y_sub must be shape [B, Ny, Nx, Lj]")
    if F_feat.ndim != 4:
        raise ValueError("F_feat must be shape [B, Hf, Wf, Cf]")

    # Extract fibers
    Yfibers = extract_fibers_Y(Y_sub)   # (N, Lj)
    Ffibers = extract_corresponding_F_fibers(F_feat, Y_spatial_shape, patch_size=patch_size)  # (N, Df)

    N = Yfibers.shape[0]
    if sample_limit is not None and N > sample_limit:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(N, size=sample_limit, replace=False)
        Yfibers = Yfibers[idx]
        Ffibers = Ffibers[idx]
        N = sample_limit
        if verbose:
            print(f"[MI] Subsampled to {N} fibers (limit={sample_limit})")

    # 1) cluster Ffibers into K clusters
    if verbose:
        print("[MI] Running KMeans clustering on F fibers (K=%d)..." % (n_clusters,))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(Ffibers)  # shape (N,)

    # 2) Estimate H(Y) with KDE
    if verbose:
        print("[MI] Estimating H(Y) with KDE (dim=%d, n=%d)..." % (Yfibers.shape[1], N))
    if bandwidth is None:
        # heuristic bandwidth - given high dimensionality this is only heuristic
        bandwidth_used = _silverman_bandwidth(Yfibers)
    else:
        bandwidth_used = float(bandwidth)

    H_Y = estimate_entropy_kde(Yfibers, bandwidth=bandwidth_used)
    if verbose:
        print(f"[MI] H(Y) ≈ {H_Y:.4f} bits per fiber (bandwidth={bandwidth_used:.6g})")

    # 3) Compute conditional entropies H(Y | F1 = k)
    H_cond_sum = 0.0
    cluster_sizes = []
    for k in range(n_clusters):
        mask = (labels == k)
        nk = int(mask.sum())
        cluster_sizes.append(nk)
        if nk == 0:
            # no samples for cluster -> skip
            if verbose:
                print(f"[MI] cluster {k} empty (nk=0) -> contributes 0 to conditional sum")
            continue
        Yk = Yfibers[mask]
        # if too few samples, KDE will be unstable; we fallback to global H(Y) (conservative)
        if nk < max(10, 2 * Yk.shape[1]):
            # fallback: use global H(Y) (this makes I estimate conservative / lower)
            Hk = H_Y
            if verbose:
                print(f"[MI] cluster {k} small (nk={nk}) - fallback Hk = H_Y ({H_Y:.4f})")
        else:
            # estimate cluster-specific bandwidth (optionally scale)
            if bandwidth is None:
                bw_k = _silverman_bandwidth(Yk)
            else:
                bw_k = bandwidth_used
            Hk = estimate_entropy_kde(Yk, bandwidth=bw_k)
            if verbose:
                print(f"[MI] cluster {k}: nk={nk}, Hk={Hk:.4f} (bw={bw_k:.6g})")
        p_k = float(nk) / float(N)
        H_cond_sum += p_k * Hk

    I_bits = H_Y - H_cond_sum

    return {
        'I_bits_per_fiber': float(I_bits),
        'H_Y': float(H_Y),
        'H_cond': float(H_cond_sum),
        'cluster_sizes': cluster_sizes,
        'n_used': int(N),
        'bandwidth': float(bandwidth_used)
    }


# --------------------------
# convenience wrappers to integrate with TF model outputs
# --------------------------
# in metrics.py (append)
def compute_rate_metrics_from_model_outputs(model_out: Dict[str, np.ndarray], input_pixels: int) -> Dict[str, Any]:
    # y
    if 'y_likelihoods' not in model_out:
        raise KeyError("model_out must contain 'y_likelihoods'")
    y_like = np.asarray(model_out['y_likelihoods'])
    if 'y1_likelihoods' in model_out:
        y1_like = np.asarray(model_out['y1_likelihoods'])
    else:
        if y_like.ndim != 4:
            raise ValueError("y_likelihoods expected as [B,Hy,Wx,C]")
        C = y_like.shape[-1]
        L1 = C // 2
        y1_like = y_like[..., :L1]

    H_full = rate_from_likelihoods(y_like, divide_by_pixels=input_pixels)
    H_base = rate_from_likelihoods(y1_like, divide_by_pixels=input_pixels)
    frac = rate_fraction_base(y1_like, y_like)

    # include z bits if present. The TF entropy bottleneck returns bits already; some implementations
    # may return likelihoods for z instead. We support both:
    z_bits = 0.0
    if 'z_bits' in model_out:
        z_bits = float(model_out['z_bits'])
    elif 'z_likelihoods' in model_out:
        # if given likelihoods for z, convert them
        z_bits = bits_from_likelihoods_np(np.asarray(model_out['z_likelihoods']))
    elif 'z' in model_out and 'z_tilde' in model_out and 'entropy_bottleneck_bits' in model_out:
        # user could pass already computed entropy bottleneck number
        z_bits = float(model_out['entropy_bottleneck_bits'])

    # Add z bits into total bits if you want H(Y)+H(Z) = total coding cost
    # Many papers add both latent and hyperprior; decide what you want to report.
    total_bits = H_full['bits'] + z_bits
    total_bpp = total_bits / float(input_pixels) if input_pixels > 0 else 0.0
    # base bits do not include z bits (base is subset of y bits)
    return {
        'H_Y_bits': H_full['bits'],
        'H_Y_bpp': H_full['bpp'],
        'H_Y1_bits': H_base['bits'],
        'H_Y1_bpp': H_base['bpp'],
        'fraction_percent': frac,
        'H_Z_bits': z_bits,
        'Total_bits_including_z': total_bits,
        'Total_bpp_including_z': total_bpp
    }
