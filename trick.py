import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import eig
import doodads as dd


def compare_columns_modulo_sign(init_u, final_u, display=False):
    signs = np.zeros(init_u.shape[1])
    for col in range(init_u.shape[1]):
        signs[col] = 1 if np.allclose(init_u[:,col], final_u[:,col]) else -1
    vmax = np.max(np.abs([init_u, final_u]))
    final_u_mod = signs * final_u
    if display:
        import matplotlib.pyplot as plt
        fig, (ax_iu, ax_fu, ax_du) = plt.subplots(ncols=3, figsize=(14, 4))
        dd.matshow(init_u, vmin=-vmax, vmax=vmax, ax=ax_iu)
        ax_iu.set_title(r'$\mathbf{U}_\mathrm{first}$')
        dd.matshow(final_u_mod, vmin=-vmax, vmax=vmax, ax=ax_fu)
        ax_fu.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$')
        diff_vmax = np.max(np.abs(final_u_mod - init_u))
        dd.matshow(final_u_mod - init_u, cmap='RdBu_r', vmax=diff_vmax, vmin=-diff_vmax, ax=ax_du)
        ax_du.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$ - $\mathbf{U}_\mathrm{first}$')
    return np.allclose(final_u_mod, init_u)


def orthonormalize(matrix_a):
    '''Orthonormalize `matrix_a` to produce `matrix_q` which is an
    orthogonal basis of the column space of `matrix_a`
    '''
    # Benchmarked against a somewhat optimized numpy implementation of MGS,
    # confirmed that calculating R along the way is a negligible
    # cost in time
    matrix_q, _ = np.linalg.qr(matrix_a)
    return matrix_q


def truncated_svd(mtx_x, n_singular_vals):
    '''Compute the truncated SVD up to the first `n_singular_vals` values
    using `scipy.sparse.linalg.svds` unless that would be the full decomposition
    (in which case use `numpy.linalg.svd`)
    '''
    if n_singular_vals > mtx_x.shape[0]:
        raise ValueError(
            f"Number of singular values {n_singular_vals} cannot exceed number of rows {mtx_x.shape[0]}")
    if n_singular_vals == mtx_x.shape[0]:
        # truncated SVD solver only works when truncating
        # use regular SVD if not
        mtx_u, diag_s, mtx_vt = np.linalg.svd(mtx_x, full_matrices=False)
        return mtx_u, diag_s, mtx_vt.T
    initial_u, initial_s, initial_vh = svds(
        csc_matrix(mtx_x), k=n_singular_vals)
    # Greatest singular values should be towards top left of the diagonal matrix
    # for consistency with (unstated) convention in Brand derivation, but
    # the sparse SVD in SciPy uses the opposite convention
    desc_sval_order = np.argsort(initial_s)[::-1]
    initial_s = initial_s[desc_sval_order]
    initial_u = initial_u[:, desc_sval_order]
    initial_vh = initial_vh[desc_sval_order]
    initial_v = initial_vh.T
    return initial_u, initial_s, initial_v


def downdate(mtx_u, diag_s, mtx_v, col_data_to_remove, col_idxs_to_remove):
    '''returns new_mtx_u, new_diag_s, new_mtx_v'''
    dim_p, dim_q = mtx_u.shape[0], mtx_v.shape[0]
    dim_r = len(diag_s)
    assert mtx_u.shape[1] == dim_r
    assert mtx_v.shape[1] == dim_r

    mtx_a = -col_data_to_remove
    dim_c = col_data_to_remove.shape[1]
    mtx_b = np.zeros((dim_q, dim_c))
    # In each column of B, place a 1 in the row corresponding to the
    # column of the full matrix X we're removing
    mtx_b[col_idxs_to_remove, np.arange(dim_c)] = 1

    # Compute P to augment U and [[U^T A], [R_A]]
    mtx_a_orth_u = mtx_a - mtx_u @ (mtx_u.T @ mtx_a)
    mtx_p, temp_r = np.linalg.qr(mtx_a_orth_u)
    mtx_ra = mtx_p.T @ mtx_a_orth_u
    dim_d = mtx_p.shape[1]

    mtx_uta = mtx_u.T @ mtx_a
    mtx_u_p = np.hstack([mtx_u, mtx_p])
    mtx_uta_ra = np.vstack([mtx_uta, mtx_ra])

    # Compute Q to augment V and [[V^T B], [R_B]]
    mtx_b_orth_v = mtx_b - mtx_v @ (mtx_v.T @ mtx_b)
    mtx_q, temp_r = np.linalg.qr(mtx_b_orth_v)
    mtx_rb = mtx_q.T @ mtx_b_orth_v
    dim_f = mtx_q.shape[1]

    mtx_vtb = mtx_v.T @ mtx_b
    mtx_v_q = np.hstack([mtx_v, mtx_q])
    mtx_vtb_rb = np.vstack([mtx_vtb, mtx_rb])

    # Full calculation
    mtx_k_second_term = mtx_uta_ra @ mtx_vtb_rb.T
    pad_rows = dim_d
    pad_cols = dim_f
    mtx_k = np.pad(np.diag(diag_s), [(0, pad_rows), (0, pad_cols)])
    mtx_k += mtx_k_second_term

    # Possible optimization
    # mtx_k = np.diag(diag_s) + mtx_uta @ mtx_vtb.T

    # Smaller (dimension r x r) SVD
    mtx_uprime, diag_sprime, mtx_vprimet = np.linalg.svd(mtx_k, full_matrices=False)
    mtx_vprime = mtx_vprimet.T

    # update SVD
    new_mtx_u = mtx_u_p @ mtx_uprime
    new_diag_s = diag_sprime
    new_mtx_v = mtx_v_q @ mtx_vprime
    # possible optimization
    # new_mtx_u = mtx_u @ mtx_uprime
    # new_diag_s = diag_sprime
    # new_mtx_v = (mtx_v @ mtx_vprime).T
    return new_mtx_u, new_diag_s, new_mtx_v

def minimal_downdate(mtx_u, diag_s, mtx_v, col_data_to_remove, col_idxs_to_remove):
    '''returns new_mtx_u, new_diag_s, new_mtx_v'''
    dim_p, dim_q = mtx_u.shape[0], mtx_v.shape[0]
    dim_r = len(diag_s)
    assert mtx_u.shape[1] == dim_r
    assert mtx_v.shape[1] == dim_r

    mtx_a = -col_data_to_remove
    dim_c = col_data_to_remove.shape[1]
    mtx_b = np.zeros((dim_q, dim_c))
    # In each column of B, place a 1 in the row corresponding to the
    # column of the full matrix X we're removing
    mtx_b[col_idxs_to_remove, np.arange(dim_c)] = 1

    # Omit computation of P, R_A, Q, R_B
    # as they represent the portion of the update matrix AB^T
    # not captured in the original basis and we're making
    # the assumption that downdating our (potentially truncated)
    # SVD doesn't require new basis vectors, merely rotating the
    # existing ones. Indeed, P, R_A, Q, and R_B are very close to
    # machine zero

    # "Eigen-code" the update matrices from both sides
    # into the space where X is diagonalized (and truncated)
    mtx_uta = mtx_u.T @ mtx_a   # U in p x r, A in p x c, O(p r c) -> r x c
    mtx_vtb = mtx_v.T @ mtx_b   # V in q x r, B in q x c, O(q r c) -> r x c
    
    # Additive modification to inner diagonal matrix
    mtx_k = np.diag(diag_s)
    mtx_k += mtx_uta @ mtx_vtb.T  # U^T A is r x c, (V^T B)^T is c x r, O(r c r) -> r x r

    # Smaller (dimension r x r) SVD to re-diagonalize, giving
    # rotations of the left and right singular vectors and
    # updated singular values
    mtx_uprime, diag_sprime, mtx_vprimet = np.linalg.svd(mtx_k, full_matrices=False)  # K is r x r, O(r^3)
    mtx_vprime = mtx_vprimet.T

    # Compute new SVD by applying the rotations
    new_mtx_u = mtx_u @ mtx_uprime
    new_diag_s = diag_sprime
    new_mtx_v = mtx_v @ mtx_vprime
    # columns of X become rows of V, delete the dropped ones
    # new_mtx_v = np.delete(new_mtx_v, col_idxs_to_remove, axis=0)
    return new_mtx_u, new_diag_s, new_mtx_v
