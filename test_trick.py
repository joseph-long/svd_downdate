import numpy as np
import trick

def test_downdate():
    dim_p = 6
    dim_q = 5
    # np.random.seed(0)
    # Initialize p x q noise matrix X
    mtx_x = np.random.randn(dim_p, dim_q)
    
    # Initialize thin SVD
    dim_r = dim_q  # for truncated, r < q
    mtx_u, diag_s, mtx_vt = np.linalg.svd(mtx_x, full_matrices=False)
    mtx_v = mtx_vt.T

    # Select columns to remove
    col_idxs_to_remove = [1, 2]
    col_data_to_remove = mtx_x[:,col_idxs_to_remove]
    new_mtx_u, new_diag_s, new_mtx_v = trick.downdate(
        mtx_u,
        diag_s,
        mtx_v,
        col_data_to_remove,
        col_idxs_to_remove
    )

    # X with columns zeroed for comparison
    final_mtx_x = mtx_x.copy()
    final_mtx_x[:,col_idxs_to_remove] = 0
    assert np.allclose(new_mtx_u @ np.diag(new_diag_s) @ new_mtx_v.T, final_mtx_x)

    # SVD of final matrix for comparison
    final_mtx_u, final_diag_s, final_mtx_vt = np.linalg.svd(final_mtx_x)

    n_nonzero = np.count_nonzero(final_diag_s > 1e-14)
    assert n_nonzero == 3

    assert trick.compare_columns_modulo_sign(
        new_mtx_u[:,:n_nonzero],
        final_mtx_u[:,:n_nonzero],
    )



def test_minimal_downdate():
    dim_p = 6
    dim_q = 5
    # np.random.seed(0)
    # Initialize p x q noise matrix X
    mtx_x = np.random.randn(dim_p, dim_q)
    
    # Initialize thin SVD
    dim_r = dim_q  # for truncated, r < q
    mtx_u, diag_s, mtx_vt = np.linalg.svd(mtx_x, full_matrices=False)
    mtx_v = mtx_vt.T

    # Select columns to remove
    col_idxs_to_remove = [1, 2]
    col_data_to_remove = mtx_x[:,col_idxs_to_remove]
    new_mtx_u, new_diag_s, new_mtx_v = trick.minimal_downdate(
        mtx_u,
        diag_s,
        mtx_v,
        col_data_to_remove,
        col_idxs_to_remove
    )

    # X with columns zeroed for comparison
    final_mtx_x = mtx_x.copy()
    final_mtx_x[:,col_idxs_to_remove] = 0
    assert np.allclose(new_mtx_u @ np.diag(new_diag_s) @ new_mtx_v.T, final_mtx_x)

    # SVD of final matrix for comparison
    final_mtx_u, final_diag_s, final_mtx_vt = np.linalg.svd(final_mtx_x)

    n_nonzero = np.count_nonzero(final_diag_s > 1e-14)
    assert n_nonzero == 3

    assert trick.compare_columns_modulo_sign(
        new_mtx_u[:,:n_nonzero],
        final_mtx_u[:,:n_nonzero],
    )
