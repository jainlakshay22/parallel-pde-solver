import numpy as np


def compute_local_cct_matrix(
    returns_x: np.ndarray, 
    returns_y: np.ndarray, 
    k: int = 12
) -> np.ndarray:
    
    rx = np.asarray(returns_x, dtype=np.float64)
    ry = np.asarray(returns_y, dtype=np.float64)
    n = len(rx)
    
    # Zero-pad ends by k so boundaries at t=0 and t=N-1 can have full (2k+1) windows
    rx_pad = np.pad(rx, (k, k), mode="constant", constant_values=0.0)
    ry_pad = np.pad(ry, (k, k), mode="constant", constant_values=0.0)
    
    cct_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        seg_x = rx_pad[i : i + 2 * k + 1]
        norm_x = np.linalg.norm(seg_x)
        
        for j in range(n):
            seg_y = ry_pad[j : j + 2 * k + 1]
            norm_y = np.linalg.norm(seg_y)
            
            if norm_x == 0.0 or norm_y == 0.0:
                cct_matrix[i, j] = 0.0
            else:
                val = np.dot(seg_x, seg_y) / (norm_x * norm_y)
                cct_matrix[i, j] = np.clip(val, -1.0, 1.0)
                
    return cct_matrix


def compute_cr_distance_matrix(cct_matrix: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - cct_matrix)