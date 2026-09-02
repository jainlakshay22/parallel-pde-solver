import numpy as np

def compute_ac(returns_x: np.ndarray, returns_y: np.ndarray, k_values: list[int] = [5, 12, 25, 50]) -> tuple[float, int]:
    """
    Computes Aligned Correlation Distance (loose metric).
    Lower AC distance = better aligned pair.
    
    Optimizes over window sizes k in {5, 12, 25, 50}.
    """
    rx = np.asarray(returns_x, dtype=np.float64)
    ry = np.asarray(returns_y, dtype=np.float64)
    n = len(rx)
    
    best_ac = np.inf
    best_k = k_values[0]
    
    for k in k_values:
        rx_pad = np.pad(rx, (k, k), mode="constant", constant_values=0.0)
        ry_pad = np.pad(ry, (k, k), mode="constant", constant_values=0.0)
        
        # Build local CR matrix
        CR = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sx = rx_pad[i : i + 2 * k + 1]
            norm_sx = np.linalg.norm(sx)
            for j in range(n):
                sy = ry_pad[j : j + 2 * k + 1]
                norm_sy = np.linalg.norm(sy)
                if norm_sx == 0.0 or norm_sy == 0.0:
                    CR[i, j] = 2.0
                else:
                    c = np.clip(np.dot(sx, sy) / (norm_sx * norm_sy), -1.0, 1.0)
                    CR[i, j] = 2.0 * (1.0 - c)
                    
        # Dynamic Programming alignment
        D = np.full((n, n), np.inf, dtype=np.float64)
        D[0, 0] = CR[0, 0]
        for i in range(1, n):
            D[i, 0] = D[i - 1, 0] + CR[i, 0]
        for j in range(1, n):
            D[0, j] = D[0, j - 1] + CR[0, j]
            
        for i in range(1, n):
            for j in range(1, n):
                D[i, j] = CR[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
                
        # Backtrack optimal alignment path
        i, j = n - 1, n - 1
        path_p, path_q = [i], [j]
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
                if step == 0:
                    i, j = i - 1, j - 1
                elif step == 1:
                    i -= 1
                else:
                    j -= 1
            path_p.append(i)
            path_q.append(j)
            
        path_p = np.array(path_p)
        path_q = np.array(path_q)
        
        # Compute AC metric on aligned points
        aligned_rx = rx[path_p]
        aligned_ry = ry[path_q]
        norm_arx = np.linalg.norm(aligned_rx)
        norm_ary = np.linalg.norm(aligned_ry)
        
        if norm_arx == 0.0 or norm_ary == 0.0:
            gamma = 0.0
        else:
            gamma = np.clip(np.dot(aligned_rx, aligned_ry) / (norm_arx * norm_ary), -1.0, 1.0)
            
        ac_dist = np.sqrt(max(0.0, 2.0 * (1.0 - gamma)))
        if ac_dist < best_ac:
            best_ac = ac_dist
            best_k = k
            
    return float(best_ac), int(best_k)