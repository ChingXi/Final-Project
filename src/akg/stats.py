from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    if b < 0 or c < 0:
        raise ValueError(f"counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)
