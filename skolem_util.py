from scipy.special import binom as comb

def combinadics_from_index(N, k):
    """
    This function converts a natural number N as sequences of k-combinations.
    This is referred to as the combinatorial number system of degree k (for some positive integer k).

    See https://planetcalc.com/8592/
    """
    combinadics = [0] * k

    c_start = max(N, k)

    for j in range(k, 0, -1):
        #############################################################
        # Find the value of c using the bisection method.
        #############################################################
        left = 0
        
        right = c_start + 1  # Increase the range by 1 to include c_start
        
        while left < right:
        
            mid = (left + right) // 2
            q = comb(mid, j)
            
            if q <= N:
                left = mid + 1
            else:
                right = mid
                
        c = left - 1  # Return the previous value since q > N
        #############################################################

        q = comb(c, j)
        c_start = c - 1
        N -= q
        combinadics[k - j] = c

    return combinadics

def index_from_combinadics(combinadics):
    """
    This function returns index from the combinadics.

    Note:
        assert index_from_combinadics(combinadics_from_index(N, k)) == N
    """
    return sum(
        comb(c, j + 1) for j, c in enumerate(reversed(combinadics))
    )

def skolem(ntuple):
    """
    Evaluate the Skolem polynomial from tuple
    """
    return int(sum(
        comb(sum(ntuple[:k]) + k - 1, k) for k in range(1, len(ntuple) + 1)
    ))

def tuple_from_skolem(S, k):
    "The inverse of the Skolem polynomials"

    combinadics = combinadics_from_index(S, k)

    # assert index_from_combinadics(combinadics) == S

    ntuple = [0] * k

    ntuple[0] = combinadics[-1]

    for j in range(1, k):
        ntuple[j] = combinadics[k - j - 1] - j - sum(ntuple[:j])

    return ntuple