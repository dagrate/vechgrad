@everywhere module cputils
using tnsrutils
export mat2vectcp3, vect2matcp3, objfuncp3

function mat2vectcp3( A,B,C )
    I = size(A)[1] ;
    J = size(B)[1] ;
    K = size(C)[1] ;
    R = size(A)[2] ;
#
    x = zeros( I*R + J*R + K*R ) ;
    cnt = 1 ;
    for n = 1:3
        if n == 1
            elts = I*R ;
            x[cnt:elts] = vcat(A...) ;
        elseif n == 2
            elts = cnt + J*R - 1 ;
            x[cnt:elts] = vcat(B...) ;
        elseif n == 3
            elts = cnt + K*R - 1 ;
            x[cnt:elts] = vcat(C...) ;
        end
        cnt = elts + 1 ;
    end
    return x ;
#
#     Last card of function mat2veccp3.
#
end

function fill2darr(xk, arr, i0, l, c)
    for n = 1:c
        arr[:,n] = xk[i0:i0+l-1] ;
        i0 += l ;
    end
    return arr ;
#
# Last card of function fill2darr.
#
end

function fill3darr(xk, arr, i0, d, l)
    for m = 1:d
        for n = 1:l
            arr[n, n, m] = xk[i0] ;
            i0 += 1 ;
        end
    end
    return arr ;
#
# Last card of function fill3darr.
#
end

function vect2matcp3(xk, a, latfact)
    I = a[1] ;
    J = a[2] ;
    K = a[3] ;
    R = latfact ;
    A = zeros(I,R) ;
    B = zeros(J,R) ;
    C = zeros(K,R) ;
#
    cnt = 1 ;
    index = 0 ;
    for n = 1:3
        if n == 1
            elts = cnt + I * R - 1 ;
            A = fill2darr(xk, A, cnt, I, R) ;
        elseif n == 2
            elts = cnt + J * R - 1 ;
            B = fill2darr(xk, B, cnt, J, R) ;
        elseif n == 3
            elts = cnt + K * R - 1 ;
            C = fill2darr(xk, C, cnt, K, R) ;
        end
        index = index + 1 ;
        cnt = elts + 1 ;
    end
    return A, B, C ;
#
# Last card of function vect2matcp3.
#
end

function objfuncp3(X, xk, latfact)
    a = size(X) ;
    I = size(X)[1] ;
    J = size(X)[2] ;
    K = size(X)[3] ;
    R = latfact ;
    A, B, C = vect2matcp3(xk, a, latfact) ;
    Xh = tnsrutils.buildCP3(A, B, C) ;
    fxk = tnsrutils.objfun(X,Xh) ;
    return fxk ;
#
# Last card of function objfuncp3.
#
end
#
#   Last card of module cputils.
#
end
