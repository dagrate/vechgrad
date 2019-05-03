@everywhere module tuckerutils
using tnsrutils
export mat2vecttucker, vect2mattucker, objfuntucker

function mat2vecttucker( A,B,C,G )
    I = size(A)[1] ;
    J = size(B)[1] ;
    K = size(C)[1] ;
    P = size(A)[2] ;
    Q = size(B)[2] ;
    R = size(C)[2] ;
#
    x = zeros( I*P + J*Q + K*R + P*Q*R ) ;
    cnt = 1 ;
    for n = 1:4
        if n == 1
            elts = I*P ;
            x[cnt:elts] = vcat(A...) ;
        elseif n == 2
            elts = cnt + J*Q - 1 ;
            x[cnt:elts] = vcat(B...) ;
        elseif n == 3
            elts = cnt + K*R - 1 ;
            x[cnt:elts] = vcat(C...) ;
        elseif n == 4
            elts = cnt + P*Q*R - 1 ;
            x[cnt:elts] = vcat(G...) ;
        end
        cnt = elts + 1 ;
    end
    return x ;
#
#     Last card of function mat2vectucker.
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

function fill3darr(xk, arr, i0)
    P = size(arr)[1] ;
    Q = size(arr)[2] ;
    R = size(arr)[3] ;
    for lr = 1:R
        for lq = 1:Q
            for lp = 1:P
                arr[lp, lq, lr] = xk[i0] ;
                i0 += 1 ;
            end
        end
    end
    return arr ;
#
# Last card of function fill3darr.
#
end

function vect2mattucker(xk, a, latfact)
    I = a[1] ;
    J = a[2] ;
    K = a[3] ;
    P = latfact[1] ;
    Q = latfact[2] ;
    R = latfact[3] ;
    A = zeros(I,P) ;
    B = zeros(J,Q) ;
    C = zeros(K,R) ;
    G = zeros(P,Q,R) ;
#
    cnt = 1 ;
    index = 0 ;
    for n = 1:4
        if n == 1
            elts = cnt + I * P - 1 ;
            A = fill2darr(xk, A, cnt, I, P) ;
        elseif n == 2
            elts = cnt + J * Q - 1 ;
            B = fill2darr(xk, B, cnt, J, Q) ;
        elseif n == 3
            elts = cnt + K * R - 1 ;
            C = fill2darr(xk, C, cnt, K, R) ;
        elseif n == 4
            elts = cnt + P * Q * R - 1 ;
            G = fill3darr(xk, G, cnt) ;
        end
        index = index + 1 ;
        cnt = elts + 1 ;
    end
    return A, B, C, G ;
#
# Last card of function vect2mattucker.
#
end

function objfuntucker(X, xk, latfact)
    a = size(X) ;
    I = size(X)[1] ;
    J = size(X)[2] ;
    K = size(X)[3] ;
    P = latfact[1] ;
    Q = latfact[2] ;
    R = latfact[3] ;
    A, B, C, G = vect2mattucker(xk, a, latfact) ;
    Xh = tnsrutils.buildtucker(A, B, C, G) ;
    fxk = tnsrutils.objfun(X,Xh) ;
    return fxk ;
#
# Last card of function objfuntucker.
#
end
#
#   Last card of module tuckerutils.
#
end
