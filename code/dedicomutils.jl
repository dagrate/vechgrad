@everywhere module dedicomutils
using tnsrutils
export mat2vectDedicom, vect2matDedicom, objfunDedicom

function mat2vectDedicom( A,D,H )
    if length(size(D)) == 3
        I = size(A)[1] ;
        K = size(D)[3] ;
        P = size(A)[2] ;
    else
        println("only 3-way Dedicom implemented") ;
    end
#
    x = zeros( I*P + K*P + P*P ) ;
    cnt = 1 ;
    for n = 1:3
        if n == 1 || n == 3
            if n == 1
                elts = I*P ;
                x[cnt:elts] = vcat(A...) ;
            elseif n == 3
                elts = cnt + P*P - 1 ;
                x[cnt:elts] = vcat(H...) ;
            end
            cnt = elts + 1 ;
        else
            maxLin = P ;
            for ktr = 1:K
                elts = cnt + maxLin - 1 ;
                x[cnt:elts] = diag( D[:,:,ktr] ) ;
                cnt = elts + 1 ;
            end
        end
    end
    return x ;
#
#     Last card of function mat2vecDedicom.
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

function vect2matDedicom(xk, a, latfact)
    I = a[1] ;
    K = a[3] ;
    P = latfact ;
    A = zeros(I,P) ;
    D = zeros(P,P,K) ;
    H = zeros(P,P) ;
#
    cnt = 1 ;
    index = 0 ;
    for n = 1:3
        if n == 1 || n == 3
            if n == 1
                elts = cnt + I * P - 1 ;
                A = fill2darr(xk, A, cnt, I, P) ;
            elseif n == 3
                elts = cnt + P * P - 1 ;
                H = fill2darr(xk, H, cnt, P, P) ;
            end
        else
            elts = cnt + K * P - 1 ;
            D = fill3darr(xk, D, cnt, K, P) ;
        end
        index = index + 1 ;
        cnt = elts + 1 ;
    end
    return A, D, H ;
#
# Last card of function vect2matParatuck2.
#
end

function objfunDedicom(X, xk, latfact)
    a = size(X) ;
    A, DA, H = vect2matDedicom(xk, a, latfact) ;
    Xh = tnsrutils.buildDedicom(A, DA, H) ;
    fxk = tnsrutils.objfun(X,Xh) ;
    return fxk ;
#
# Last card of function objfunDedicom.
#
end
#
#   Last card of module dedicomutils.
#
end
