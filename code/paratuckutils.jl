@everywhere module paratuckutils
using tnsrutils
export mat2vectParatuck2, vect2matParatuck2, objfunParatuck2

function mat2vectParatuck2( A,DA,H,DB,B )
    if length(size(DA)) == 3
        I = size(A)[1] ;
        J = size(B)[1] ;
        K = size(DA)[3] ;
        P = size(A)[2] ;
        Q = size(B)[2] ;
    else
        println("only 3-way Paratuck2 implemented") ;
    end
#
    x = zeros( I*P + K*P + P*Q + K*Q + Q*J ) ;
    cnt = 1 ;
    for n = 1:5
        if n == 1 || n == 3 || n == 5
            if n == 1
                elts = I*P ;
                x[cnt:elts] = vcat(A...) ;
            elseif n == 3
                elts = cnt + P*Q - 1 ;
                x[cnt:elts] = vcat(H...) ;
            elseif n == 5
                elts = cnt + J*Q - 1 ;
                x[cnt:elts] = vcat(B...) ;
            end
            cnt = elts + 1 ;
        else
            if n == 2
                maxLin = P ;
                for ktr = 1:K
                    elts = cnt + maxLin - 1 ;
                    x[cnt:elts] = diag( DA[:,:,ktr] ) ;
                    cnt = elts + 1 ;
                end
            else
                maxLin = Q ;
                for ktr = 1:K
                    elts = cnt + maxLin - 1 ;
                    x[cnt:elts] = diag( DB[:,:,ktr] ) ;
                    cnt = elts + 1 ;
                end
            end
        end
    end
    return x ;
#
#     Last card of function mat2vecParatuck2.
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

function vect2matParatuck2(xk, a, latfact)
    I = a[1] ;
    J = a[2] ;
    K = a[3] ;
    P = latfact[1] ;
    Q = latfact[2] ;
    A = zeros(I,P) ;
    DA = zeros(P,P,K) ;
    H = zeros(P,Q) ;
    DB = zeros(Q,Q,K) ;
    B = zeros(J,Q) ;
#
    cnt = 1 ;
    index = 0 ;
    for n = 1:5
        if n == 1 || n == 3 || n == 5
            if n == 1
                elts = cnt + I * P - 1 ;
                A = fill2darr(xk, A, cnt, I, P) ;
            elseif n == 3
                elts = cnt + P * Q - 1 ;
                H = fill2darr(xk, H, cnt, P, Q) ;
            elseif n == 5
                elts = cnt + J * Q - 1 ;
                B = fill2darr(xk, B, cnt, J, Q) ;
            end
        else
            if n == 2
                elts = cnt + K * P - 1 ;
                DA = fill3darr(xk, DA, cnt, K, P) ;
            elseif n == 4
                elts = cnt + K * Q - 1 ;
                DB = fill3darr(xk, DB, cnt, K, Q) ;
            end
        end
        index = index + 1 ;
        cnt = elts + 1 ;
    end
    return A, DA, H, DB, B ;
#
# Last card of function vect2matParatuck2.
#
end

function objfunParatuck2(X, xk, latfact)
    a = size(X) ;
    A, DA, R, DB, B = paratuckutils.vect2matParatuck2(xk, a, latfact) ;
    BT = B';
    Xh = tnsrutils.buildparatuck2(A, DA, R, DB, BT) ;
    fxk = tnsrutils.objfun(X,Xh) ;
    return fxk ;
#
# Last card of function objfunParatuck2.
#
end
#
#   Last card of module paratuckutils.
#
end
