@everywhere module alsparatuck
include("tnsrutils.jl") ;
include("paratuckutils.jl") ;
import paratuckutils, tnsrutils
export nnparatuck2als

function nnparatuck2als(X, latfact, A, R, B, DA, DB, maxiter=1000, epsobjfun=1.0E-3)
    filepath = "" ;

    Xsize = size(X) ;
    I = Xsize[1] ;
    J = Xsize[2] ;
    K = Xsize[3] ;
    P = latfact[1] ;
    Q = latfact[2] ;
#
# we store the evolution of the calculation
    fk = zeros( maxiter + 1 ) ;
    tmstamp = zeros( maxiter + 1 ) ;
#
# initialization of matrices and tensors
    #A, DA, R, DB, B = tnsrutils.paratuck2init(Xsize, latfact) ;
    Xh = tnsrutils.buildparatuck2(A, DA, R, DB, B') ;
    curvo = tnsrutils.objfun(X,Xh) ;
    print("   >> initialization    Obj Fun:") ;
    println(round(curvo, 3)) ;
    fk[1] = curvo ;
    tmstamp[1] = time() ;
#
# non-negative paratuck2 ALS iterative process
    i = 1 ;
    while i <= maxiter
# 1 estimation of A
        tmpX = zeros(I, J*K) ;
        tmpF = zeros(P, J*K) ;
        for n = 1:K
            tmpX[:,(n-1)*J+1:n*J] = X[:,:,n] ;
            tmpF[:,(n-1)*J+1:n*J] = DA[:,:,n] * R * DB[:,:,n] * B';
        end
        num = (tmpX * tmpF') + 1.0E-9 ;
        denum = A * (tmpF * tmpF') + 1.0E-9 ;
        A = A .* (num ./ denum) ;
#
# 2 estimation of DA
        for n = 1:K
            F = B * DB[:,:,n] * R' ;
            Z = (tnsrutils.krao(F, A))' ;
            vecXn = hcat(X[:,:,n]...) ;
            num = (vecXn * Z') + 1.0E-9 ;
            denum = ( diag(DA[:,:,n])' * Z * Z' ) + 1.0E-9 ;
            DA[:,:,n] = DA[:,:,n] .* (num ./ denum) ;
        end
#
# 3 estimation of R
        vecXk = zeros( I * J * K ) ;
        Z = zeros(I*J*K, P*Q) ;
        for n = 1:K
            strt = I*J*(n-1)+1 ;
            nd = I*J*n ;
            vecXk[strt:nd] = hcat(X[:,:,n]...) ;
            Z[strt:nd, :] = kron( (B * DB[:,:,n]), (A * DA[:,:,n]) ) ;
        end
        num = (Z' * vecXk ) + 1.0E-9 ;
        fact01 = hcat(R...) ;
        denum = ( fact01 * ( Z' * Z ) ) + 1.0E-9 ;
        R = reshape( fact01' .* ( num ./ Vector(denum[1,:]) ), (P,Q) ) ;
#
# 4 estimation of DB
        for n = 1:K
            F = (R' * DA[:,:,n] * A')' ;
            Z = tnsrutils.krao(B, F) ;
            vecXk = hcat(X[:,:,n]...) ;
            num = ( vecXk * Z ) + 1.0E-9 ;
            denum = ( diag(DB[:,:,n])' * Z' * Z ) + 1.0E-9 ;
            DB[:,:,n] = DB[:,:,n] .* (num ./ denum) ;
        end
#
# 5 estimation of B
        tmpX = zeros(J, I*K) ;
        tmpF = zeros(Q, I*K) ;
        for n =1:K
            tmpX[:,(n-1)*I+1:n*I] = X[:,:,n]' ;
            tmpF[:,(n-1)*I+1:n*I] = (A * DA[:,:,n] * R * DB[:,:,n])' ;
        end
        num = (tmpX * tmpF') + 1.0E-9 ;
        denum = (B * tmpF * tmpF') + 1.0E-9 ;
        B = B .* (num ./ denum) ;
#
# convergence criteria
        Xh = tnsrutils.buildparatuck2(A, DA, R, DB, B') ;
        curvo = tnsrutils.objfun(X,Xh) ;
        fk[i+1] = curvo ;
        tmstamp[i+1] = time() ;
#
# display calculation evolution
        if i%(maxiter/10) == 0 || i == maxiter
            print("   >> ", i/maxiter*100, "%") ;
            print("     Obj. fun: ", round(curvo, 3)) ;
            println("") ;
            xk = paratuckutils.mat2vectParatuck2( A, DA, R, DB, B ) ;
            writedlm(string(filepath, string(i), "xk.txt"), xk, ",") ;
            writedlm(string(filepath, string(i),"fk.txt"), fk, ",") ;
            writedlm(string(filepath, string(i),"tmstamp.txt"), tmstamp, ",") ;
        end
#
        if curvo > epsobjfun
            i += 1 ;
        else
            i = maxiter + 1 ;
        end
    end
    print("=====================") ;
    println("") ;
    println("End of execution") ;
    print("   >> Number of iterations: ") ;
    println(i - 1) ;
    Xh = tnsrutils.buildparatuck2(A, DA, R, DB, B') ;
    print("   >> Overall difference: ") ;
    println(round(tnsrutils.objfun(X,Xh), 3)) ;
#
# we save the final results of the calculation
    xk = paratuckutils.mat2vectParatuck2( A, DA, R, DB, B ) ;
    writedlm(string(filepath, string(i), "xk.txt"), xk, ",") ;
    writedlm(string(filepath, string(i),"fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(i),"tmstamp.txt"), tmstamp, ",") ;
    return A, DA, R, DB, B ;
#
# Last card of function nnparatuck2als.
#
end
#
#   Last card of module paratuckals.
#
end
