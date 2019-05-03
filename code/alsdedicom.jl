@everywhere module alsdedicom
include("tnsrutils.jl") ;
include("dedicomutils.jl") ;
import dedicomutils, tnsrutils
export nndedicomals

function nndedicomals(X, latfact, A, D, H, maxiter=1000, epsobjfun=1.0E-3)
    filepath = "" ;

    Xsize = size(X) ;
    I = Xsize[1] ;
    J = Xsize[2] ;
    K = Xsize[3] ;
    P = latfact ;
#
# we store the evolution of the calculation
    fk = zeros( maxiter + 1 ) ;
    tmstamp = zeros( maxiter + 1 ) ;
#
# initialization of matrices and tensors
    #A, D, H = tnsrutils.dedicominit(Xsize, latfact) ;
    Xh = tnsrutils.buildDedicom(A, D, H) ;
    curvo = tnsrutils.objfun(X,Xh) ;
    print("   >> initialization    Obj Fun:") ;
    println(round(curvo, 3)) ;
    fk[1] = curvo ;
    tmstamp[1] = time() ;
#
# non-negative dedicom ALS iterative process
    i = 1 ;
    while i <= maxiter
# 1 estimation of A
        tmpX = zeros(I, J*K) ;
        tmpF = zeros(P, J*K) ;
        for n = 1:K
            tmpX[:,(n-1)*J+1:n*J] = X[:,:,n] ;
            tmpF[:,(n-1)*J+1:n*J] = D[:,:,n] * H * D[:,:,n] * A';
        end
        num = (tmpX * tmpF') + 1.0E-9 ;
        denum = A * (tmpF * tmpF') + 1.0E-9 ;
        A = A .* (num ./ denum) ;
#
# 2 estimation of D
        for n = 1:K
            F = A * D[:,:,n] * H' ;
            Z = (tnsrutils.krao(F, A))' ;
            vecXn = hcat(X[:,:,n]...) ;
            num = (vecXn * Z') + 1.0E-9 ;
            denum = ( diag(D[:,:,n])' * Z * Z' ) + 1.0E-9 ;
            D[:,:,n] = D[:,:,n] .* (num ./ denum) ;
        end
#
# 3 estimation of R
        vecXk = zeros( I * J * K ) ;
        Z = zeros(I*J*K, P*P) ;
        for n = 1:K
            strt = I*J*(n-1)+1 ;
            nd = I*J*n ;
            vecXk[strt:nd] = hcat(X[:,:,n]...) ;
            Z[strt:nd, :] = kron( (A * D[:,:,n]), (A * D[:,:,n]) ) ;
        end
        num = (Z' * vecXk ) + 1.0E-9 ;
        fact01 = hcat(H...) ;
        denum = ( fact01 * ( Z' * Z ) ) + 1.0E-9 ;
        H = reshape( fact01' .* ( num ./ Vector(denum[1,:]) ), (P,P) ) ;
#
# convergence criteria
        Xh = tnsrutils.buildDedicom(A, D, H) ;
        curvo = tnsrutils.objfun(X,Xh) ;
        fk[i+1] = curvo ;
        tmstamp[i+1] = time() ;
#
# display calculation evolution
        if i%(maxiter/10) == 0
            print("   >> ", i/maxiter*100, "%") ;
            print("     Obj. fun: ", round(curvo, 3)) ;
            println("") ;
            xk = dedicomutils.mat2vectDedicom( A,D,H ) ;
            writedlm(string(filepath, string(i), "xk.txt"), xk, ",") ;
            writedlm(string(filepath, string(i),"fk.txt"), fk, ",") ;
            writedlm(string(filepath, string(i),"tmstamp.txt"), tmstamp, ",") ;
        end

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
    Xh = tnsrutils.buildDedicom(A, D, H) ;
    print("   >> Overall difference: ") ;
    println(round(tnsrutils.objfun(X, Xh), 3)) ;
#
# we save the final results of the calculation
    xk = dedicomutils.mat2vectDedicom( A,D,H ) ;
    writedlm(string(filepath, string(i), "xk.txt"), xk, ",") ;
    writedlm(string(filepath, string(i),"fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(i),"tmstamp.txt"), tmstamp, ",") ;
    return A, D, H ;
#
# Last card of function nndedicomals.
#
end
#
#   Last card of module dedicomals.
#
end
