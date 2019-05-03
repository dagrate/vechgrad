@everywhere module alscp
include("tnsrutils.jl") ;
include("cputils.jl") ;
import cputils, tnsrutils
export nncpals

function nncpals(X, rankR, A, B, C, maxiter=1000, epsobjfun=1.0E-3)
    filepath = "" ;

    Xsize = collect(size(X)) ;
    allmode = collect(1:length(Xsize)) ;
    if length(Xsize) > 3
        return println("Only third order parafac implemented") ;
    end
#
# we store the evolution of the calculation
    fk = zeros( maxiter + 1 ) ;
    tmstamp = zeros( maxiter + 1 ) ;
#
# random initialization
    # A = rand(Xsize[1], rankR) ;
    # B = rand(Xsize[2], rankR) ;
    # C = rand(Xsize[3], rankR) ;
    Xh = tnsrutils.buildCP3(A,B,C) ;
    curvo = tnsrutils.objfun(X,Xh) ;
    print("   >> initialization    Obj Fun:") ;
    println(round(curvo, 3)) ;
#
    fk[1] = curvo ;
    tmstamp[1] = time() ;
#
# non-negative CP ALS iterative process
    i = 1 ;
    while i <= maxiter
        for mode = 1:length(Xsize)
#
# Hadamard product on all mode != n
            V = ones(rankR, rankR) ;
            for n = 1:length(Xsize)
                if n != mode
                    if n == 1
                        tmpV = A ;
                    elseif n == 2
                        tmpV = B ;
                    elseif n == 3
                        tmpV = C ;
                    end
                    V = V .* (tmpV' * tmpV) ;
                end
            end
#
# Khatri-Rao product on all mode != n
            if mode == 1
                W = tnsrutils.krao(C, B) ;
            elseif mode == 2
                W = tnsrutils.krao(C, A) ;
            elseif mode == 3
                W = tnsrutils.krao(B, A) ;
            end
#
# non negative update
            num = (tnsrutils.tnsrunfold(X, mode) * W) + 1.0E-9 ;
            if mode == 1
                denum = (A * (W' * W)) + 1.0E-9 ;
                A = A .* (num ./ denum) ;
            elseif mode == 2
                denum = (B * (W' * W)) + 1.0E-9 ;
                B = B .* (num ./ denum) ;
            elseif mode == 3
                denum = (C * (W' * W)) + 1.0E-9 ;
                C = C .* (num ./ denum) ;
            end
        end
#
# calculation evolution
        Xh = tnsrutils.buildCP3(A,B,C) ;
        curvo = tnsrutils.objfun(X,Xh) ;
        fk[i+1] = curvo ;
        tmstamp[i+1] = time() ;
#
        if i%(maxiter/10) == 0 || i == maxiter
            print("   >> ", i / maxiter * 100, "%") ;
            print("     Obj. fun: ", round(curvo, 3)) ;
            println("") ;
            xk = cputils.mat2vectcp3(A, B, C) ;
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
    print("Number of iterations: ") ;
    println(i - 1) ;
    Xh = tnsrutils.buildCP3(A, B, C) ;
    print("Overall difference: ") ;
    println(round(tnsrutils.objfun(X, Xh), 3)) ;
#
# we save the final results of the calculation
    xk = cputils.mat2vectcp3(A, B, C) ;
    writedlm(string(filepath, string(i), "xk.txt"), xk, ",") ;
    writedlm(string(filepath, string(i),"fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(i),"tmstamp.txt"), tmstamp, ",") ;
#
    return A, B, C ;
#
# Last card of function nncpals.
#
end

#
#   Last card of module alscp.
#
end
