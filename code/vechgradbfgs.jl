include("rosnbrck.jl") ;
include("tnsrutils.jl") ;
include("cputils.jl") ;
include("tuckerutils.jl")
include("dedicomutils.jl") ;
include("paratuckutils.jl") ;
include("utilsappder.jl") ;
include("utilsappderparfor.jl") ;
include("linsrch.jl") ;

import rosnbrck, tnsrutils, cputils, tuckerutils,
        dedicomutils, paratuckutils, utilsappder, utilsappderparfor, linsrch
using StatsBase
using Optim

function fcp3(x)
    A, B, C = cputils.vect2matcp3(x, a, latfact) ;
    Xh = tnsrutils.buildCP3(A, B, C) ;
    return tnsrutils.objfun(X,Xh) ;
#
# Last card of function objfuncp3bfgs.
#
end

function fdedicom(x)
    A, D, H = dedicomutils.vect2matDedicom(x, a, latfact) ;
    Xh = tnsrutils.buildDedicom(A, D, H) ;
    return tnsrutils.objfun(X,Xh) ;
#
# Last card of function objfunParatuck2.
#
end

function f(x)
    A, DA, R, DB, B = paratuckutils.vect2matParatuck2(x, a, latfact) ;
    BT = B';
    Xh = tnsrutils.buildparatuck2(A, DA, R, DB, BT) ;
    return tnsrutils.objfun(X,Xh) ;
#
# Last card of function objfunParatuck2.
#
end

function g!(G, x)
#
# even if batchind is not used in the function, we let it as input for
# compatibility issue with the serial execution
#
    #f = objfuncp3bfgs ;
    order = 4 ;
    epsilon = 1.0E-4 ;
#
    #dx = SharedArray{Float64}( length(xf), length(xf) ) ;
    dx = zeros( length(x), length(x) ) ;
    #jac = SharedArray{Float64}( length(xf) ) ;
    #@parallel for i = 1:length(xk)
    for i = 1:length(x)
        dx[i, i] = epsilon ;
    end
#
# gradient computation
    #@sync @parallel for n = 1:length(xk)
    for n = 1:length(x)
        eps2 = 1 / (24*epsilon) ;
        fact0 = f( x-2*dx[n,:] ) ;
        fact1 = f( x-dx[n,:] ) ;
        fact2 = f( x+dx[n,:] ) ;
        fact3 = f( x+2*dx[n,:] ) ;
        G[n] = ( 2*fact0 - 16*fact1 + 16*fact2 - 2*fact3 ) * eps2 ;
    end
    #return jac ;
#
# Last card of function approxJbfgscp3.
#
end



rsltn = 2 ;
a = (4,5,6) ;
maxitrtn = 10 ;
batchnmb = 50 ;
epochnmb = 1 ;
sim = "cifar10" ;

flpath = ""
filepath = "" ;
# we initialize the tensor X and a
if sim == "cifar10" || sim == "cifar100"
    a = (32, 32, 16) ;
    if sim == "cifar10"
        X = readdlm( string(flpath, "cifar10_.csv") ) ;
    else
        X = readdlm( string(flpath, "cifar100_.csv") ) ;
    end
elseif sim == "mnist"
    a = (28, 28, 16) ;
    X = readdlm( string(flpath, "mnist_.csv") ) ;
elseif sim == "coco" || sim == "lfw"
    a = (64, 64, 8) ;
    if sim == "coco"
        X = readdlm( string(flpath, "iscoco_.csv") ) ;
    else
        X = readdlm( string(flpath, "lfw_.csv") ) ;
    end
else
    X = tnsrutils.tnsrinit(a) ;
end
X = reshape(X, a[1], a[2], a[3]) ;


if rsltn == 1
    latfact = 2 ;
    A, B, C = tnsrutils.cp3init(a, latfact) ;
    x0 = cputils.mat2vectcp3(A, B, C) ;
    results = optimize(objfuncp3bfgs, x0, NelderMead(), Optim.Options(iterations = maxitrtn)) ;
elseif rsltn == 2
# resolution of Paratuck2
    if sim == "cifar10" || sim == "cifar100"
        latfact = (8,10) ;
        A = readdlm( string(flpath, "cifar_paratuck_A.txt") ) ;
        H = readdlm( string(flpath, "cifar_paratuck_H.txt") ) ;
        B = readdlm( string(flpath, "cifar_paratuck_B.txt") ) ;
        DA = readdlm( string(flpath, "cifar_paratuck_DA.txt") ) ;
        DB = readdlm( string(flpath, "cifar_paratuck_DB.txt") ) ;
    elseif sim == "mnist"
        latfact = (8,10) ;
        A = readdlm( string(flpath, "mnist_paratuck_A.txt") ) ;
        H = readdlm( string(flpath, "mnist_paratuck_H.txt") ) ;
        B = readdlm( string(flpath, "mnist_paratuck_B.txt") ) ;
        DA = readdlm( string(flpath, "mnist_paratuck_DA.txt") ) ;
        DB = readdlm( string(flpath, "mnist_paratuck_DB.txt") ) ;
    elseif sim == "coco" || sim == "lfw"
        latfact = (8,12) ;
        A = readdlm( string(flpath, "coco_lfw_paratuck_A.txt") ) ;
        H = readdlm( string(flpath, "coco_lfw_paratuck_H.txt") ) ;
        B = readdlm( string(flpath, "coco_lfw_paratuck_B.txt") ) ;
        DA = readdlm( string(flpath, "coco_lfw_paratuck_DA.txt") ) ;
        DB = readdlm( string(flpath, "coco_lfw_paratuck_DB.txt") ) ;
    else
        latfact = (2,3) ;
        A, DA, H, DB, B = tnsrutils.paratuck2init(a, latfact) ;
    end
    DA = reshape(DA, latfact[1], latfact[1], a[3]) ;
    DB = reshape(DB, latfact[2], latfact[2], a[3]) ;
    x0 = paratuckutils.mat2vectParatuck2( A,DA,H,DB,B ) ;
    xf = paratuckutils.mat2vectParatuck2( A,DA,H,DB,B ) ;
    k = 0 ;
    fk = zeros( maxitrtn+1 ) ;
    tmstamp = zeros( maxitrtn+1 ) ;
    fk[k+1] = f( xf ) ;
    tmstamp[k+1] = time() ;
    println( "f(xk) at initialization: ", round(fk[k+1], 3) ) ;
    for n = 1:maxitrtn
        results = optimize(f, g!, xf,
            BFGS(), Optim.Options(iterations = 1) ) ;
        xf = Optim.minimizer(results) ;
        k = k + 1 ;
        fk[k+1] = f( xf ) ;
        tmstamp[k+1] = time() ;
        if n%(maxitrtn/10) == 0
            print( "   >> ", n/maxitrtn*100, "%" ) ;
            print( "     f(xk) = ", round(fk[k+1], 3) ) ;
            println( "" ) ;
        end
    end
    writedlm(string(filepath, string(maxitrtn), "xk.txt"), xf, ",") ;
    writedlm(string(filepath, string(maxitrtn), "fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(maxitrtn), "tmstamp.txt"), tmstamp, ",") ;
    println(results) ;
    xf = Optim.minimizer(results) ;
    println( fk[end] ) ;
elseif rsltn == 3
    # resolution of dedicom
    if sim == "cifar10" || sim == "cifar100"
        latfact = 10 ;
        A = readdlm( string(flpath, "cifar_dedicom_A.txt") ) ;
        D = readdlm( string(flpath, "cifar_dedicom_D.txt") ) ;
        H = readdlm( string(flpath, "cifar_dedicom_H.txt") ) ;
    elseif sim == "mnist"
        latfact = 10 ;
        A = readdlm( string(flpath, "mnist_dedicom_A.txt") ) ;
        D = readdlm( string(flpath, "mnist_dedicom_D.txt") ) ;
        H = readdlm( string(flpath, "mnist_dedicom_H.txt") ) ;
    elseif sim == "coco" || sim == "lfw"
        latfact = 11 ;
        A = readdlm( string(flpath, "coco_lfw_dedicom_A.txt") ) ;
        D = readdlm( string(flpath, "coco_lfw_dedicom_D.txt") ) ;
        H = readdlm( string(flpath, "coco_lfw_dedicom_H.txt") ) ;
    else
        latfact = 5 ;
        A, D, H = tnsrutils.dedicominit(a, latfact) ;
    end
    D = reshape(D, latfact, latfact, a[3]) ;
    x0 = dedicomutils.mat2vectDedicom( A,D,H ) ;
    xf = dedicomutils.mat2vectDedicom( A,D,H ) ;
    k = 0 ;
    fk = zeros( maxitrtn+1 ) ;
    tmstamp = zeros( maxitrtn+1 ) ;
    fk[k+1] = f( xf ) ;
    tmstamp[k+1] = time() ;
    println( "f(xk) at initialization: ", round(fk[k+1], 3) ) ;
    for n = 1:maxitrtn
        results = optimize(f, g!, xf,
            BFGS(), Optim.Options(iterations = 1) ) ;
        xf = Optim.minimizer(results) ;
        k = k + 1 ;
        fk[k+1] = f( xf ) ;
        tmstamp[k+1] = time() ;
        if n%(maxitrtn/10) == 0
            print( "   >> ", n/maxitrtn*100, "%" ) ;
            print( "     f(xk) = ", round(fk[k+1], 3) ) ;
            println( "" ) ;
        end
    end
    writedlm(string(filepath, string(maxitrtn), "xk.txt"), xf, ",") ;
    writedlm(string(filepath, string(maxitrtn), "fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(maxitrtn), "tmstamp.txt"), tmstamp, ",") ;
    println(results) ;
    xf = Optim.minimizer(results) ;
    println( fk[end] ) ;
elseif rsltn == 4
# resolution of CP
    if sim == "cifar10" || sim == "cifar100"
        latfact = 12 ;
        A = readdlm( string(flpath, "cifar_cp_A.txt") ) ;
        B = readdlm( string(flpath, "cifar_cp_B.txt") ) ;
        C = readdlm( string(flpath, "cifar_cp_C.txt") ) ;
    elseif sim == "mnist"
        latfact = 12 ;
        A = readdlm( string(flpath, "mnist_cp_A.txt") ) ;
        B = readdlm( string(flpath, "mnist_cp_B.txt") ) ;
        C = readdlm( string(flpath, "mnist_cp_C.txt") ) ;
    elseif sim == "coco" || sim == "lfw"
        latfact = 16 ;
        A = readdlm( string(flpath, "coco_lfw_cp_A.txt") ) ;
        B = readdlm( string(flpath, "coco_lfw_cp_B.txt") ) ;
        C = readdlm( string(flpath, "coco_lfw_cp_C.txt") ) ;
    else
        latfact = 2 ;
        A, B, C = tnsrutils.cp3init(a, latfact) ;
    end
    x0 = cputils.mat2vectcp3( A, B, C ) ;
    xf = cputils.mat2vectcp3( A, B, C ) ;
    k = 0 ;
    fk = zeros( maxitrtn+1 ) ;
    tmstamp = zeros( maxitrtn+1 ) ;
    fk[k+1] = f( xf ) ;
    tmstamp[k+1] = time() ;
    println( "f(xk) at initialization: ", round(fk[k+1], 3) ) ;
    for n = 1:maxitrtn
        results = optimize(f, g!, xf,
            BFGS(), Optim.Options(iterations = 1) ) ;
        xf = Optim.minimizer(results) ;
        k = k + 1 ;
        fk[k+1] = f( xf ) ;
        tmstamp[k+1] = time() ;
        if n%(maxitrtn/10) == 0
            print( "   >> ", n/maxitrtn*100, "%" ) ;
            print( "     f(xk) = ", round(fk[k+1], 3) ) ;
            println( "" ) ;
        end
    end
    writedlm(string(filepath, string(maxitrtn), "xk.txt"), xf, ",") ;
    writedlm(string(filepath, string(maxitrtn), "fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(maxitrtn), "tmstamp.txt"), tmstamp, ",") ;
    println(results) ;
    xf = Optim.minimizer(results) ;
    println( fk[end] ) ;
    #A, B, C = cputils.vect2matcp3(xf, a, latfact) ;
    #display(tnsrutils.buildCP3(A, B, C)) ;
end
