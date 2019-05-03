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

function gradescent(X, x0, latfact;
    f = rosnbrck.ros, gf = rosnbrck.gradros,
    maxiter=100, xtol=1.0E-2, batch=[], epoch=1)
"""
gradescent performs Gradient Descent algorithm. It requires an objective
function and a gradient function.

# Arguments
- `X::Array`: tensor to decompose.
- `x0::Array`: initial guess of the tensor decomposition solution.
- `latfact::Array`: latent factors involved in the decomposition.
- `f::function`: objective function to solve.
- `gf::function`: gradient of the objective function.
- `maxiter::Integer`: maximum number of iterations.
- `xtol::float`: gradient tolerance.
- `batch::Integer`: number of elements for stochastic gradient descent.
- `epoch::Integer`: number of gradient descent loop.
...
"""
    filepath = "" ;

    # scase = 1 >> standard gradient descent (GD)
    # scase = 2 >> Nesterov accelerated gradient descent (NAG)
    # scase = 3 >> Adam gradient descent
    # scase = 4 >> RMSProp gradient descent
    # scase = 5 >> SAGA gradient descent
    # scase = 6 >> AdaGrad gradient descent
    scase = 4 ;
    if scase == 1
        # GD
        eta = 0.0001 ;
    elseif scase == 2
        # NAG
        gm = 0.9 ;
        eta = 0.0001 ;
        v = zeros( maxiter+1, length(x0) ) ;
    elseif scase == 3
        # Adam
        bta1 = 0.9 ;
        bta2 = 0.999 ;
        epsadam = 1.0E-8 ;
        eta = 0.001 ;
        v = zeros( maxiter+1, length(x0) ) ;
        m = zeros( maxiter+1, length(x0) ) ;
        vhat = zeros( maxiter+1, length(x0) ) ;
        mhat = zeros( maxiter+1, length(x0) ) ;
    elseif scase == 4
        # RMSProp
        gamma = 0.9 ;
        eta = 0.001 ;
        epsilon = 0.0001 ;
        prevb = zeros( maxiter+1, length(x0) ) ;
    elseif scase == 5
        # SAGA
        eta = 0.0001 ;
        grd = zeros( maxiter+1, length(x0) ) ;
    elseif scase == 6
        # AdaGrad
        eta = 0.01 ;
        epsilon = 0.000001 ;
        G = zeros( length(x0) ) ;
    end
#
    println( "/"^35, "\n\tNEW EXECUTION\n", "/"^35 ) ;
#
# calculation information at end of minimization
    statusmessage = Dict(1 => "Optimization terminated successfully.\n",
                      2 => "Maximum number of function evaluations has been exceeded.\n",
                      3 => "Maximum number of iterations has been exceeded.\n",
                      4 => "Desired error not necessarily achieved due to precision loss.\n") ;
#
# detect non nul entries in x0 for CP sparse resolution
    indx = findn(x0) ;
#
# Newtong CG variable initialization
    fk = zeros( maxiter ) ;
    tmstamp = zeros( maxiter) ;
    xk = x0 ;
    hcalls = 0 ;
    k = 0 ;
    cg_maxiter = 20*length(x0) ;
    if f == rosnbrck.ros
        curf = f( xk ) ;
    else
        curf = f( X, xk, latfact );
    end
    old_fval = curf ;
#
# information for stochastic descent
    batchshp = length( xk ) ;
    if length(batch) == 0
        print("\nStandard Descent For Numerical Resolution\n") ;
        stochind = 1:length(xk) ;
    else
        #stochind = rand( 1:batchshp, batch ) ;
        stochind = sample( indx, batch, replace=false ) ;
    end
    if f == rosnbrck.ros
        maggrad = sum( abs.( gf( xk ) ) )  ;
    else
        maggrad = sum( abs.( gf( X, xk, latfact, f=f, batchind=stochind ) ) ) ;
    end
#
# gradient minimization loop
# ========================
    println( "f(xk) at initialization: ", round(curf, 3) ) ;
    while (sum( maggrad ) > xtol) && (k < maxiter)
        if length(batch) != 0
            #print("\n-> new stoch. descent\n") ;
            #stochind = rand( 1:batchshp, batch ) ;
            stochind = sample( indx, batch, replace=false ) ;
        end
#
# inner loop for stochastic gradient descent
        for n = 1:epoch
            if f == rosnbrck.ros
                curf = f( xk ) ;
                b = - gf( xk ) ;
            else
                curf = f( X, xk, latfact ) ;
                b = - gf( X, xk, latfact, f=f, batchind=stochind ) ;
            end
            fk[k+1] = curf ;
            tmstamp[k+1] = time() ;
            maggrad = sum( abs.( b ) )  ;
#
# compute the gradient descent
            if scase == 1
                # GD
                #alphak = linsrch.backtrackinglinesearch(f, gf, b, xk,
                #        X=X, latfact=latfact, stochind=stochind) ;
                #alphak = linsrch.strongwolfelinesearch(f, gf, b, xk,
                #            X=X, latfact=latfact, stochind=stochind) ;
                #xk = xk - alphak .* (-b) ;
                xk = xk - eta .* (-b) ;
            elseif scase == 2
                # NAG
                xk = xk - v[k+1] ;
                v[k+2, :] = gm .* v[k+1, :] +
                  eta .* gf( X, xk-gm*v[k+1, :], latfact, f=f, batchind=stochind ) ;
            elseif scase == 3
                # ADAM
                mhat[k+1, :] = m[k+1, :] / ( 1 - bta1 ) ;
                vhat[k+1, :] = v[k+1, :] / ( 1 - bta2 ) ;
                xk = xk - ( eta ./ ( sqrt.(vhat[k+1, :]) .+ epsadam) ) .* mhat[k+1, :] ;
                m[k+2, :] = bta1 .* m[k+1, :] + ( 1 - bta1 ) .* (-b) ;
                v[k+2, :] = bta2 .* v[k+1, :] + ( 1- bta2 ) .* b.^2 ;
            elseif scase == 4
                # RMSProp
                # alphak = linsrch.strongwolfelinesearch(f, gf, b, xk,
                #             X=X, latfact=latfact, stochind=stochind) ;
                if k == 0
                    meansquare = (1 - gamma) .* (b.^2) ;
                else
                    meansquare = gamma .* prevb[k, :] + (1 - gamma) .* (b.^2) ;
                end
                prevb[k+1, :] = meansquare ;
                xk = xk - (eta ./ (sqrt.(meansquare + epsilon))) .* (-b) ;
            elseif scase == 5
                # SAGA
                grd[k+1, :] = -b ;
                if k == 0
                    xk = xk - eta .* (-b) ;
                else
                    #alphak = linsrch.strongwolfelinesearch(f, gf, b, xk,
                    #            X=X, latfact=latfact, stochind=stochind) ;
                    xk = xk - eta .* ( (-b) - grd[k, :] .+ mean( grd[k, :] ) ) ;
                end
            elseif scase == 6
                # AdaGrad
                G = G + (b.^2) + epsilon;
                xk = xk - (eta ./ (sqrt.(G + epsilon))) .* (-b) ;
            end
            k += 1 ;
#
# save calculation evolution
            if k%(maxiter/10) == 0 || k == maxiter
                print( "   >> ", k/maxiter*100, "%" ) ;
                print( "     f(xk) = ", round(curf, 3),
                      " | norm[grad(f)] = ", round(maggrad, 3) );
                println( "" ) ;
                writedlm(string(filepath, string(k), "xk.txt"), xk, ",") ;
                writedlm(string(filepath, string(k), "fk.txt"), fk, ",") ;
                writedlm(string(filepath, string(k), "tmstamp.txt"), tmstamp, ",") ;
            end
        end
    end
#
# end of execution
    if k == maxiter
        print( statusmessage[3] ) ;
    else
        print( statusmessage[1] ) ;
        if f == rosnbrck.ros
            print("\tCurrent function value: ", f( xk ), "\n" ) ;
        elseif f == paratuckutils.objfunParatuck2
            print("\tCurrent function value: ", f( X, xk, latfact ), "\n" ) ;
        end
        print("\tIterations: ", k, "\n" ) ;
        print("\tHessian evaluations: ", hcalls, "\n" ) ;
        print("\tx_final: \n", xk ) ;
    end
    return xk ;
#
# Last card of function gradescent.
#
end


rsltn = 2 ;
a = (3,3,5) ;
maxitrtn = 10 ;
batchnmb = 50 ;
epochnmb = 1 ;
sim = "bus" ;
flpath = ""

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
elseif sim == "bus"
    a = (80, 80, 3) ;
    X = readdlm( string(flpath, "bus_matrix.csv") ) ;
else
    X = tnsrutils.tnsrinit(a) ;
end
X = reshape(X, a[1], a[2], a[3]) ;


if rsltn == 1
# resolution of rosnbrck functionj
    a = (3,4,2) ;
    latfact = (5,7) ;
    X = tnsrutils.tnsrinit(a) ;
    x0 = [2, 3] ;
    xf = gradescent(X, x0, latfact) ;
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
    elseif sim == "bus"
        latfact = (8,12) ;
        A = readdlm( string(flpath, "bus_paratuck_A.txt") ) ;
        H = readdlm( string(flpath, "bus_paratuck_H.txt") ) ;
        B = readdlm( string(flpath, "bus_paratuck_B.txt") ) ;
        DA = readdlm( string(flpath, "bus_paratuck_DA.txt") ) ;
        DB = readdlm( string(flpath, "bus_paratuck_DB.txt") ) ;
    else
        latfact = (2,3) ;
        A, DA, H, DB, B = tnsrutils.paratuck2init(a, latfact) ;
    end
    DA = reshape(DA, latfact[1], latfact[1], a[3]) ;
    DB = reshape(DB, latfact[2], latfact[2], a[3]) ;
    x0 = paratuckutils.mat2vectParatuck2( A,DA,H,DB,B ) ;
    # line to be used only if calculation interrupted too early
    # x0 = readdlm( string(flpath, "cifar10_par_nag_xk_3000.txt") ) ;
    xf = gradescent(X, x0, latfact, f = paratuckutils.objfunParatuck2,
                gf = utilsappderparfor.approxJ, maxiter=maxitrtn) ;
    println(a) ;
#
# assess the quality of the results
    A, DA, H, DB, B = paratuckutils.vect2matParatuck2(xf, a, latfact) ;
    #display(tnsrutils.buildparatuck2(A, DA, H, DB, B')) ;
    #print("\n") ;
elseif rsltn == 3
# resolution of Dedicom
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
    xf = gradescent(X, x0, latfact, f = dedicomutils.objfunDedicom,
                gf = utilsappderparfor.approxJ, maxiter=maxitrtn, batch=batchnmb, epoch=epochnmb) ;
#
# assess the quality of the results
    A, D, H = dedicomutils.vect2matDedicom(xf, a, latfact) ;
    # display(tnsrutils.buildDedicom(A, D, H)) ;
    # print("\n") ;
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
    x0 = cputils.mat2vectcp3(A, B, C) ;
    xf = gradescent(X, x0, latfact, f = cputils.objfuncp3,
            gf = utilsappderparfor.approxJ, maxiter=maxitrtn) ;
# assess the quality of the results
    #A, B, C = cputils.vect2matcp3(xf, a, latfact) ;
    #display(tnsrutils.buildCP3(A, B, C)) ;
    #print("\n") ;
elseif rsltn == 5
# resolution of Tucker
    latfact = (3,4,2) ;
    A, B, C, G = tnsrutils.tuckerinit(a, latfact) ;
    x0 = tuckerutils.mat2vecttucker(A, B, C, G);
    xf = gradescent(X, x0, latfact, f = tuckerutils.objfuntucker,
                gf = utilsappderparfor.approxJ, maxiter=maxitrtn) ;
#
# assess the quality of the results
    A, B, C, G = tuckerutils.vect2mattucker(xf, a, latfact) ;
    display(tnsrutils.buildtucker(A, B, C, G)) ;
    print("\n") ;
end

#Profile.clear()
#@profile gradescent(X, x0, latfact, f = paratuckutils.objfunParatuck2,
#            gf = utilsappder.approxJ, hf = utilsappder.approxHess,
#            maxiter=maxitrtn)
#Juno.profiletree()
#Juno.profiler()
