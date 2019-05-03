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

function nonlinearcg(X, x0, latfact;
    f = rosnbrck.ros, gf = rosnbrck.gradros,
    maxiter=100, xtol=1.0E-2, batch=[], epoch=1)
"""
nonlinearcg performs non-linear conjugate gradient descent algorithm.
It requires an objective function and a gradient function.

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
# nonlinearcg variable initialization
    x_i = zeros( maxiter + 2, length(x0) ) ;
    grad_i = zeros( maxiter + 2, length(x0) ) ;
    d_i = zeros( maxiter + 2, length(x0) ) ;

    fk = zeros( maxiter + 2 ) ;
    tmstamp = zeros( maxiter + 2 ) ;
    xk = x0 ;
    hcalls = 0 ;
    gcalls = 0;
    k = 0 ;
    cg_maxiter = 20*length(x0) ;

    x_i[1, :] = x0 ;
    xk = x0  ;
    gcalls += 1 ;

    if f == rosnbrck.ros
        curf = f( xk ) ;
    else
        curf = f( X, xk, latfact ) ;
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
        gfk0 = gf( X, xk, latfact, f=f, batchind=stochind ) ;
        pk = -gfk0 ;
        maggrad = sum( abs.( gfk0 ) ) ;
    end
#
# first optimization step
# =======================
    alphak = linsrch.strongwolfelinesearch(f, gf, pk, xk,
                X=X, latfact=latfact, stochind=stochind) ;
    x_i[k + 2, :] = x_i[k + 1, :] + alphak .* pk ;
    grad_i[k + 1, :] = gfk0 ;
    d_i[k + 1, :] = pk ;
    k += 1 ;
#
# gradient minimization loop
# ==========================
    println( "f(xk) at initialization: ", round(curf, 3) ) ;
    while (sum( maggrad ) > xtol) && (k < maxiter)
        if length(batch) != 0
            stochind = sample( indx, batch, replace=false ) ;
        end
#
# inner loop for stochastic gradient descent
        for n = 1:epoch

            xk = x_i[k + 1, :] ;
            xj = x_i[k, :] ;
            gfj = grad_i[k, :] ;
            dj = d_i[k, :] ;

            if f == rosnbrck.ros
                curf = f( xk ) ;
                b = - gf( xk ) ;
            else
                curf = f( X, xk, latfact ) ;
                b = - gf( X, xk, latfact, f=f, batchind=stochind ) ;
                gfk = -b ;
                gcalls += 1 ;
            end
            fk[k+1] = curf ;
            tmstamp[k+1] = time() ;
            maggrad = sum( abs.( b ) )  ;
#
# compute the gradient descent
            yk = gfk - gfj + 1.0e-7 ;
            beta = vecdot(gfk, yk) ;
            beta = beta ./ vecdot(dj, yk) ;

            dk = -gfk + beta .* dj ;
            pk = dk ;
            grad_i[k + 1, :] = gfk ;
            d_i[k + 1, :] = dk ;

            alphak = linsrch.strongwolfelinesearch(f, gf, pk, xk,
                        X=X, latfact=latfact, stochind=stochind) ;
            xk = xk + alphak .* pk ;
            x_i[k + 2, :] = xk ;

            k += 1 ;
#
# save calculation evolution
            if k % (maxiter / 10) == 0 || k == maxiter
                print( "   >> ", k / maxiter * 100, "%" ) ;
                println( "     f(xk) = ", round(curf, 3),
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
# Last card of function nonlinearcg.
#
end


rsltn = 4 ;
a = (4,5,6) ;
maxitrtn = 250 ;
batchnmb = 50 ;
epochnmb = 1 ;
flpath = ""
simall = ["cifar10", "cifar100", "mnist", "coco", "lfw"] ;
sim = "cifar10" ;

for isim = 1:length(simall)
    sim = simall[isim] ;

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
        else
            latfact = (2,3) ;
            A, DA, H, DB, B = tnsrutils.paratuck2init(a, latfact) ;
        end
        DA = reshape(DA, latfact[1], latfact[1], a[3]) ;
        DB = reshape(DB, latfact[2], latfact[2], a[3]) ;
        x0 = paratuckutils.mat2vectParatuck2( A,DA,H,DB,B ) ;
        xf = nonlinearcg(X, x0, latfact, f = paratuckutils.objfunParatuck2,
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
        xf = nonlinearcg(X, x0, latfact, f = dedicomutils.objfunDedicom,
        gf = utilsappderparfor.approxJ, maxiter=maxitrtn) ;
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
        x0 = cputils.mat2vectcp3(A, B, C);
        xf = nonlinearcg(X, x0, latfact, f = cputils.objfuncp3,
        gf = utilsappderparfor.approxJ, maxiter=maxitrtn) ;
        # assess the quality of the results
        A, B, C = cputils.vect2matcp3(xf, a, latfact) ;
        # display(tnsrutils.buildCP3(A, B, C)) ;
        # print("\n") ;
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

end
