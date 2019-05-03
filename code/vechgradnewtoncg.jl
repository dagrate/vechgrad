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

function newtonCG(X, x0, latfact;
    f = rosnbrck.ros, gf = rosnbrck.gradros, hf = rosnbrck.hessros,
    maxiter=100, xtol=1.0E-2, batch=[], epoch=1)
"""
newtonCG performs Newton Conjugate Gradient algorithm. It requires an objective
function, a gradient function and a Hessian function. The inverse of the
Hessian matrix is computed using the conjugate gradient.

# Arguments
- `X::Array`: tensor to decompose.
- `x0::Array`: initial guess of the tensor decomposition solution.
- `latfact::Array`: latent factors involved in the decomposition.
- `f::function`: objective function to solve.
- `gf::function`: gradient of the objective function.
- `hf::function`: hessian of the objective funtion.
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
# Newtong CG variable initialization
    fk = zeros( maxiter ) ;
    gradfk = zeros( maxiter ) ;
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
        stochind = sample( indx, batch, replace=false ) ;
    end
    if f == rosnbrck.ros
        maggrad = sum( abs.( gf( xk ) ) )  ;
    else
        maggrad = sum( abs.( gf( X, xk, latfact, f=f, batchind=stochind ) ) ) ;
    end
#
# Newton minimization loop
# ========================
    println( "f(xk) at initialization: ", round(curf, 3) ) ;
    while (sum( maggrad ) > xtol) && (k < maxiter)
        if length(batch) != 0
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
            gradfk[k+1] = sum( vecdot(b, b) ) ;
            tmstamp[k+1] = time() ;
            maggrad = sum( abs.( b ) )  ;
#
# Compute a search direction pk by applying the CG method to
#  del2 f(xk) p = - grad f(xk) starting from 0.
            eta = min( 0.5, sqrt(maggrad) ) ;
            termcond = eta .* maggrad ;
            xsupi = zeros( length( xk ) ) ;
            ri = -b ;
            psupi = -ri ;
            dri0 = vecdot(ri, ri) ;
            if f == rosnbrck.ros
                A = hf( xk ) ;
            end
            hcalls += 1 ;
#
# cg loop to avoid calculation of inverse Hessian
# ===============================================
            k2 = 0 ;
            curv = 1.0 ;
            float64eps = 2.2204460492503131e-16 ;
            i = 0 ;
            # Nocedal and Wright
            cg_maxiter = 30 ;
            perturbation = 1.0E-5 ;
            term2 = -b ; #gf( X, xk, latfact, f=f, batchind=stochind ) ;
            while (sum( abs.( ri ) ) > termcond) && (k2 < cg_maxiter)
                #Ap = A * psupi ;
                # Nocedal and Wright
                term1 = gf( X, xk+perturbation.*psupi, latfact, f=f, batchind=stochind ) ;
                Ap = ( term1 .- term2 ) ./ perturbation ;
                curv = vecdot(psupi, Ap) ;
                if 0 <= curv <= 3 * float64eps
                    break ;
                elseif curv < 0
                    if (i > 0)
                        break ;
                    else
                        # fall back to steepest descent direction
                        xsupi = dri0 / (-curv) * b ;
                        break ;
                    end
                end
                alphai = dri0 ./ curv ;
                xsupi = xsupi + alphai .* psupi ;
                ri = ri + alphai .* Ap ;
                dri1 = vecdot(ri, ri) ;
                betai = dri1 ./ dri0 ;
                psupi = -ri + betai .* psupi ;
                i += 1 ;
                dri0 = dri1 ;
                k2 += 1 ;
            end
#
# line search for minimization
            pk = xsupi ;
            gfk = -b ;
            if f == rosnbrck.ros
                alphak = linsrch.strongwolfelinesearch(f, gf, pk, xk) ;
            else
                #alphak = linsrch.backtrackinglinesearch(f, gf, pk, xk,
                #        X=X, latfact=latfact, stochind=stochind) ;
                alphak = linsrch.strongwolfelinesearch(f, gf, pk, xk,
                            X=X, latfact=latfact, stochind=stochind) ;
            end
            xk = xk + alphak .* pk ;
            k += 1 ;
#
# save calculation evolution
            if k % (maxiter / 10) == 0
                print( "   >> ", k/maxiter*100, "%" ) ;
                print( "     f(xk) = ", round(curf, 3),
                      " | norm[grad(f)] = ", round(maggrad, 3) );
                println( "" ) ;
                #println("\t-> results save") ;
                writedlm(string(filepath, string(k), "xk.txt"), xk, ",") ;
                writedlm(string(filepath, string(k), "fk.txt"), fk, ",") ;
                writedlm(string(filepath, string(k), "gradfk.txt"), gradfk, ",") ;
                writedlm(string(filepath, string(k), "tmstamp.txt"), tmstamp, ",") ;
            end
        end
    end
#
# end of execution
    writedlm(string(filepath, string(k), "xk.txt"), xk, ",") ;
    writedlm(string(filepath, string(k), "fk.txt"), fk, ",") ;
    writedlm(string(filepath, string(k), "gradfk.txt"), gradfk, ",") ;
    writedlm(string(filepath, string(k), "tmstamp.txt"), tmstamp, ",") ;
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
# Last card of function newtonCG.
#
end

rsltn = 4 ;
a = (2,2,2) ;
maxitrtn = 10 ;
batchnmb = 50 ;
epochnmb = 1 ;
sim = "test" ;


flpath = "" ;
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
# resolution of rosnbrck function
    a = (3,4,2) ;
    latfact = (5,7) ;
    X = tnsrutils.tnsrinit(a) ;
    x0 = [2, 3] ;
    xf = newtonCG(X, x0, latfact) ;
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
    tic() ;
    xf = newtonCG(X, x0, latfact, f = paratuckutils.objfunParatuck2,
                gf = utilsappderparfor.approxJ, hf = utilsappder.approxHess,
                maxiter=maxitrtn) ;
    toc() ;
#
# assess the quality of the results
    A, DA, H, DB, B = paratuckutils.vect2matParatuck2(xf, a, latfact) ;
    # display(tnsrutils.buildparatuck2(A, DA, H, DB, B')) ;
    # print("\n") ;
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
    xf = newtonCG(X, x0, latfact, f = dedicomutils.objfunDedicom,
                gf = utilsappderparfor.approxJ, hf = utilsappder.approxHess,
                maxiter=maxitrtn, batch=batchnmb, epoch=epochnmb) ;
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
    xf = newtonCG(X, x0, latfact, f = cputils.objfuncp3,
                gf = utilsappderparfor.approxJ, hf = utilsappder.approxHess,
                maxiter=maxitrtn) ;
#
# assess the quality of the results
    A, B, C = cputils.vect2matcp3(xf, a, latfact) ;
    # display(tnsrutils.buildCP3(A, B, C)) ;
    # print("\n") ;
elseif rsltn == 5
# resolution of Tucker
    latfact = (3,4,2) ;
    A, B, C, G = tnsrutils.tuckerinit(a, latfact) ;
    x0 = tuckerutils.mat2vecttucker(A, B, C, G);
    xf = newtonCG(X, x0, latfact, f = tuckerutils.objfuntucker,
                gf = utilsappderparfor.approxJ, hf = utilsappder.approxHess,
                maxiter=maxitrtn) ;
#
# assess the quality of the results
    A, B, C, G = tuckerutils.vect2mattucker(xf, a, latfact) ;
    display(tnsrutils.buildtucker(A, B, C, G)) ;
    print("\n") ;
end

#Profile.clear()
#@profile newtonCG(X, x0, latfact, f = paratuckutils.objfunParatuck2,
#            gf = utilsappder.approxJ, hf = utilsappder.approxHess,
#            maxiter=maxitrtn)
#Juno.profiletree()
#Juno.profiler()
