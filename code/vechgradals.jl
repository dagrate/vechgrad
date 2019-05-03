include("rosnbrck.jl") ;
include("tnsrutils.jl") ;
include("cputils.jl") ;
include("tuckerutils.jl")
include("dedicomutils.jl") ;
include("paratuckutils.jl") ;
include("utilsappder.jl") ;
include("utilsappderparfor.jl") ;
include("linsrch.jl") ;

include("alscp.jl") ;
include("alsdedicom.jl") ;
include("alsparatuck.jl") ;

import rosnbrck, tnsrutils, cputils, tuckerutils,
        dedicomutils, paratuckutils, utilsappder, utilsappderparfor, linsrch,
        alscp, alsdedicom, alsparatuck
using StatsBase
using Optim

rsltn = 2 ;
a = (4,4,6) ;
maxitrtn = 10000 ;
batchnmb = 50 ;
epochnmb = 1 ;
sim = "bus" ;
desc_bef = ["fk", "xk", "tmstamp"] ;
desc_aft = ["fk_", "xk_", "tm_"] ;
rs = ["_ros", "_par", "_ded", "_cp"] ;

flpath = "/home/jeremy/Documents/SnT/aa_code/app_vechgrad/julia/"
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
    a = (60, 60, 3) ;
    X = readdlm( string(flpath, "bus_matrix_tmp.csv") ) ;
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
        latfact = (50,50) ;
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
    println(a) ;
    A, DA, R, DB, B = alsparatuck.nnparatuck2als(X, latfact, A, H, B, DA, DB, maxitrtn) ;
    Xh = tnsrutils.buildparatuck2(A, DA, R, DB, B') ;
#
# assess the quality of the results
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
    A, D, H = alsdedicom.nndedicomals(X, latfact, A, D, H, maxitrtn) ;
#
# assess the quality of the results
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
    A, B, C = alscp.nncpals(X, latfact, A, B, C, maxitrtn) ;
# assess the quality of the results
    # display(tnsrutils.buildCP3(A, B, C)) ;
    # print("\n") ;
elseif rsltn == 5
# resolution of Tucker
    println("   >> Non-Negative Tucker decomposition:") ;
    println("   >> to do ...") ;
end

for n = 1:3
    path1 = "" ;
    nm1 = string(path1, string(maxitrtn), desc_bef[n], ".txt") ;
    nm2 = string(path1, sim, rs[rsltn], "_als_", desc_aft[n], string(maxitrtn), ".txt") ;
    mv(nm1, nm2) ;
end

#Profile.clear()
#@profile gradescent(X, x0, latfact, f = paratuckutils.objfunParatuck2,
#            gf = utilsappder.approxJ, hf = utilsappder.approxHess,
#            maxiter=maxitrtn)
#Juno.profiletree()
#Juno.profiler()
