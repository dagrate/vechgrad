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

rsltn = 2 ;
a = (60,60,3) ;
latfact = (30,30) ;
sim = "bus" ;
flpath = ""
rsltnall = ["als", "sgd", "nag", "adam", "rmsprop", "saga", "adagrad", "ncg", "bfgs", "vechgrad"] ;
#rsltnall = ["sgd", "nag", "adam", "rmsprop", "saga", "adagrad"] ;

# manage the bus simulation results
#for rsltn = 1:length(rsltnall)
rsltn = 9
    xf = readdlm( string(flpath, "bus_par_", rsltnall[rsltn],"_xk.txt") ) ;
    #xf = readdlm( string(flpath, "2000xk.txt") ) ;
    A, DA, H, DB, B = paratuckutils.vect2matParatuck2(xf, a, latfact) ;
    Xh = tnsrutils.buildparatuck2(A, DA, H, DB, B') ;
    writedlm(string(flpath, "bus_Xh_", rsltnall[rsltn], ".txt"), vcat(Xh...)) ;
#end
