@everywhere module utilsappderparfor
using paratuckutils, rosnbrck
export approxJ, approxHess

#function testpmap(x, y)
#    jac = zeros( length(x) ) ;
#    for n = 1:length(x)
#        jac[n] = x[n] + y[n] ;
#    end
#    return jac ;
#end

function approxJ(X, xk, latfact;
    f=paratuckutils.objfunParatuck2, order=4, epsilon=1.0E-4, batchind=[])
#
# even if batchind is not used in the function, we let it as input for
# compatibility issue with the serial execution
#
    dx = SharedArray{Float64}( length(xk), length(xk) ) ;
    jac = SharedArray{Float64}( length(xk) ) ;
    @parallel for i = 1:length(xk)
        #dx[i, :] = 0 ;
        dx[i, i] = epsilon ;
        #jac[i] = 0 ;
    end
#
# gradient computation
    @sync @parallel for n = 1:length(xk)
        #println(n) ;
        if order == 2
            eps2 = 1 / (2*epsilon) ;
            if f == rosnbrck.ros
                fact0 = f( xk + dx[n,:] ) ;
                fact1 = f( xk - dx[n,:] ) ;
            else
                fact0 = f( X, xk + dx[n,:], latfact ) ;
                fact1 = f( X, xk - dx[n,:], latfact ) ;
            end
            jac[n] = ( fact0 - fact1 ) * eps2 ;
        else
            eps2 = 1 / (24*epsilon) ;
            if f == rosnbrck.ros
                fact0 = f( xk-2*dx[n,:] ) ;
                fact1 = f( xk-dx[n,:] ) ;
                fact2 = f( xk+dx[n,:] ) ;
                fact3 = f( xk+2*dx[n,:] ) ;
            else
                fact0 = f( X, xk-2*dx[n,:], latfact ) ;
                fact1 = f( X, xk-dx[n,:], latfact ) ;
                fact2 = f( X, xk+dx[n,:], latfact ) ;
                fact3 = f( X, xk+2*dx[n,:], latfact ) ;
            end
            jac[n] = ( 2*fact0 - 16*fact1 + 16*fact2 - 2*fact3 ) * eps2 ;
        end
    end
    return jac ;
#
# Last card of function approxJ.
#
end

function approxHess(X, xk, latfact;
    f=paratuckutils.objfunParatuck2, batchind=[])
    #println("Warning: Hessian not implemented for multiprocessing. Calculation on single process") ;
#
# define constants
    df = approxJ ;
    order = 4 ;
    epsilon = 1.0E-4 ;
#
    if length(batchind) == 0
        batchshp = length(xk) ;
        batchind = 1:length(xk) ;
    else
        batchshp = length(batchind) ;
    end
    #hes = zeros( length(xk), length(xk) ) ;
    dx1 = SharedArray{Float64}( length(xk), length(xk), length(xk) ) ;
    dx2 = SharedArray{Float64}( length(xk), length(xk), length(xk) ) ;
    dx3 = SharedArray{Float64}( length(xk), length(xk), length(xk) ) ;
    dx4 = SharedArray{Float64}( length(xk), length(xk), length(xk) ) ;
    hes = SharedArray{Float64}( length(xk), length(xk) ) ;
    # we fill the first epsilon representing the first derivative
    for lk = 1:length(xk)
        for lj = 1:length(xk)
            dx1[lj, lk, lk] = epsilon ;
            dx2[lj, lk, lk] = epsilon ;
            dx3[lj, lk, lk] = -epsilon ;
            dx4[lj, lk, lk] = -epsilon ;
        end
    end
    # we fill the second epsilon representing the second derivative
    for lk = 1:length(xk)
        for lj = 1:length(xk)
            dx1[lj, lj, lk] += epsilon ;
            dx2[lj, lj, lk] += -epsilon ;
            dx3[lj, lj, lk] += epsilon ;
            dx4[lj, lj, lk] += -epsilon ;
        end
    end
#
# Hessian computation
    dx = zeros( length(xk) ) ;
    if order == 2
        eps2 = 1 / ( 2 * epsilon ) ;
        for n = 1:batchshp
            e = batchind[n] ;
            dx[e] = epsilon ;
            fact1 = df( X, xk + dx, latfact ) ;
            fact2 = df( X, xk - dx, latfact ) ;
            hes[e, :] = (fact1 - fact2) * eps2 ;
            dx[e] = 0.0 ;
        end
    elseif order == 4
        eps2 = 1 / ( epsilon^2 ) ;
        eps22 = 1 / ( 4 * ( epsilon^2 ) ) ;
#
# computation of the upper Hessian matrix
        @sync @parallel for n = 1:length(xk)
            for m = 1:length(xk)
                if f == rosnbrck.ros
                    fact1 = f( xk + dx1[m, :, n] ) ;
                    fact2 = f( xk + dx2[m, :, n] ) ;
                    fact3 = f( xk + dx3[m, :, n] ) ;
                    fact4 = f( xk + dx4[m, :, n] ) ;
                else
                    fact1 = f( X, xk + dx1[m, :, n], latfact ) ;
                    fact2 = f( X, xk + dx2[m, :, n], latfact ) ;
                    fact3 = f( X, xk + dx3[m, :, n], latfact ) ;
                    fact4 = f( X, xk + dx4[m, :, n], latfact ) ;
                end
                hes[n, m] = ( eps22*( fact1 - fact2 - fact3 + fact4 ) ) ;
            end
        end
    end
#
# copy upper Hessian matrix to lower triangle
# it is faster to compute all sotch ind than copying upper Hessian matrix
#    hes = Symmetric(hes) ;
    return hes
#
# Last card of function approxHess.
#
end
#
# Last card of module paratuckutils.
#
end
