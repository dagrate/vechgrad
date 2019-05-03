module utilsappder
using paratuckutils, rosnbrck
export approxJ, approxHess

function approxJ(X, xk, latfact;
    f=paratuckutils.objfunParatuck2, order=4, epsilon=1.0E-4, batchind=[])
    if length(batchind) == 0
        batchshp = length(xk) ;
        batchind = 1:length(xk) ;
    else
        batchshp = length(batchind) ;
    end
    jac = zeros( length(xk) ) ;
    dx = zeros( length(xk) ) ;
#
# gradient computation
    for n = 1:batchshp
        e = batchind[n] ;
        if order == 2
            eps2 = 1 / (2*epsilon) ;
            dx[e] = epsilon ;
            if f == rosnbrck.ros
                fact0 = f( xk + dx ) ;
                fact1 = f( xk - dx ) ;
            else
                fact0 = f( X, xk + dx, latfact ) ;
                fact1 = f( X, xk - dx, latfact ) ;
            end
            jac[e] = ( fact0 - fact1 ) * eps2 ;
            dx[e] = 0.0 ;
        else
            eps2 = 1 / (24*epsilon) ;
            dx[e] = epsilon ;
            if f == rosnbrck.ros
                fact0 = f( xk-2*dx ) ;
                fact1 = f( xk-dx ) ;
                fact2 = f( xk+dx ) ;
                fact3 = f( xk+2*dx ) ;
            else
                fact0 = f( X, xk-2*dx, latfact ) ;
                fact1 = f( X, xk-dx, latfact ) ;
                fact2 = f( X, xk+dx, latfact ) ;
                fact3 = f( X, xk+2*dx, latfact ) ;
            end
            jac[e] = ( 2*fact0 - 16*fact1 + 16*fact2 - 2*fact3 ) * eps2 ;
            dx[e] = 0.0 ;
        end
    end
    return jac ;
#
# Last card of function approxJ.
#
end

function approxHess(X, xk, latfact;
    f=paratuckutils.objfunParatuck2, batchind=[])
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
    hes = zeros( length(xk), length(xk) ) ;
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
        dx1 = zeros( length(xk) ) ;
        dx2 = zeros( length(xk) ) ;
        dx3 = zeros( length(xk) ) ;
        dx4 = zeros( length(xk) ) ;
#
# computation of the Hessian diagonal
        #for n = 1:batchshp
        #    e = batchind[n] ;
        #    dx[e] = epsilon ;
        #    if f == rosnbrck.ros
        #        fact1 = f( xk + dx ) ;
        #        fact2 = f( xk ) ;
        #        fact3 = f( xk - dx ) ;
        #    else
        #        fact1 = f( X, xk + dx, latfact ) ;
        #        fact2 = f( X, xk, latfact ) ;
        #        fact3 = f( X, xk - dx, latfact ) ;
        #    end
        #    hes[e,e] = eps2 * ( fact1 - 2 * fact2 + fact3 ) ;
        #    dx[e] = 0.0 ;
        #end
#
# computation of the upper Hessian matrix
        for n = 1:batchshp
            for m = 1:batchshp
                indn = batchind[n] ;
                indm = batchind[m] ;
                dx1[indn] += epsilon ;
                dx1[indm] += epsilon ;
                dx2[indn] += epsilon ;
                dx2[indm] += -epsilon ;
                dx3[indn] += -epsilon ;
                dx3[indm] += epsilon ;
                dx4[indn] += -epsilon ;
                dx4[indm] += -epsilon ;
                if f == rosnbrck.ros
                    fact1 = f( xk + dx1 ) ;
                    fact2 = f( xk + dx2 ) ;
                    fact3 = f( xk + dx3 ) ;
                    fact4 = f( xk + dx4 ) ;
                else
                    fact1 = f( X, xk + dx1, latfact ) ;
                    fact2 = f( X, xk + dx2, latfact ) ;
                    fact3 = f( X, xk + dx3, latfact ) ;
                    fact4 = f( X, xk + dx4, latfact ) ;
                end
                hes[indn, indm] = ( eps22*( fact1 - fact2 - fact3 + fact4 ) ) ;
                dx1[indn] = 0.0 ;
                dx1[indm] = 0.0 ;
                dx2[indn] = 0.0 ;
                dx2[indm] = 0.0 ;
                dx3[indn] = 0.0 ;
                dx3[indm] = 0.0 ;
                dx4[indn] = 0.0 ;
                dx4[indm] = 0.0 ;
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
