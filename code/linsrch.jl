@everywhere module linsrch
using paratuckutils, rosnbrck
export backtrackinglinesearch, strongwolfelinesearch

function backtrackinglinesearch(f, gf, p, xk;
    rho=0.5, c=1.0e-4, X=[], latfact=[], stochind=[])
# Backtracking line search
#
# Inputs:
#       problem     function (cost/grad/hess)
#       p           search direction
#       x           current iterate
#       rho         backtrack step between (0,1), e.g., 1/2
#       c           parameter between 0 and 1, e.g., 1e^-4
# Output:
#       alpha       step size calculated by this algorithm
#
# Reference:
#       Jorge Nocedal and Stephen Wright,
#       "Numerical optimization,"
#       Springer Science & Business Media, 2006.
#
#       Algorithm 3.1 in Section 3.1.
#
# Created by J. Charlier on 20 Feb. 18
    alpha = 1 ;
    if f == rosnbrck.ros
        f0 = f( xk ) ;
        g0 = gf( xk ) ;
    else
        f0 = f( X, xk, latfact ) ;
        g0 = - gf( X, xk, latfact, f=f, batchind=stochind )
    end
    x0 = xk;
    x = xk + alpha .* p ;
    if f == rosnbrck.ros
        fk = f( x ) ;
    else
        fk = f( X, x, latfact ) ;
    end
#
# repeat until the Armijo condition meets
    while fk > f0 + c * alpha * vecdot(g0, p)
        alpha = rho * alpha ;
        x = x0 + alpha .* p ;
        if f == rosnbrck.ros
            fk = f( x ) ;
        else
            fk = f( X, x, latfact ) ;
        end
    end
    return alpha ;
#
# Last card of function backtrackinglinesearch.
#
end

function strongwolfelinesearch(f, gf, d, x0;
    c1=1.0e-4, c2=0.9, X=[], latfact=[], stochind=[])
#function [alphas,fs,gs] = strong_wolfe_line_search(problem, d, x0, c1, c2)
#function [alphas,fs,gs] = strongwolfe(myFx,d,x0,fx0,gx0)
# Function strongwolfe performs Line search satisfying strong Wolfe conditions
# Input
#   myFx:   the optimized function handle
#   d:      the direction we want to search
#   x0:     vector of initial start
#   fx0:    the function value at x0
#   gx0:    the gradient value at x0
# Output
#   alphas: step size
#   fs:     the function value at x0+alphas*d
#   gs:     the gradient value at x0+alphas*d
#
# Notice
#   I use f and g to save caculation. This funcion strongwolfe is called by LBFGS_opt.m.
# Ref
#   Numerical Optimization, by Nocedal and Wright
# Author:
#   Guipeng Li @THU
#   guipenglee@gmail.com
#
# Modified by H.Kasai on 1 Nov. 16
# Modified by J. Charlier on 20 Feb. 18
    if f == rosnbrck.ros
        fx0 = f( x0 ) ;
        gx0 = gf( x0 ) ;
    else
        fx0 = f( X, x0, latfact ) ;
        gx0 = - gf( X, x0, latfact, f=f, batchind=stochind ) ;
    end
    maxIter = 10 ;
    alpham = 20 ;
    alphap = 0 ;
    alphax = 1;
    gx0 = vecdot(gx0, d) ;
    fxp = fx0 ;
    gxp = gx0 ;
    i = 1 ;
#
# Line search algorithm satisfying strong Wolfe conditions
# Algorithms 3.2 on page 59 in Numerical Optimization, by Nocedal and Wright
# alphap is alpha_{i-1}
# alphax is alpha_i
# alphas is what we want.
    cnt = 1;
    while cnt == 1
        xx = x0 + alphax .* d ;
        if f == rosnbrck.ros
            fxx = f( xx ) ;
            gxx = gf( xx ) ;
        else
            fxx = f( X, xx, latfact ) ;
            gxx = - gf( X, xx, latfact, f=f, batchind=stochind )
        end
        fs = fxx ;
        gs = gxx ;
        gxx = vecdot(gxx, d) ;
        if (fxx > fx0 + c1 * alphax .* gx0) || ((i > 1) && (fxx >= fxp))
            if f == rosnbrck.ros
                alphas = Zoom(f, gf, x0, d, alphap, alphax, fx0, gx0) ;
            else
                alphas = Zoom(f, gf, x0, d, alphap, alphax, fx0, gx0,
                                X=X, latfact=latfact, stochind=stochind) ;
            end
            cnt = 0 ;
            return alphas;
        end
        if abs.(gxx) <= -c2 * gx0
            alphas = alphax ;
            cnt = 0 ;
            return alphas ;
        end
        if gxx >= 0
            if f == rosnbrck.ros
                alphas = Zoom(f, gf, x0, d, alphax, alphap, fx0, gx0);
            else
                alphas = Zoom(f, gf, x0, d, alphax, alphap, fx0, gx0,
                                X=X, latfact=latfact, stochind=stochind);
            end
            cnt = 0 ;
            return alphas;
        end
        alphap = alphax ;
        fxp = fxx ;
        gxp = gxx ;
        if i > maxIter
          alphas = alphax ;
          cnt = 0 ;
          return alphas ;
        end
        r = 0.8 ;
        alphax = alphax + ( alpham - alphax ) * r ;
        i += 1 ;
    end
    return alphas ;
#
# Last card of function strongwolfelinesearch.
#
end

function Zoom(f, gf, x0, d, alphal, alphah, fx0, gx0;
    X=[], latfact=[], stochind=[])
#function [alphas,fs,gs] = Zoom(problem,x0,d,alphal,alphah,fx0,gx0)
# Algorithms 3.2 on page 59 in
# Numerical Optimization, by Nocedal and Wright
    # This function is called by strongwolfe
    c1 = 1e-4 ;
    c2 = 0.9 ;
    i = 0 ;
    maxIter = 10;
    cnt = 1 ;
    while cnt == 1
        # bisection
        alphax = 0.5 * ( alphal + alphah ) ;
        alphas = alphax ;
        xx = x0 + alphax .* d ;
        if f == rosnbrck.ros
            fxx = f( xx ) ;
            gxx = gf( xx ) ;
        else
            fxx = f( X, xx, latfact ) ;
            gxx = - gf( X, xx, latfact, f=f, batchind=stochind )
        end
        fs = fxx ;
        gs = gxx ;
        gxx = vecdot(gxx, d) ;
        xl = x0 + alphal .* d;
        if f == rosnbrck.ros
            fxl = f( xl );
        else
            fxl = f( X, xl, latfact ) ;
        end
        if ((fxx > fx0 + c1 * alphax .* gx0) || (fxx >= fxl))
            alphah = alphax ;
        else
            if abs.(gxx) <= -c2 * gx0
                alphas = alphax ;
                cnt = 0 ;
                return alphas;
            end
            if gxx .* ( alphah - alphal ) >= 0
                alphah = alphal;
            end
            alphal = alphax;
        end
        i += 1 ;
        if i > maxIter
            alphas = alphax;
            cnt = 0 ;
            return alphas ;
        end
    end
    return alphas ;
#
# Last card of function Zoom
#
end
#
# Last card of module linsrch.
#
end
