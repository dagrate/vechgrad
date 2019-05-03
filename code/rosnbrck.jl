@everywhere module rosnbrck
export ros, gradros, hessros


function ros(x)
    #return x.^2 ;
    return (1 - x[1])^2 + 100*( (x[2]-(x[1]^2) ) ^2 ) ;
end


function gradros(x)
    results = zeros( length( x ) ) ;
    results[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - (x[1]^2) ) ;
    results[2] = 200 * (x[2] - (x[1]^2))
    return results
#
# Last card of function gradros.
#
end

function hessros(x)
    results = zeros( ( length( x ), length( x ) ) ) ;
    results[1, 1] = 1200 * x[1]^2 - 400 * x[2] + 2 ;
    results[1, 2] = -400 * x[1] ;
    results[2, 1] = -400 * x[1] ;
    results[2, 2] = 200 ;
    return results
#
# Last card of function hessros.
#
end
#
# Last card of module rosnbrck.
#
end
