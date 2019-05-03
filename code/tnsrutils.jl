@everywhere module tnsrutils
export tnsrinit, tnsrinitbis, krao, errorfun, objfun, tnsrunfold
export buildCP3, paratuck2init, buildparatuck2

#
function tnsrinit(tnsrSize)
#
# compute a standard tensor for backtesting operations
#
    cur = 0;
    X = zeros(tnsrSize);
    if length(tnsrSize) == 3
        for ktr = 1:tnsrSize[3]
            for itr = 1:tnsrSize[1]
                for jtr = 1:tnsrSize[2]
                    cur = cur + 1;
                    X[itr, jtr, ktr] = cur;
                end
            end
        end
    elseif length(tnsrSize) == 4
        for itr = 1:tnsrSize[1]
            for jtr = 1:tnsrSize[2]
                for ktr = 1:tnsrSize[3]
                    for ltr = 1:tnsrSize[4]
                        cur = cur + 1;
                        X[itr, jtr, ktr, ltr] = cur;
                    end
                end
            end
        end
    else
        display("5-way tensor initialization not implemented");
    end
    return X;
#
#     Last card of function tensorinit.
#
end

function tnsrinitbis(tnsrSize)
#
# compute a standard tensor for backtesting operations
# as described by Kolda in Tensor Decompositions and Applications
#
    cur = 0;
    X = zeros(tnsrSize);
    for ktr = 1:tnsrSize[3]
            for jtr = 1:tnsrSize[2]
                for itr = 1:tnsrSize[1]
                cur = cur + 1 ;
                X[itr, jtr, ktr] = cur ;
            end
        end
    end
    return X;
#
#     Last card of function tensorinitbis.
#
end

function krao(matC,matD)
    (I,F) = size(matC) ;
    (J,F1) = size(matD) ;
#
    kraomat = zeros(I*J, F) ;
    for f = 1:F
        kraom = matD[:,f] * matC[:,f].' ;
        kraomat[:,f] = kraom[:];
    end
    return kraomat ;
#
# Last card of function krao
#
end

function errorfun(X, Xh)
    return (X-Xh) .* (X-Xh) ;
#
# Last card of function errorfunct.
#
end

function objfun(X, Xh)
    Z = errorfun(X,Xh) ;
    return sqrt( sum(Z) ) ;
#
# Last card of function objfun
#
end

function tnsrunfold(X, mode)
    dim = collect(size(X)) ;
    ncol = Int(cumprod(dim)[end]/dim[mode]) ;
    if mode == 1
        Z = reshape(permutedims(X, [1,2,3]), dim[mode], ncol) ;
    elseif mode == 2
        Z = reshape(permutedims(X, [2,1,3]), dim[mode], ncol) ;
    elseif mode == 3
        Z = reshape(permutedims(X, [3,1,2]), dim[mode], ncol) ;
    end
    return Z ;
#
# Last card of function tnsrunfold.
#
end

function nmodeproduct(X,U,mode)
    dim = collect(size(X)) ;
    dim[mode] = size(U)[1] ;
    Y = U *tnsrutils.tnsrunfold(X,mode) ;
    return reshape(Y, (dim[1],dim[2],dim[3])) ;
#
# Last card of function nmodeproduct.
#
end

function cp3init(a, latfact)
    R = latfact ;
    A = rand(a[1], R) ;
    B = rand(a[2], R) ;
    C = rand(a[3], R) ;
    return A, B, C ;
#
# Last card of function cp3init.
#
end

function buildCP3(A, B, C)
    I = size(A)[1] ;
    J = size(B)[1] ;
    K = size(C)[1] ;
    Xh = zeros(I,J,K) ;
#
# manage the case where R == 1
    if length( size( A ) ) == 1
        tmp = kron( C, kron( A, B' ) ) ;
        irow = 1 ;
        for lk = 1:K
            Xh[:,:,lk] += tmp[irow:irow+I-1, :] ;
            irow += I ;
        end
    else
        R = size(A)[2] ;
        for li = 1:I
        	for lj = 1:J
        		for lk = 1:K
        			for lr = 1:R
        				Xh[li,lj,lk] += A[li, lr] * B[lj, lr] * C[lk, lr] ;
        			end
        		end
        	end
        end
        #kron is highly inefficient for medium to large tensors
        #for lr = 1:R
        #    tmp = kron( C[:,lr], kron( A[:,lr], B[:,lr]' ) ) ;
        #    irow = 1 ;
        #    for lk = 1:K
        #        Xh[:,:,lk] += tmp[irow:irow+I-1, :] ;
        #        irow += I ;
        #    end
        #end
    end
    return Xh ;
#
# Last card of function buildCP3.
#
end

function paratuck2init(a, latfact)
    P = latfact[1] ;
    Q = latfact[2]
#
    A = rand(a[1], P) ;
    DA = zeros(P, P, a[3]) ;
    for n = 1:a[3]
        DA[:,:,n] = eye(P, P) ;
    end
    H = rand(P, Q) ;
    DB = zeros(Q, Q, a[3]) ;
    for n = 1:a[3]
        DB[:,:,n] = eye(Q, Q) ;
    end
    B = rand(a[2], Q) ;
    return A, DA, H, DB, B ;
#
# Last card of function paratuck2init.
#
end

function buildparatuck2(A, DA, R, DB, BT)
    I = size(A)[1] ;
    J = size(BT)[2] ;
    K = size(DA)[3] ;
#
# paratuck2 computation
    Xh = zeros(I,J,K) ;
    for n = 1:K
        Xh[:,:,n] = A * DA[:,:,n] * R * DB[:,:,n] * BT ;
    end
    return Xh
#
# Last card of function buildparatuck2.
#
end

function dedicominit(a, latfact)
    P = latfact ;
    A = rand(a[1], P) ;
    D = zeros(P, P, a[3]) ;
    for n = 1:a[3]
        D[:,:,n] = eye(P, P) ;
    end
    H = rand(P, P) ;
    return A, D, H ;
#
# Last card of function dedicominit.
#
end

function buildDedicom(A, D, H)
    I = size(A)[1] ;
    K = size(D)[3] ;
    AT = A' ;
#
# dedicom computation
    Xh = zeros(I,I,K) ;
    for n = 1:K
        Xh[:,:,n] = A * D[:,:,n] * H * D[:,:,n] * AT ;
    end
    return Xh
#
# Last card of function buildDedicom.
#
end

function tuckerinit(a, latfact)
    P = latfact[1] ;
    Q = latfact[2] ;
    R = latfact[3] ;
    A = rand(a[1], P) ;
    B = rand(a[2], Q) ;
    C = rand(a[3], R) ;
#
# Tucker diagonal has to be stronger than other elements
    G = rand(P, Q, R)/10 ;
    for n = 1:R
        tmp = G[:,:,n] ;
        tmp[diagind(tmp)] = 1.0 ;
        G[:,:,n] = tmp ;
    end
    return A, B, C, G ;
#
# Last card of function tuckerinit.
#
end

function buildtucker(A, B, C, G)
    Xh = nmodeproduct(nmodeproduct(nmodeproduct(G,A,1), B, 2), C, 3) ;
    return Xh ;
#
# Last card of function buildtucker.
#
end
#
#   Last card of module tnsrutils.
#
end
