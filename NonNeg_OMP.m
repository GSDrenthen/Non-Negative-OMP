function [MWF] = NonNeg_OMP(A,y,T2Times,RI)
% Adaptations to the Canonical Implementation of Non-Negative Orthogonal
% Matching Pursuit to better suit the problem of Myelin Water Fraction
% estimation.
%
% Please cite the following paper when using this function:
% Drenthen GS, Backes WH, Aldenkamp AP, Op ’t Veld GJ, Jansen JFA. A new
% analysis approach for T2 relaxometry myelin water quantification: 
% Orthogonal Matching Pursuit. Magn Reson Med. 2019;81(5):3292-3303. 
% https://doi.org/10.1002/mrm.27600
%
% Original implementation by Mehrad Yaghoobi (http://www.mehrdadya.com/)
% Adaptations by Gerhard Drenthen

residual = zeros(RI,1);
for perm = 1:1:RI
    opt = optimset('Display','off','Algorithm','levenberg-marquardt');
    [m,n] = size(A);
    tmp = 1:1:n;
    weight_fun = exp(-1.*(0:2/RI:2-2/RI).^2);
    x = zeros(n,1);
    mag = 1;
    k = 1;
    xs = [];
    ind = [];
    r = y;
    bpr = A'*r;
    s = [randsample(tmp(T2Times<40e-3),1)';randsample(tmp(T2Times>40e-3),1)'];

    while (k <= n && mag > 0) 
        if k > 1
            bpr(s) = 0;
            [mag,ind] = max(bpr); 
        end
        if mag > 0
            s = [s;ind];
            As = A(:,s);
            xs = lsqnonneg(As,y,opt);
            if sum(abs(r)) < sum(abs(y - As*xs))
                break
            end      
            r = y - As*xs;
            bpr = A'*r;
        end        
        k = k+1;
    end
    k = k-1;
    x(s) = xs;
    residual(perm) = sum((y - A*x).^2);
    
    MWF(perm) = sum(x(T2Times < 40e-3)) / sum(x);
end

[~, ind] = sort(residual,'ascend');
MWF = sum(MWF(ind).*weight_fun ./ sum(weight_fun));
end
