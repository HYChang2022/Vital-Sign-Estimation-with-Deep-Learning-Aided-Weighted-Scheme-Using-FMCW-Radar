function [omegaList, gainList, residueList] = f_extractSpectrum(y, S,...
			      	   tau, overSamplingRate, R_s , R_c)
% SUMMARY:
% 
%   given measurements: y = S * (mixture of sinusoids) + white noise
%          and a parameter tau which relates reduction in residue to
%          sparsity of the explanation (number of sinusoids), 
%          this module returns two **ordered** lists: a list of 
%          estimates of frequencies of the sinusoids in the mixture 
%          and and another list of corresponding gains 
% INPUT:
%    y - measurements
%    S - measurement matrix: can be compressive, or
%        just the identity matrix
%    tau - algorithm parameter which determines what the minimum payoff
%          we expect per sinusoid
%        - should be slightly more than the noise level sigma2
%        We solve for gains and continuous frequencies which minimize:
%        minimize norm(y - sum of sinusoids)^2 + tau * ell0_norm(gain)
%        ** LARGE VALUES OF TAU PROMOTE SPARSITY **% 
%    overSamplingRate (optional) -  used to determine how finely 
%              we sample frequency grid on which we precompute
%              responses when measurements are compressive (using 
%              IFFTs) how fine (in terms of multiples of the FFT 
%              grid) do we want the coarse grid to be?
%              number of grid points = overSamplingRate * N
%              Default value is 4
% 
%    R_s (optional) : number of Newton steps for a Single frequency update
%                           Default value is 1
%    R_c (optional) : number of rounds of Cyclic Refinement for all of the
%                           already detected sinusoids
%                           Default value is 3
% 
% OUTPUT:
%   omegaAns    - frequencies
%   gainAns     - gains of estimated frequencies
%   residueList - trajectory of the energy in the residual
%                 measurements as we add new sinsoids

% setting default values to 'overSamplingRate','R_s','R_c'
if ~exist('overSamplingRate','var'), overSamplingRate = 4;
elseif isempty(overSamplingRate), overSamplingRate = 4; end

if ~exist('R_s','var'), R_s = 1;
elseif isempty(R_s),    R_s = 1; end

if ~exist('R_c','var'), R_c = 3;
elseif isempty(R_c),    R_c = 3; end

% Algorithm preprocessing
%   compute overSamplingRate*M IFFTs once when measurement matrix
%   is not equal to eye(N); avoids repetitive IFFT computations 
%   later on
sampledManifold = preProcessMeasMat(S, overSamplingRate);

if sampledManifold.is_eye
	S = []; % erase identity matrix
end

omegaList = [];
gainList  = [];
y_r = y;

residueList = [ y_r' * y_r ];

while true
    
    % keep detecting new sinusoids until power in residue 
    % becomes small; *** how small *** determined by *** tau ***
    
    % detect gain and frequency of an additional sinusoid
    [omega_new, gain_new, y_r, res_inf_normSq_rot] = ...
        detectNew(y_r, sampledManifold);
    % detecttNew removes the contribution of the newly detected
    % from the old residue  y_r(input) and reports the new residual
    % measurement y_r (output)
    
    % stopping criterion:
    if res_inf_normSq_rot < tau
        break;
    end
    
    % newly detected sinusoid is coarse - so we refine it to 
    % imitate detection on the continuum
    for i = 1:R_s
        [omega_new, gain_new, y_r] = refineOne(y_r, omega_new, ...
            gain_new, S, sampledManifold.ant_idx, true);
    end
    % refineOne checks whether the refinement step decreases
    % the l-2 norm of the residue y_r
    
    % Add newly detected sinusoid to the ordered lists
    omegaList = [omegaList; omega_new];
    gainList  = [gainList; gain_new];
    
    % refine all frequencies detected so far
    % can be interpreted as a search for better frequency supports
    [omegaList, gainList, y_r] = refineAll(y_r, omegaList,...
        gainList, S, sampledManifold.ant_idx, R_s, R_c);
    omegaList;
    % refineAll only uses refineOne to tweak parameters and the energy 
    % in the residual measurements y_r can only decrease as a result

    % Solve least squares for the dictionary set [Ax(omega)] omega in 
    % omegaList
    [omegaList, gainList, y_r] = solveLeastSquares(y , omegaList, ...
        S, sampledManifold.ant_idx);    
    omegaList;
    % ensures that for the support we have settled on our choice of 
    % gains is optimal (in terms giving us the lowest residual energy)
    % though the practical value of the least-squares step is debatable
    % when the frequency support is well-conditioned (since least-squares
    % is a by-product of refineAll once the frequencies have converged)
    % we need this step for theoretical guarantees on convergence rates
 
    residue_new = y_r'*y_r;
    residueList = [residueList; residue_new];
    
end

% revert to standard notion of sinusoid: 
%           exp(1j*(0:(N-1))'*omega)/sqrt(N)
gainList = gainList .* exp(1j*sampledManifold.ant_idx(1)*omegaList);
omegaList = wrap_2pi(omegaList);

end

% -----------------------------------------------------------------

function [omega, gain, y_r, res_inf_normSq_rot] = detectNew(y,...
					 sampledManifold)
% SUMMARY:
% 
% 	detects a new sinusoid on the coarse grid
% 
% INPUT:
% 	y - measurements
% 	sampledManifold - when measurements are compressive,
% 		  we precompute (using IFFT operation) a
% 		  **dictionary** of responses for sinusoids 
%  		  corresponding to frequencies on a coarse grid 
% 		  and store them in the MATLAB **structure** 
% 		  sampledManifold
%
% OUTPUT:
% 	omega - frequency on [0,2*pi) which best explains
% 			the measurements y
% 	gain  - corresponding complex gain
% 	y_r   - after removing the detected sinusoid from the
%         	measurements y, y_r is the ***residual measurement***
%          res_inf_normSq_rot - max energy among DFT directions - needed
%          for stopping criterion

R = length(sampledManifold.coarseOmega);
N = sampledManifold.length;
OSR = round(R/N);

if sampledManifold.is_eye
    gains  = fft(y, R)/sqrt(N);
    
    if sampledManifold.ant_idx(1)~=0
        gains = gains.*exp(-1j*sampledManifold.coarseOmega(:)*...
            sampledManifold.ant_idx(1));
    end
    prob = (abs(gains).^2);
else
    energy = sampledManifold.map_IfftMat_norm_sq.';
    gains = (sampledManifold.map_IfftMat'*y)./energy;
    prob = (abs(gains).^2).*energy;
end

[~,IDX] = max(prob);

omega = sampledManifold.coarseOmega(IDX);
gain = gains(IDX);

% compute the response corresponding to the
% current estimate of the sinusoid
if sampledManifold.is_eye
    x = exp(1j*sampledManifold.ant_idx * omega)...
        /sqrt(sampledManifold.length);
else
    x = sampledManifold.map_IfftMat(:,IDX);
end

% residual measurements after subtracting
% out the newly detected sinusoid
y_r = y - gain * x;

% For stopping criterion
% we check only DFT frequencies - gives us a handle
% the false alarm rate (a measure of how often we would 
% over estimate the size of the support set)

res_inf_normSq_rot = max(prob(1:OSR:end));

end

% --------------------------------------------------------------------

function omega_prime= wrap_2pi(omega)
% SUMMARY: Restricts frequencies to [0, 2*pi)
% INPUT: A vector of spatial frequencies
% OUTPUT: A vector of coresponding spatial frequencies in [0, 2*pi)

omega_prime = angle(exp(1j*omega));
omega_prime(omega_prime < 0) = omega_prime(omega_prime < 0) + 2*pi;

end

% --------------------------------------------------------------------

function [omega, gain, y_r] = refineOne(y_r, omega, gain, S,...
			 ant_idx, isOrth)
% SUMMARY:
%   Refines parameters (gain and frequency) of a single sinusoid
%   and updates the residual measurement vector y_r to reflect
%   the refinement -- This function applies one Newton step only.
% INPUT:
% 	y_r - residual measurement (all detected sinusoids removed)
%	omega - current estimate of frequency of sinusoid we want to
% 			refine
%	gain - current estimate of gain of sinusoid we want to refine
% 	S - measurement matrix
%           if [], measurements are direct i.e S = eye(N),
% 	    where N is length of sinusoid	
% 	ant_idx - translating indexes to phases in definition of sinusoid
%	isOrth - binary flag - is y_r orthogonal to x(omega) 
%	       - default - false
% OUTPUT:
%       refined versions of omega, gain and y_r
%       (see INPUT for definitions)

if ~exist('isOrth', 'var'), isOrth = false; end;

if isempty(S)
    is_eye = true;
else
    is_eye = false;
end

N = length(ant_idx);
x_theta  = exp(1j*ant_idx*omega)/sqrt(N);
dx_theta = 1j * ant_idx .* x_theta;
d2x_theta = -(ant_idx.^2) .* x_theta;

if ~is_eye
    x_theta   = S * x_theta;
    dx_theta  = S * dx_theta;
    d2x_theta = S * d2x_theta;
end

% add the current estimate of the sinusoid to residue
y = y_r + gain*x_theta;


% UPDATE GAIN
% recompute gain and residue to ensure that 
% y_r is orthogonal to x_theta - this property
% is lost when we refine other sinusoids
if ~isOrth
    if is_eye
        energy = 1;
    else
        energy = x_theta'*x_theta;
    end
    gain = (x_theta'*y)/energy;
    y_r = y - gain*x_theta;
end

der1 = -2*real(gain * y_r'*dx_theta);
der2 = -2*real(gain * y_r'*d2x_theta) +...
    2*abs(gain)^2*(dx_theta'*dx_theta);

% UPDATE OMEGA
if der2 > 0
    omega_next = omega - der1/der2;
else
    omega_next = omega - sign(der1)*(1/4)*(2*pi/N)*rand(1);
end

% COMPUTE x_theta for omega_next so that we can compute 
% gains_next and y_r_next
x_theta  = exp(1j*ant_idx*omega_next)/sqrt(N);
if is_eye
    energy = 1;
else
    x_theta = S * x_theta;
    energy = (x_theta'*x_theta);
end

% UPDATE GAIN
gain_next = (x_theta'*y)/energy;

% UPDATE RESIDUE
y_r_next = y - gain_next*x_theta;

% check for decrease in residue -  needed as a result of 
% non-convexity of residue (even when the cost surface 
% is locally convex); This is the same as checking whether 
% |<y, x_theta>|^2/<x_theta,x_theta> improves as we ensured 
% that y - gain*x_theta is perp to x_theta by recomputing gain

if (y_r_next'*y_r_next) <= (y_r'*y_r)
    % commit only if the residue decreases
    omega = omega_next;
    gain = gain_next;
    y_r = y_r_next;
end

end

% --------------------------------------------------------------------

function [omegaList, gainList, y_r] = solveLeastSquares(y , omegaList, ...
    S, ant_idx)
% SUMMARY:
%    Reestimates all gains wholesale
N = length(ant_idx);
if isempty(S) % is this an identity matrix 
    A = exp(1j*ant_idx*omegaList.')/sqrt(N);
else
    A = S * exp(1j*ant_idx*omegaList.')/sqrt(N);
end   

% update gains
gainList = (A'*A)\(A'*y);
% update residues
y_r = y - A*gainList;

% energy in the residual measurement y_r is guaranteed to not increase 
% as a result of this operation. Therefore, we refrain from checking 
% whether the energy in y_r has increased
end

% --------------------------------------------------------------------

function [omegaList, gainList, y_r] = refineAll(y_r, omegaList,...
    gainList, S, ant_idx, R_s, R_c)
% SUMMARY:
%   uses refineOne algorithm to refine frequencies and gains of
%   of all sinusoids
% INPUT:
%    y_r - residual measurement after all detected sinusoids have been
%          removed
%    omegaList - list of frequencies of known(detected) sinusoids
%    gainList  - list of gains of known(detected) sinusoids
%    S - measurement matrix
%        if [], measurements are direct i.e S = eye(N), where N is
%       length of sinusoid	
%    ant_idx - translating indexes to phases in definition of sinusoid
%    R_s - number of times each sinusoid in the list is refined
%    R_c - number of cycles of refinement fot all of frequencies that have
%       been estimated till now
%
% OUTPUT:
%       refined versions of inputs omegaList, gainList, y_r

K = length(omegaList); % number of sinusoids


% Total rounds of cyclic refinement is "R_c"
for i = 1:R_c
    
    % chose an ordering for refinement
    
    % % *random* ordering
    % order = randperm(K);
    
    % *sequential* ordering
    order = 1:K;
    
    for j = 1:K
        l = order(j);
        
        
        
        % parameters of the l-th sinusoid
        omega = omegaList(l);
        gain = gainList(l);
            
        % refinement repeated "R_s" times per sinusoid
        for kk = 1:R_s

            % refine our current estimates of (gain, omega) of the
            % l-th sinusoid
            [omega, gain, y_r] = refineOne(y_r,...
                       omega, gain, S, ant_idx, false);
                   
        end
        
        omegaList(l) = omega;
        gainList(l) = gain;
            % refineOne ensures that (gain, omega) pair are altered iff
            % the energy in the residual measurements y_r does not 
            % increase
        
    end
    
end

end

% ----------------------------------------------------------------

function sampledManifold = preProcessMeasMat(S, overSamplingRate)
% SUMMARY:
%   compute overSamplingRate*M IFFTs once when measurement matrix
%   is not equal to eye(N); avoids repetitive IFFT computations 
%   later on
% INPUT:
%   S - M times N measurement matrix
%       M - number of measurements
%       N - length of sinusoid
%   examples: eye(N) - normal measurement matrix
%             diag(hamming(N)) - normal measurements with hamming
%                                weights
%             randn(M,N)/sqrt(N) - compressive measurement matrix
%             randn(M,N)/sqrt(N) * diag(hamming(N)) - compressive
%                       measurements with hamming weights matrix
%    overSamplingRate (optional) - how fine (in terms of multiples
%       of the FFT grid) do we want the coarse grid to be?
%       number of grid points = overSamplingRate * N
%       Default value is 3
%
% OUTPUT:
%       data structure with sinusoid responses 
%       precomputed using IFFTs when S is not equal to eye(N)


M = size(S,1);
N = size(S,2);

sampledManifold.length = N;
R = round(overSamplingRate*N);

sampledManifold.coarseOmega = 2*pi*(0:(R-1))/R;  % omegaCoarse

% definition of a sinusoid: 
%              exp(-1j*omega((N-1)/2):((N-1)/2))/sqrt(N)

ant_idx = 0:(N-1);
ant_idx = ant_idx - (N-1)/2;

% IFFT definition of a sinusoid(omega) takes the following form:
% 	sinusoid    = @(omega) exp(1j*(0:(N-1)).'*omega);
% To reiterate, we assume that a sinusoid is given by
%	sinusoid    = @(omega) exp(1j*ant_idx.'*omega)/sqrt(N);
% So we store this information in sampledManifold container
sampledManifold.ant_idx = ant_idx(:);

% Check if the measurement matrix is the identity matrix
if M == N
    is_eye = norm(eye(N) - S, 'fro') == 0;
else
    is_eye = false;
end
sampledManifold.is_eye = is_eye;

% WHEN THE MEASUREMENT MATRIX NOT THE IDENTITY MATRIX
% we compute the Fourier transforms of sensing weights 
% and store them for use in the coarse stage (can use 
% IFFT to speed up these steps)

% IN THE FOLLOWING COMPUTATIONS WE COMPENSATE FOR THE 
% DIFFERENCES BETWEEN NOTION OF A SINUSOID THAT WE TAKE 
% AND THAT TAKEN BY "IFFT"

if ~is_eye
    
    % S times x(omegaCoarse)
    sampledManifold.map_IfftMat = R/sqrt(N)*ifft(S,R,2) * ...
        sparse(1:R, 1:R, exp(1j * sampledManifold.coarseOmega...
        * ant_idx(1)), R, R);
    
    % norm square of S times x(omegaCoarse)
    % energy as a function of frequency
    sampledManifold.map_IfftMat_norm_sq = ...
        sum(abs(sampledManifold.map_IfftMat).^2, 1);
    
end
end