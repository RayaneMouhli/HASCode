 % Main calculation program for Hybrid Angular Spectrum HASgui8, which uses generalized ERFA for a phased%   array.  It is assumed the the gui has already loaded the ERFA_ file, which concurrently loads the%   pressure on the ERFA plane (perfa) and the transducer parameters freq (fMHz), radius of curvature (R), %	size of the ERFA plane (Len), radiated power (Pr), distance between the xducer and ERFA plane (sx),%   and location file of the elements (ElemLoc).  Generalized ERFA first combines the element%	responses with the correct phases for focusing, then propagates to the front plane of the Modl in the%	freq domain.  Interpolation matches the points on the propagated perfa with the points on the Modl.%   In addition, the paramHAS7_ file, which reads in the Modl sample spacings and media properties, and %   the Modl_ file of media integers have also been loaded, in that order.%%   This script is executed in the main workspace, so all variables there are accessable.% %   HASgui7 has warnings that display when the sample spacing in the propagated perfa pattern is greater than%	twice (later 1x) the sample spacing in the Modl (perhaps affecting the accuracy of interp2), and when%	the size of the perfa pattern is not large enough to cover the Modl front plane when it is offset from the%   xducer axis (although this is redundant with interp2 and is now commented out).%%  This version also restructures the HAS section to improve the algorithm (see Notes of 7/7/09 and%   7/14/09), including adding the backward (only) reflected wave, weighting M2 by the expected%   beam region, and implementing an optional low pass filter on the angular spectrum to cut off evanescent%   waves.%%   Changes:%   1/27/11 - Made most of the large arrays and matrices single precision to save memory (perfa was already%       single, and Modl was already int8 [later int16 for HU of bone]).%   12/28/12 - Version CalcHAS_ERFA6c: Added absorption [requiring aabs(i) coefficients] and scattering [both%       Approach B for scatterers smaller than a voxel, and statistical Approach C for scatterers on a scale  %       larger than several voxels, requiring correlation length corrl and randvc(i)]. Assumed that the %       pressure full att coeff a = aabs + scatt.  Q calculation now based on abs coeff aabs only. %       Also added maxpoutnorm to allow modification of display normalization (later dropped), and%       corrected waitbarorig calls.%   2/16/13 - Changed interp2 to phasor_interp2 to improve interpolation of complex erfa values.%   2/18/13 - Bypassed the random number generation in the scattering sections in cases where there%       is no scattering (either max(a - aabs)=0 and/or max(randvc)=0).%   4/11/13 - Changed warning on erfa spacing to be more stringent: erfa spacing should be <= 1x model spacing.%   4/23/13 - Dropped adj. maxpoutnorm and maxQnorm capability in HASgui. Now done with colormapeditor.%   4/24/13 - Changed the way that single precision is done in the memory preallocation section.%   5/19/15 - Changed call to displa7 (ARFI displacements wdisp added).  Also cleared wdisp for new calculation.%   5/3/16 - Reduced several model parameter files (e.g., attmodl) to 2D matrices to save memory.  Also corrected%       the one-layer-off mistake (Rn->Rn+1) in backward wave (see Working Notes2, 7/7/09, revised 4/29/16).%   7/20/16 - Major change: Now use full integration of excess propagation constant for propagation in the space%       domain rather than the approximation employed in earlier versions of CalcHAS. (Versions CalcHAS_ERFA7b %       and HASgui7b have the option to chose the approx or not.) The full integration is slighty slower due to a 'for'%       loop (depending upon number of media types) but is more accurate for models with small voxels %       (thus large alpha and beta) and/or largely variable speeds of sound or attenuation. %       (see Working Notes2, 7/7/09, revised 4/29/16). %       Also, statistical scattering by Approach C is now used only to vary the pprimeterm in the full integration%       step, not to vary each individual parameter (i.e., a, c and rho).%%   3/5/17 - This version does NOT have the "anti-leakage" sections used for microparticles.%          The waitbar is also made to be the MATLAB default.%   3/10/17 - Multiple reflections added. See the sheets entitled "Latest Diagram of Progression of Pressure in %       CalcHAS" 3/10/17 and "Stategy for Multiple Reflections" 3/10/17 to get an overview of the layout for multiple %       reflections. It uses the variable numrefl from the gui. To get ready for multiple reflections, the variable pforb%       was added, to be symmetrical with pbackb; pforb is also now used as the beam profile weighting of bprime.  %   5/5/17 - Optional plots of the beam power profiles at each loop added.%   1/12/2019 - Changed call in line 93 to SteeringPhasesPA8, which switches sign of h to -h to account for the%       opposite direction of x-axis in HAS (LHS) compared to the original development of the steering equations.%   5/15/2019 - Bypassed calculation of transferfa if emdist (distance to move front pattern) is zero.%   6/14/2019 - Added the capability of implementing phase correction based on: %       1. Checkbox7 (loads a 4D variable called PhCorr1 from a file obtained earlier with the script %            PhCorrMethod1_UsingCalcHAS.m) or%       2. Checkbox8 (calls script PhCorrMethod2_UsingTimeRev.m, which propagates beam back to transducer) or%       3. Checkbox9 (calls script PhCorrMethod3_UsingRayTrace.m, which finds phase along ray to each element.%       They all assume the intended focal location is given by the (possibly steered) focal location found in the gui %       and override the phases found for electronic steering alone.%   7/18/19 - Added tests when PhCorrMethod1 is implemented to make sure the configuration when PhCorr1 was%       calculated matches the current configuration, and checks whether the current focus is inside the ROI. The%       PhCorr1 array must be generated with the program 'PhCorrMethod1_UsingPreCalcHAS1' or later versions.%   04/20/2020, Michelle Kline, %       added ability to run this script from either HAS_NoGUI.m or from the interface.%       used variable usingGUI, which is either set in HAS_NoGUI.m, or initializegui7.m%   04/23/2020, Michelle Kline and D.A. Christensen%       made changes to better match experimental data (search for "DAC") %       rpave vs. rave, etc.%%	Copyright D.A. Christensen 2019.%		June 18, 2019% ----------------------------------------------% Michelle Kline, July 2020% added logic to ensure that proper variables/files are loaded for phase correction section% Modified by Michelle Kline, December 2020% Pared down Doug Christensen's CalcHAS_ERFA8e.m workspace script to create this calcHAS() function. % Used the version of CalcHAS_ERFA8e.m from the "calcHASchanges" branch of git repo. (Michelle Kline % and D.A. Christensen made changes to better match experimental data ((search for "DAC"))% rpave vs. rave, etc.% Also, now assumes that media properties (a, c, rho, etc.) are 3D matrices of the same size as the% Modl. This allows for continuous properties instead of a one-property-to-one-mediumrunAcousticTest('testModel_ones5x5x5',[2 2 2], 0, 0); correspondence.% Suggested:% Use runCalcHAS() to call calcHAS()%% Inputs:% 1) erfa parameters struct with the following fields:%    ElemLoc      element locations            %    Len          size of the ERFA plane                    %    R            transducer radius of curvature               %    dxp                         %    dyp                           %    fMHz         transducer frequency                 %    isPA         is this a phased array? (true/false)                %    perfa        pressure on the ERFA plane       %    pfilename    name of file containing ERFA parameters                %    relem        radius of circular element of array transducer, if applicable (m)            %    sx                %    Pr     		 acoustic power output of transducer%    erfafilenm   the ERFA file used to create the phases% 2) model parameters struct with the following fields:%    Modl         the model%    c0           speed of sound in water (m/s)%    c            matrix of speed of sound values (m/s)%    a            matrix of total attenuation values%    rho0         density of water%    rho          matrix of density values%    Dx           model resolution in 2nd dimension%    Dy           model resolution in 1st dimension%    Dz           model resolution in 3rd dimension%    modlfilenm   the model file used to create the phases%    randvc       random variation% 3) positioning parameterss struct with the following fields:%    offsetxmm    mechanical offset from center of Modl (along 2nd dimension) (mm)%    offsetymm    mechanical offset from center of Modl (along 1st dimension) (mm)%    dmm          distance from Xducer base to Modl base (mm)%    angpgvect    steering phases% 4) numrefl        number of reflections desired% Outputs:% 1) Q % 2) maxQ% 3) pout% 4) maxpout% MMK FIXME: scattering approach C: corrlfunction [Q, maxQ, pout, maxpout] = calcHAS(erfaParams, modelParams, positioningParams, numrefl)     % MMK not sure if necessary...    clear Aprime layxs1 layxs2 M2 pfor ptot transf Aprimeback Refl Z ind pback pref wdisp;            %% pre-initialization ==========================================================================    % define outputs (in case we have to exit early)    Q =[];    maxQ = 0;    pout = [];    maxpout = 0;        %% check input =================================================================================    if nargin ~= 4 || ~isstruct(erfaParams) || ~isstruct(modelParams) || ~isstruct(positioningParams)        disp('Usage: [Q, maxQ, pout, maxpout] = calcHAS(erfaParams, modelParams, positioningParams, numrefl)');        return;    end        %% argument validation =========================================================================    % if aabs field not present, use a    if ~isfield(modelParams,'aabs')        modelParams.aabs=modelParams.a;     end    % grab all of the fields in erfaParams and compare to list of expected fields    % ugly, but 'contains' not available pre-R2016b    expectedErfaFields = {'ElemLoc'; 'Len'; 'R'; 'dxp'; 'dyp'; 'fMHz'; 'isPA'; 'perfa';         'pfilename'; 'relem'; 'sx'; 'Pr'};    actualErfaFields = fieldnames(erfaParams);    for iterator = 1:length(expectedErfaFields)        index = strfind(actualErfaFields,expectedErfaFields{iterator});        if isempty(find(not(cellfun('isempty',index)),1))             disp(['Error: missing parameter: erfaParams.' expectedErfaFields{iterator}]);            return;        end    end    expectedModelFields = {'Modl'; 'c0'; 'c'; 'a'; 'rho0'; 'rho'; 'Dx'; 'Dy'; 'Dz';         'randvc'};    actualModelFields = fieldnames(modelParams);    for iterator = 1:length(expectedModelFields)        index = strfind(actualModelFields,expectedModelFields{iterator});        if isempty(find(not(cellfun('isempty',index)),1))            disp(['Error: missing parameter: modelParams.' expectedModelFields{iterator}]);            return;        end    end    expectedPositioningFields = {'offsetxmm'; 'offsetymm'; 'dmm'; 'angpgvect'};    actualPositioningFields = fieldnames(positioningParams);    for iterator = 1:length(expectedPositioningFields)        index = strfind(actualPositioningFields,expectedPositioningFields{iterator});        if isempty(find(not(cellfun('isempty',index)),1))             disp(['Error: missing parameter: positioningParams.' expectedPositioningFields{iterator}]);            return;        end    end     % check 3D property matrix dimensions     %   total attenuation    if size(modelParams.a) ~= size(modelParams.Modl)        disp('modelParams.a (attenuation) must be the same dimensions as modelParams.Modl');        return;    end    %   pressure absorption coefficient    if size(modelParams.aabs) ~= size(modelParams.Modl)        disp('modelParams.aabs must be the same dimensions as modelParams.Modl');        return;    end    %   speed of sound (no random variation in it now)    if size(modelParams.c) ~= size(modelParams.Modl)        disp('modelParams.c must be the same dimensions as modelParams.Modl');        return;    else        if min(min(min(modelParams.c)))==0            disp('Some speed of sound values (modelParams.c) are zero');            return;         end    end    % density    if size(modelParams.rho) ~= size(modelParams.Modl)        disp('modelParams.rho must be the same dimensions as modelParams.Modl');        return;    end    %% initialization ==============================================================================        % convert pressure absorption coefficient units from [Np/cm*MHz] to [Np/m], assume linear in f [in MHz].    absmodl=modelParams.aabs*1e2*erfaParams.fMHz;         % convert to Hz (MMK why not just input Hz?)    f=erfaParams.fMHz*1e6;	    % always use single precision perfa to save memory    if strcmp(class(erfaParams.perfa),'double')  %#ok<STISA>        erfaParams.perfa=single(erfaParams.perfa);     end     % size of ERFA plane    [lmaxerfa,mmaxerfa,~]=size(erfaParams.perfa);     % sample spacing in ERFA plane, in m    Dyerfa=(erfaParams.Len(1)/(lmaxerfa-1)) * 1000;     Dxerfa=(erfaParams.Len(2)/(mmaxerfa-1)) * 1000;     % setting up axes in mm units for interpolation    yaxiserfa=Dyerfa*(-(lmaxerfa-1)/2:(lmaxerfa-1)/2);        xaxiserfa=Dxerfa*(-(mmaxerfa-1)/2:(mmaxerfa-1)/2);     % adjust axes for offsets, all in mm    xaxiserfaoffs=xaxiserfa+positioningParams.offsetxmm;     yaxiserfaoffs=yaxiserfa+positioningParams.offsetymm;           % size of model    [lmax,mmax,nmax] = size(modelParams.Modl);  % The size of the model (lmax,mmax,nmax) sets the size of the simulation        % space.  lmax and y are vertical; mmax and x are horizontal. NOTE: Be aware of the (y,x,z) order of arrays        % assumed here.  Therefore lmax by mmax is also the y by x size of the pressure pattern pp on the front         % plane of the modelParams.Modl, interpolated from perfa. Note: lmax and mmax should be ODD numbers to keep         % fftshift symmetrical, with the dc term at exact center of spectrum.        % axes (between centers of end points) for imagesc and interp.    xaxisinterp=modelParams.Dx*(-(mmax-1)/2:(mmax-1)/2); % (mm)    yaxisinterp=modelParams.Dy*(-(lmax-1)/2:(lmax-1)/2); % (mm)       % longitudinal axis has full Dz at the center of the first voxel, since HAS calculates    % travel through a full distance Dz for each voxel and attributes the resulting pressure     % to that voxel.    lx=((mmax-1) * modelParams.Dx)/1000; % note conversion from mm to m    ly=((lmax-1) * modelParams.Dy)/1000;        if Dyerfa>modelParams.Dy || Dxerfa>modelParams.Dx        disp(['Warning: the sample spacing on the ERFA plane is more than that of the model. '...            'Should use a higher resolution ERFA file.']);    end        % convert to meter units (lowercase in meters);    dz=modelParams.Dz/1000;         % Preallocate memory of 3D variables used later inside slice 'for' loop for speed; also make single precision.    % These 3D variables used later in calculations or debugging, but some variables are 2D to save memory.      Z=zeros(size(modelParams.Modl),'single');       % acoustic  impedance; needed for ARFI and Q calculation (could recalc).    Refl=zeros(size(modelParams.Modl),'single');    % reflection coefficient; saved for backward wave calculation (could recalc).    sqrtexpon=zeros(size(modelParams.Modl),'single');  % sqrt(1-alpha_sqr-beta_sqr); used in transf, r and rp.    transf=zeros(size(modelParams.Modl),'single');  % transfer function; needed for backward wave propagation.    Aprime=zeros(size(modelParams.Modl),'single');    % angular spectra; used for debugging with pout=Aprime (could make matrix)    pfor=zeros(size(modelParams.Modl),'single');	% pressure array, forward propagation.    pfortot=zeros(size(modelParams.Modl),'single');	% initialize accumulated pressure array, forward propagation.    pref=zeros(size(modelParams.Modl),'single');        % back reflection    % pprime=zeros(size(modelParams.Modl),'single');      % after passage through thin space-domain layer.    pbacktot=zeros(size(modelParams.Modl),'single');       % initialize accumulated pressure array, back propagation.    bprime=zeros(1,nmax);   % vector of mean propagation coefficient.    TempA=zeros(size(modelParams.Modl),'single');    % angular spectra after propagation, to test for evanescent decay.        %% main calculation ============================================================================        tic % time the computation        % ----- This section handles the generalized ERFA calculations ----    if erfaParams.isPA  %  phased array?  If so, electronically steer or use phase correction angles.        % multiply perfa pages by steering phases or phase-correction phases and sum.        angarr=repmat(single(positioningParams.angpgvect),[lmaxerfa,mmaxerfa,1]);               serfa=sum(erfaParams.perfa.*angarr,3); % summed perfa (single page).        serfa=serfa*sqrt(erfaParams.Pr);   % adjust for total radiated power since perfa normalized to 1 W.        clear angarr % to free memory.    else        serfa=erfaParams.perfa*sqrt(erfaParams.Pr);   % solid transducer, so no need to consider phase.  Adjust power as above.    end  % end of 'is phased array' if statement.    clear erfaParams.perfa; % to free memory    hwb=waitbar(0,'INITIALIZING','Name','Initializing');         % Interpolate summed erfa onto smaller grid to match pp. Note conj to account for R-S.       % Custom interp code that more correctly interpolates complex matrices such as    % pressure waves than does interp2.  Uses normalized input matrix to find accurate phase angles.      % xaxiserfaoffs, xaxisinterp are row vectors; yaxiserfaoffs, yaxisinterp are col. vectors.    if min(serfa(:))==0        zia = interp2(xaxiserfaoffs,yaxiserfaoffs,serfa,xaxisinterp,yaxisinterp','*linear',0);    % to avoid 0/0    else        zia = interp2(xaxiserfaoffs,yaxiserfaoffs,serfa./abs(serfa),xaxisinterp,yaxisinterp','*linear',0);  % normalized serfa here only    end    za = angle(zia);  % in radians    zm = interp2(xaxiserfaoffs,yaxiserfaoffs,abs(serfa),xaxisinterp,yaxisinterp','*linear',0);    ppe = conj(zm.*exp(1i*za));     % distance to propagate from ERFA plane to front of modelParams.Modl; okay to be negative.    emdist=(positioningParams.dmm/1000)-erfaParams.sx;  % note conversion of dmm to meters    if emdist ~= 0 % only do translation to new distance if emdist is not zero:        % These next lines change the distance from the ERFA plane to front face of modelParams.Modl, depending on gui input.            % Note that mmax is the max x index, lx is the extent of model in x in meters, and bprime is the mean            % propagation constant.        ferfa=fftshift(fft2(ppe));   % into freq domain to allow propagation to a different distance between ERFA and model.        bprimeerfa=2*pi*f/modelParams.c0;  % region in front of modelParams.Modl is water, so mean bprime = omega/modelParams.c0.        alpha=((1:mmax)-ceil(mmax/2))*(2*pi/(bprimeerfa*lx));   % vector of alpha in water (so use bprimeerfa).        beta=((1:lmax)-ceil(lmax/2))*(2*pi/(bprimeerfa*ly));         % vector of beta in water.        [alpha_sq,  beta_sq] = meshgrid(alpha.^2, beta.^2);   % lmax by mmax matrices for water transfer function.        expon=1 - alpha_sq - beta_sq;   % inside sqrt part of exponent in transfer function below. When expon neg,               % evanescent waves will decay for positive emdist, but will blow up in the backward direction for negative              % emdist. So the next lines filter out those evanescent waves. (Note: Since the direction cosines alpha =              % fx times lambda and beta = fy times lambda, alpha and beta can be > 1 for high spatial frequencies caused              % by spatial details that are < lambda. Then 1 - alpha_sq - beta_sq will be negative and transfer function will              % result in decaying expon waves for positive emdist, but increasing waves for neg emdist.)        if emdist<0            transferfa=zeros(lmax,mmax);            ind2=find(expon>0);            transferfa(ind2)=exp(1i*bprimeerfa*emdist*sqrt(expon(ind2)));        else            transferfa=exp(1i*bprimeerfa*emdist*sqrt(expon));        end        pp=ifft2(ifftshift(ferfa.*transferfa));  % pressure matrix at front face of model.    else        pp=ppe;    end    waitbar(0.5)    % --------- Approach C to model scattering; this assumes random variation > voxel size -----------------    if max(modelParams.randvc)~=0   % do next lines only if there will be scattering due to non-zero std dev of parameters.    % The next lines can modify pprimeterm such that there is random spatial variation in overall propagation in space:        rrandcrs=1:corrl:lmax+corrl; % rrandcrs is a course row (vector) grid for the random parameter variation, etc,        crandcrs=1:corrl:mmax+corrl; % where corrl is correlation length (in indices) of the random variation.        prandcrs=1:corrl:nmax+corrl;        randarrcrs=ones(length(rrandcrs),length(crandcrs),length(prandcrs)); % make a course grid array, spacing = corrl.        randarrcrs=single(randn(size(randarrcrs))); %array now contains normal random numbers: mean = 0, std dev = 1.        [cfine,rfine,pfine]=meshgrid(1:mmax,1:lmax,1:nmax); % define indices of final finer array of random numbers.        randarrfine=interp3(crandcrs,rrandcrs,prandcrs,randarrcrs,cfine,rfine,pfine,'*linear'); %interp to finer array.        vararray=abs(1+modelParams.randvc(modelParams.Modl).*randarrfine); % array of random variation around 1, never negative due to abs.    else        vararray=ones(size(modelParams.Modl),'single'); % if no scattering.    end    % -----------------        waitbar(1)        % Set up values for (virtual) layer 0 -- assume water in region in front of modl.    A0=fftshift(fft2(pp));    % pp is pressure pattern on front plane of modl.    Z0=modelParams.rho0*modelParams.c0;   % impedance of water.        close(hwb)    %  ======== Start of Multiple Reflection 'for' loop ===============================    nr= 0;   % number indicating what reflection is being done in multiple reflections.    for loop = 1:1:(numrefl/2) +1        %----- Start of forward increment in n (slice number) ( + z propagation direction) -------------        hwb1=waitbar(0,['Calculating Forward Propagation -----> Reflection #', num2str(nr)]);         for n=1:nmax            if nr==0     % first loop through forward propagation section (0 reflections):                % Set up 2D acoustic property matrices (to save memory) for this particular plane:                attmodl=modelParams.a(:,:,n)*1e2*erfaParams.fMHz;     % modelParams.a(i) is pressure total attenuation coefficient (assume linear freq dep).                rhomodl=modelParams.rho(:,:,n);  % modelParams.rho is density.                b=2*pi*f./modelParams.c(:,:,n);    % 2D matrix of propagation constant.                Z(:,:,n)=rhomodl.*modelParams.c(:,:,n).*(1-1i*attmodl./b);% impedance of layer n (slightly complex-be careful in P calcs).                if n==1                    Refl(:,:,1)=(Z(:,:,1) - Z0)./(Z(:,:,1) + Z0);     % layer 1 exception.                    pforb=pp.*(1+Refl(:,:,1));   %  pforb due to source pressure pattern pp.                    %  pforb--sim to pbackb--added 3/10/17; see "Latest Diagram....                else                    Refl(:,:,n)=(Z(:,:,n) - Z(:,:,n-1))./(Z(:,:,n) + Z(:,:,n-1));  % pressure reflection coeficient from layer n.                    pref(:,:,n-1)=Refl(:,:,n).*pfor(:,:,n-1);   % reflected pressure from forward propagation.                    pforb=pfor(:,:,n-1).*(1+Refl(:,:,n));                end                bprime(n)=sum(sum(abs(pforb).*b))./sum(sum(abs(pforb)));    % mean propagation constant,                % averaged over entire plane area, weighted by expected beam region.                alpha=((1:mmax)-ceil(mmax/2))*(2*pi/(bprime(n)*lx));   % vector of alpha for this layer.                beta=((1:lmax)-ceil(lmax/2))*(2*pi/(bprime(n)*ly));         % vector of beta. See earlier comments on alpha, beta.                [alpha_sq,  beta_sq] = meshgrid(alpha.^2, beta.^2);   % lmax by mmax matrices for transfer function (and r, rp).                sqrtexpon(:,:,n) = sqrt(1 - alpha_sq - beta_sq);  % sqrt part of exponent in transfer function and in r, rp below.                % (Even if it is imag when arg negative, evanescent waves will decay since bprime and dz always pos).                transf(:,:,n)=exp(1i*bprime(n).*dz.*sqrtexpon(:,:,n));            else   % second or later loop through forward propagation section (two or more reflections):                A0=ones(size(pp));                if n==1   % layer 1 exception                    pforb=pref2(:,:,1);                else                    pref(:,:,n-1)=Refl(:,:,n).*pfor(:,:,n-1);   % reflected pressure from forward propagation.                    pforb=pfor(:,:,n-1).*(1+Refl(:,:,n)) + pref2(:,:,n);                end            end   % end of separating first loop from later loops                        % Use Full Integration (equations from Scott Almquist):            % r = dz./sqrtexpon(:,:,n);    % oblique path length as function of cos of angles (alpha, beta).            rp = dz.*sqrtexpon(:,:,n);    % phase path length as function of cos of angles (alpha, beta).            complex_idx = imag(rp) > 0;            rp(complex_idx) = 0;            % r(complex_idx) = 0;  % to avoid exp increasing when r's are complex and dbvect is neg.            if n==1 || sum(sum(abs(A)))==0                Aabs=abs(A0);            else                Aabs=abs(A);            end            Aabs(Aabs<0.5*max(Aabs(:)))=0;            Asum=sum(Aabs(:));            rpave=sum(sum(rp.*Aabs))/Asum;            bvect = 2*pi*f./modelParams.c(:,:,n);     % propagation constant matrix; (units 1/m).            avect = modelParams.a(:,:,n)*1e-4*f;   % attenuation matrix; (units Np/m) (linear freq dep).            dbvect = bvect - bprime(n);  % excess of media prop constant over mean prop constant (bprime); can be neg.            %  These next lines do full numerical integration of the exponential propagation in the            %  space domain, weighted by A (see Eq. 4, Working Notes1, 5/8/06, revised 4/29/16).            % pprimeterm = NaN(lmax,mmax);  % preallocate vector; use NaN to detect error if pprimeterm not fully found.                        % MMK bugfix 03/31/2021            % since we are not preallocating with NaN, pprimeterm will end up being a single instead of a double.             % cast to double to be consistent with CalcHAS_ERFA8e            pprimeterm=double(exp(1i*dbvect.*rpave).* exp(-avect.*rpave));                        pprime=pforb.*pprimeterm.*vararray(:,:,n); % space-domain effects; vary by random amt (Approach C).            Aprime(:,:,n) = single(fftshift(fft2(pprime)));	% complex Eq (8); wraparound fft.            A = Aprime(:,:,n).*transf(:,:,n);	   % Eq (9rev).            pmat = single(ifft2(ifftshift(A)));	  % Eq (10).            TempA(:,:,n) = A;   % temporary storage of A for debugging.            pfor(:,:,n)=pmat;                        waitbar(n/nmax)        end        % --------- End of forward propagation --------------------------                close(hwb1);        nr=nr+1;        pfortot=pfortot + pfor;  % accumulate pfor; could also add at each n slice to eliminate pfor array for memory savings.                if nr==numrefl + 1       % Branch out of loop if next backward propagation not needed.            break        end                %------------ Start of backward increment in n (slice number) ( - z propagation direction) -------------        hwb2=waitbar(1,['<----- Calculating Backward Propagation, Reflection #', num2str(nr)]);                pback=zeros(size(modelParams.Modl),'single');         % pressure array, back propagation.        Aprimeback=zeros(size(modelParams.Modl),'single');  % see Aprime.        pref2=zeros(size(modelParams.Modl),'single');       % forward reflection        pback(:,:,nmax)=0;   % set up conditions for nmax layer.        A=0;                for n=(nmax-1):-1:1     % start at nmax-1 since pref=0 at last boundary.            % Note neg Refl since in opposite direction; Refl calculated earlier.            pbackb=pback(:,:,n+1).*(1-Refl(:,:,n+1)) + pref(:,:,n);  % add transmitted backward wave to reflected wave.            % Use Full Integration. Assume that bprime(n) is the same as in forward increments (so sqrtexpon same):            % r = dz./sqrtexpon(:,:,n);    % oblique path length as function of cos of angles (alpha, beta).            rp = dz.*sqrtexpon(:,:,n);    % phase path length as function of cos of angles (alpha, beta).            complex_idx = imag(rp) > 0;            rp(complex_idx) = 0;            % r(complex_idx) = 0;  % to avoid exp increasing when r's are complex and dbvect is neg.                        if n==nmax-1 || sum(sum(abs(A)))==0                Aabs(:) = 1;            else                Aabs=abs(A);            end            Aabs(Aabs<0.5*max(Aabs(:)))=0;            Asum=sum(Aabs(:));            rpave=sum(sum(rp.*Aabs))/Asum;                              % MMK bugfix 03/31/2021            % need to redefine here, because n is changing            bvect = 2*pi*f./modelParams.c(:,:,n);     % propagation constant matrix; (units 1/m).            avect = modelParams.a(:,:,n)*1e-4*f;   % attenuation matrix; (units Np/m) (linear freq dep).                                 % excess of media prop constant over mean prop constant (bprime); can be neg.            dbvect = bvect - bprime(n);            %  These next lines do full numerical integration of the exponential propagation in the            %  space domain, weighted by A (see Eq. 4, Working Notes1, 5/8/06, revised 4/29/16).            if A==0                pprimeterm=1;       % to avoid dividing by zero.            else                pprimeterm=double(exp(1i*dbvect.*rpave).*(exp(-avect.*rpave)));            end            pbackprime=pbackb.*pprimeterm.*vararray(:,:,n);     % backward prop in space domain.            Aprimeback(:,:,n)=single(fftshift(fft2(pbackprime)));	  % complex Eq (8); wraparound fft.            A=Aprimeback(:,:,n).*transf(:,:,n);	    % Eq (9rev) in reverse direction.            pmat=single(ifft2(ifftshift(A)));	% Eq (10).            pback(:,:,n)=pmat;            pref2(:,:,n)=-Refl(:,:,n).*pback(:,:,n);    % Note negative Refl since in opposite direction.                        waitbar(n/nmax)                    end        % ------- End of backward propagation -----------------                close(hwb2);                nr = nr +1;   % get ready for next loop        pbacktot=pbacktot + pback;    end     % ================ End of Multiple Reflection 'for' loop ======================    ptot=pfortot+pbacktot;    % complex add forward and backward waves.    % Added rpave/dz term to improve approximation of exponential.  MMK and DAC. April, 2020       Q=real(absmodl.*ptot.*conj(ptot./Z))*(rpave/dz); % power deposition for complex ptot and Z; now use ONLY abs coefficient.    maxQ=max(Q(:));    pout=ptot;      % NOTE: Can use gui to view any array by loading that array into base workspace with name 'pout'.    maxpout=max(abs(pout(:)));   % this is also redundantly done in plotx7, ploty7 and plotz7 for convenience.    tocend % function calcHAS()