geomOffsetsXYZ = [0, 0, 0];
steeringXYZ = [0, 0, 0];
rotationThetaPsi = [0, 0];
pivotIndices = [0, 0, 0]; 

% This script sets up the parameters necessary to run an accoustic simulation on a segmented
% model using the hybrid angular spectrum technique (HAS, Copyright D.A. Christensen 2019).
% 
% This script first defines the variables required by HAS, optionally computes steering phases, 
% optionally rotates the model, calls calcHAS(), then, if necessary, rotates the model, Q, and pout
% back to original orientation.
% 
%        - geomOffsetsXYZ is a 1x3 vector of doubles representing x, y, and z offets from the center 
%          of the model. Used to simulate transducer positioning. (mm)
%        - steeringXYZ is a 1x3 vector of doubles representing electronic steering in x, y, and z. (m)
%        - rotationThetaPsi is a 1x2 vector of doubles representing angles (theta and psi) that 
%          describe the simulated rotation of the transducer (degrees)
%        - pivotIndices is a 1x3 vector of integers that define the pivot point about which to 
%          rotate the model
%
% Notes: 
% 1) If you are rotating the model (to simulate rotating the transducer), the function
%    rotvolpivrecenter() will be called. After rotating, note that the 3D property matrices 
%    (c, a, rho, randvc, and aabs) must be redefined using the newly rotated model. 
%    Rotvolpivrecenterinterp() will be used to rotate Q and pout back.
% 2) This function calls three outside functions, SteeringPhasesPA8(), rotvolpivrecenter(), and
%    rotvolpivrecenterinterp(), located in the HAS directory. Be sure that directory is on your
%    path.
%
% Michelle Kline, IGT/UCAIR
% December 2020
% michelle.kline.igt@gmail.com

%% ERFA parameters ---------------------------------------------------------------------------------
% load ERFA file 
disp("Loading ERFA...");
ERFA = load('C:\Users\Rayane\Documents\Code\HAS_Code\Setup\Erfa.mat');
% these are the fields loaded with the erfa .mat file:
%   ElemLoc      element locations (angles from center of curvature to the transducer face, 
%                where phi is azimuthal index, in horizontal plane and theta is elevation index) (radians)        
%   Len          size of the ERFA plane                    
%   R            transducer radius of curvature               
%   dxp          incremental size of steps in ERFA plane (m)             
%   dyp          incremental size of steps in ERFA plane (m)
%   fMHz         transducer frequency                 
%   isPA         is this a phased array? (true/false)                
%   perfa        pressure on the ERFA plane       
%   pfilename    name of file containing ERFA params                
%   relem        radius of circular element of array transducer, if applicable (m)            
%   sx           distance from furthest point of curved transducer face to ERFA plane (m)    
%   erfafilenm   the ERFA file used to create the phases
ERFA.Pr = 127.7;	% acoustic power output of transducer (W)	

%% Model parameters --------------------------------------------------------------------------------
% load the segmented model (as model.Modl)
disp("Loading model...");
model = load('C:\Users\Rayane\Documents\Code\HAS_Code\Setup\Modl_P6.mat');

% ----- media properties -----
% c = speed of sound in (m/s)  
model.c0 = 1500;       % speed of sound in water (m/s)
model.cVector(1)=1500;     % water 
model.cVector(2)=1436;     % breast cancer (liver c and rho)
model.cVector(3)=1514;     % fibro tissue (fat c and rho)
model.cVector(4)=1588.4;
model.cVector(5)=1537;
% normal breast (fat c and rho)
model.c = model.cVector(model.Modl);  % generate 3D property matrix for speed of sound
% a = pressure attenuation coefficient (Np/cm*MHz) (NOTE UNITS!)
model.aVector(1)=0.000;      
model.aVector(2)=0.043578;    
model.aVector(3)=0.0865;     
model.aVector(4)=0.071088;
model.aVector(5)=0.21158; 
model.a = model.aVector(model.Modl);  % generate 3D property matrix for attenuation
% aabs = pressure absorption coefficient [= att - scatt] in (Np/cm*MHz)
% rho =  density in (kg/m^3)
model.rho0 = 1000;     
model.rhoVector(1)=1.000e3;    
model.rhoVector(2)=0.928e3;    
model.rhoVector(3)=1.058e3;     
model.rhoVector(4)=1.090e3;     
model.rhoVector(5)=1.100e3;  
model.rho = model.rhoVector(model.Modl); % generate 3D property matrix for density
% randvc = the std dev of all parameters associated with each medium
% (statistical scattering Approach C).
model.randvcVector(1)=0.000; 
model.randvcVector(2)=0.000; 
model.randvcVector(3)=0.000; 
model.randvcVector(4)=0.000; 
model.randvcVector(5)=0.000; 
model.randvc = model.randvcVector(model.Modl);

model.aabsVector(1) = 0.000;
model.aabsVector(2) = 0.070;
model.aabsVector(3) = 0.090;
model.aabsVector(4) = 0.071088;
model.aabsVector(5) = 0.21158;
model.aabs = model.aabsVector(model.Modl);


%model.aabs = model.a;

model.parameters.corrl = 10;    % 10 indices
% ----- other model parameters -----
model.Dx = 0.5;	% resolution in 2nd MATLAB dimension (x/col)
model.Dy = 0.5;	% resolution in 1st MATLAB dimension (y/row)
model.Dz = 0.5;	% resolution in 3rd MATLAB dimension (z/pag)

%% Positioning parameters --------------------------------------------------------------------------
% ----- mechanical positioning (mechanical offset from center of model) -----
positioning.offsetxmm = geomOffsetsXYZ(1);     % ...along 2nd MATLAB dimension(mm)
positioning.offsetymm = geomOffsetsXYZ(2);     % ...along 1st MATLAB dimension(mm)
positioning.dmm = geomOffsetsXYZ(3);    % distance from transducer base to model base (mm)
% ----- electronic steering (*** NOTE: calls outside function ***) -----
positioning.h = steeringXYZ(1);     % ... in x-direction (along 2nd MATLAB dimension) (m)
positioning.v = steeringXYZ(2);     % ... in y-direction (alond 1st MATLAB dimension) (m)
positioning.z = steeringXYZ(3);     % ... in z-direction (along 3rd MATLAB dimension) (m)
ang=SteeringPhasesPA8(positioning.v, positioning.h, positioning.z, ERFA.R, ERFA.ElemLoc, ERFA.fMHz*1e6, model.c0);  % NOTE v, h, z order!
positioning.angpgvect = shiftdim((exp(-1i * ang))', -1);
clear ang

%% Number of reflections ---------------------------------------------------------------------------
numberOfReflections = 0;

%% Here is where we optionally rotate the model ----------------------------------------------------
if any(rotationThetaPsi)
    disp("Rotating...")
    notRotatedModel = model.Modl;
    model.Modl = rotvolpivrecenter(model.Modl, pivotIndices, model.Dx, model.Dy, model.Dz, rotationThetaPsi(1), rotationThetaPsi(2));
    % NOTE: after rotating the model, you must redefine the corresponding 3D property matrices.
    model.c = model.cVector(model.Modl);
    model.a = model.aVector(model.Modl);
    model.aabs = model.aabsVector(model.Modl);
    model.rho = model.rhoVector(model.Modl);
    model.randvc = model.randvcVector(model.Modl);
    % In HAS, rotating the model automatically sets the geometric focus to be the rotation pivot
    % point. Therefore, we must clear the transducer positioning offsets
    positioning.offsetxmm = 0;  
    positioning.offsetymm = 0;
end

%% now that inputs are constructed, call calcHAS()
disp("Running HAS...");
[Q, maxQ, pout_P6, maxpout] = calcHAS(ERFA, model, positioning, numberOfReflections);
%% rotate back
if any(rotationThetaPsi)
    disp("Rotating back...")
    model.Modl = notRotatedModel;
    pout_P6 = rotvolpivrecenterinterp(pout, pivotIndices, model.Dx, model.Dy, model.Dz, rotationThetaPsi(1), rotationThetaPsi(2), 0);
    Q = rotvolpivrecenterinterp(Q, pivotIndices, model.Dx, model.Dy, model.Dz, rotationThetaPsi(1), rotationThetaPsi(2), 0);
    % MMK fixme: after rotvolpivrecenterinterp, maxQ and maxpout might have changed (due to
    % interpolation?). Is this expected? How does GUI version handle this?
    maxQ = max(Q(:));
    maxpout = max(abs(pout_P6(:)));
end

% display slice at focus, use "edges6()" to display model edges
figure('Name', 'Model1');
zaxis=1:size(model.Modl,3);
yaxis=1:size(model.Modl,1);
[xxl,yyl]=edges6(squeeze(model.Modl(:,100,:)),zaxis,yaxis);
imagesc(squeeze(abs(pout_P6(:,floor(size(model.Modl,2)/2),:))));
axis image;
axis xy;
line(xxl,yyl,'LineWidth',1,'Color','w');


