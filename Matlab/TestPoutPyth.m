disp("Loading ERFA...");
ERFA = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\Erfa.mat');
ERFA.Pr = 100.0
%ERFA.Pr = 127.7

%disp("Loading model...");
model = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\Modl.mat');
%model = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\Modl_P6.mat');

%disp("Loading pout_pyth...")
pout_pyth = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\pout_pyth.mat')
%pout_pyth = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\pout_pyth_P6.mat')
 
%disp("Loading pout_mat...")
pout_mat = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\pout.mat')
%pout_mat = load('C:\Users\Rayane\Documents\Code\HAS_Code\HAS_Code\HAS_Code\pout_P6.mat')

% display slice at focus, use "edges6()" to display model edges
figure('Name', 'Pout_pyth');
zaxis=1:size(model.Modl,3);
yaxis=1:size(model.Modl,1);
[xxl,yyl]=edges6(squeeze(model.Modl(:,100,:)),zaxis,yaxis);
imagesc(squeeze(abs(pout_pyth.pout_pyth(:,floor(size(model.Modl,2)/2),:))));
colorbar
axis image;
axis xy;
line(xxl,yyl,'LineWidth',1,'Color','w');


% display slice at focus, use "edges6()" to display model edges
figure('Name', 'Pout_mat');
zaxis=1:size(model.Modl,3);
yaxis=1:size(model.Modl,1);
[xxl,yyl]=edges6(squeeze(model.Modl(:,100,:)),zaxis,yaxis);
imagesc(squeeze(abs(pout_mat.pout(:,floor(size(model.Modl,2)/2),:))));
colorbar
axis image;
axis xy;
line(xxl,yyl,'LineWidth',1,'Color','w');

% % display slice at focus, use "edges6()" to display model edges
figure('Name', 'Pout_difference');
zaxis=1:size(model.Modl,3);
yaxis=1:size(model.Modl,1);
[xxl,yyl]=edges6(squeeze(model.Modl(:,100,:)),zaxis,yaxis);
imagesc(squeeze(  abs(abs(pout_pyth.pout_pyth(:,floor(size(model.Modl,2)/2),:))  - abs(pout_mat.pout(:,floor(size(model.Modl,2)/2),:)))   )  );
colorbar
axis image;
axis xy;
line(xxl,yyl,'LineWidth',1,'Color','w');