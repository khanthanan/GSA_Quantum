clear; clc; close all;format long g;

 set(0,'DefaultAxesFontSize',18,'DefaultTextFontSize',24,...
     'DefaultAxesFontName','Helvetica',...
    'DefaultTextFontName','Helvetica',...
     'DefaultAxesFontWeight','bold','DefaultTextFontWeight','bold',...
     'DefaultLineLineWidth',1,'DefaultLineMarkerSize',10,...
     'DefaultFigureColor','w','DefaultFigureResize','on')
 
 
%% Below is an example of using the DRAMMIMO package.

%% Load the data.
tic
disp('Loading data...');

% load IBM_full_data
load IBM_full_data_athens

IBM_full_data = IBM_full_data_athens;

n = length(IBM_full_data);%length(inputData1(:,1)); %number of probability estimates for each state
sig_meas = 0.03; %scaled std. dev. of measurements
%theoretical outputs
out_state_tmp = [0.000000000000000
0.000000000000000
0.368630854944285
0.368630854944285
0
0.666666666666667
0.376681656788360
0.376681656788360];

out_state = out_state_tmp.^2;

use_fic_data = 0; % Use fictitious data or not.

if use_fic_data == 1
    outputData1 = sig_meas*randn(n,1);
    outputData2 = sig_meas*randn(n,1);
    outputData3 = out_state(3)*(1 + sig_meas*randn(n,1));
    outputData4 = out_state(4)*(1 + sig_meas*randn(n,1));
    outputData5 = sig_meas*randn(n,1);
    outputData6 = out_state(6)*(1 + sig_meas*randn(n,1));
    outputData7 = out_state(7)*(1 + sig_meas*randn(n,1));
    outputData8 = out_state(8)*(1 + sig_meas*randn(n,1));
else
    outputData1 = IBM_full_data(:,1);
    outputData2 = IBM_full_data(:,2);
    outputData3 = IBM_full_data(:,3);
    outputData4 = IBM_full_data(:,4);
    outputData5 = IBM_full_data(:,5);
    outputData6 = IBM_full_data(:,6);
    outputData7 = IBM_full_data(:,7);
    outputData8 = IBM_full_data(:,8);
    outputData9 = IBM_full_data(:,6)+IBM_full_data(:,7)+IBM_full_data(:,8);
end

inputData1 = linspace(1,n,n)';%Dummy variable inputs
inputData2 = linspace(1,n,n)';
inputData3 = linspace(1,n,n)';
inputData4 = linspace(1,n,n)';
inputData5 = linspace(1,n,n)';
inputData6 = linspace(1,n,n)';
inputData7 = linspace(1,n,n)';
inputData8 = linspace(1,n,n)';
% inputData9 = linspace(1,n,n)';


%% Set up the DRAMMIMO.

disp('Setting DRAMMIMO...');

% Set the data struct.
% Add however many sets of data here. Just make sure .xdata and .ydata
% have the same length.
% This example has two data sets.
% .xdata contains the input data.
% data.xdata = {inputData6,inputData7,inputData8,inputData9};
data.xdata = {inputData1,inputData2,inputData3,inputData4,inputData5,inputData6,inputData7,inputData8};
% .ydata contains the output data.
% data.ydata = {outputData6,outputData7,outputData8,outputData9};
% data.ydata = {outputData1,outputData2,outputData3,outputData4,outputData5,outputData6,outputData7,outputData8,outputData9};
data.ydata = {outputData1,outputData2,outputData3,outputData4,outputData5,outputData6,outputData7,outputData8};
% data.ydata = {outputData6,outputData7,outputData8};

% Set the model struct.
% Add however many number of models here. Just make sure the number matches 
% the number of data sets.
% This example has two models.
 % .fun contains the functions that can generate model predictions.
% model.fun = {@getModelResponse5, ...
%     @getModelResponse6, @getModelResponse7, @getModelResponse_quadrature};


% model.fun = {@getModelResponse, @getModelResponse1, ...
%     @getModelResponse2, @getModelResponse3, ...
%     @getModelResponse4, @getModelResponse5, ...
%     @getModelResponse6, @getModelResponse7, @getModelResponse_quadrature};

model.fun = {@getModelResponse, @getModelResponse1, ...
    @getModelResponse2, @getModelResponse3, ...
    @getModelResponse4, @getModelResponse5, ...
    @getModelResponse6, @getModelResponse7};

% model.fun = {@getModelResponse5, ...
%     @getModelResponse6, @getModelResponse7};


% .errFun contains the functions that compare model predictions with data.
% model.errFun = {@getModelResponseError5, ...
%     @getModelResponseError6, @getModelResponseError7};

% model.errFun = {@getModelResponseError5, ...
%     @getModelResponseError6, @getModelResponseError7, ...
%     @getModelResponseError_quadrature};
% 
% model.errFun = {@getModelResponseError, @getModelResponseError1, ...
%     @getModelResponseError2, @getModelResponseError3, ...
%     @getModelResponseError4, @getModelResponseError5, ...
%     @getModelResponseError6, @getModelResponseError7, ...
%     @getModelResponseError_quadrature};


model.errFun = {@getModelResponseError, @getModelResponseError1, ...
    @getModelResponseError2, @getModelResponseError3, ...
    @getModelResponseError4, @getModelResponseError5, ...
    @getModelResponseError6, @getModelResponseError7, ...
    };



theta_nom = [2.03135031847622,0.668964074268407,pi,-0.668964074268407,...
    pi,pi,0.774596669241483,pi,-0.774596669241483];

th = [2.03135031847622,0.668964074268407,pi,-0.668964074268407,pi,pi,0.774596669241483,pi,-0.774596669241483];

%parameter analysis
theta1 = th(1);
theta2 = th(2);
theta3 = th(3);
theta4 = th(4);
theta5 = th(5);
theta6 = th(6);
theta7 = th(7);
theta8 = th(8);
theta9 = th(9);

ot  = 0.5;

% % Set the modelParams struct.
% .table = {parameter name, initial value, lower limit, upper limit}.
% 
modelParams.table = {{'\theta_1', theta1, 1.75, 2.25};
                     {'\theta_2', theta2, 0, 2};
%                      {'\theta_3', theta3, (1-ot)*theta3, (1+ot)*theta3};
                     {'\theta_4', theta4, -2, 0};
%                      {'\theta_5', theta5, (1-ot)*theta5, (1+ot)*theta5};
                     {'\theta_6', theta6, (1-ot)*theta6, (1+ot)*theta6};
                     {'\theta_7', theta7, 0, 2};
%                      {'\theta_8', theta8, (1-1.5*ot)*theta8, (1+ot)*theta8};
                     {'\theta_9', theta9, -2, 0};
                    }
% modelParams.table = {{'\theta_1', theta1, 1.75, 2.25}, ...
%                      {'\theta_2', theta2, 0, 1},...
%                      {'\theta_3', theta3, (1-0.001)*theta3, (1+ot)*theta3},...
%                      {'\theta_4', theta4, -1, 0},...
%                      {'\theta_5', theta5, (1-0.001)*theta5, (1+ot)*theta5}, ...
%                      {'\theta_6', theta6, (1-0.001)*theta6, (1+0.5*ot)*theta6},...
%                      {'\theta_7', theta7, 0, 1},...
%                      {'\theta_8', theta8, (1-0.001)*theta8, (1+ot)*theta8},...
%                      {'\theta_9', theta9, -1, 0}};%,...
% % %                      %{'p_1',p1,0,1}};



% .extra can pass extra parameter values that are not being estimated to 
% each model. Empty cells if not necessary.
modelParams.extra = {{0}, {0},{0}, {0}, {0}, {0}, {0}, {0}, {0}};

paraName = {'\theta_1','\theta_2','\theta_4','\theta_6','\theta_7','\theta_9'};
% paraName = {'\theta_1','\theta_2','\theta_3','\theta_4','\theta_5','\theta_6','\theta_7','\theta_8','\theta_9'};

% Set the DRAMParams struct.
% Number of iterations that are already done.
DRAMParams.numIterationsDone = 1;
% Number of iterations that are expected to be done.
DRAMParams.numIterationsExpected = 20e6;
% Every X number of iterations, display the parameter values at this 
% iteration in the command window.
DRAMParams.numIterationsDisplay = 10000;
% Every X number of iterations, save the estimation chains up to this 
% iteration to a .mat file in current folder.
DRAMParams.numIterationsSave = 0;
% For initial run, the .previousResults struct is empty.
DRAMParams.previousResults.prior.psi_s = 1e-1*eye(numel(data.xdata));%[]; increased to ...
                                                %avoid iwishrnd singularity
DRAMParams.previousResults.prior.nu_s = [];
DRAMParams.previousResults.chain_q = [];
DRAMParams.previousResults.last_cov_q = [];
DRAMParams.previousResults.chain_cov_err = [];

%% Run the DRAMMIMO.

% The uncertainty quantification results consist of three parts:
% 1. Estimation chains.
% 2. Posterior densities.
% 3. Credible and prediction intervals.

disp('Running DRAMMIMO...');

% Get the estimation chains.
[prior, chain_q, last_cov_q, chain_cov_err] = ...
    getDRAMMIMOChains(data, model, modelParams, DRAMParams);

% The estimation chains can be obtained in multiple runs.
% Uncomment the following portion of code to have a 2nd run.
% Remember to adjust the DRAMParams struct accordingly.
% -------------------------------------------------------------------------
% DRAMParams.numIterationsDone = 5000;
% DRAMParams.numIterationsExpected = 10000;
% DRAMParams.numIterationsDisplay = 200;
% DRAMParams.numIterationsSave = 1000;
% DRAMParams.previousResults.prior.psi_s = prior.psi_s;
% DRAMParams.previousResults.prior.nu_s = prior.nu_s;
% DRAMParams.previousResults.chain_q = chain_q;
% DRAMParams.previousResults.last_cov_q = last_cov_q;
% DRAMParams.previousResults.chain_cov_err = chain_cov_err;
% [prior, chain_q, last_cov_q, chain_cov_err] = ...
%     getDRAMMIMOChains(data, model, modelParams, DRAMParams);
% -------------------------------------------------------------------------

% Get the posterior densities.
% Assume the second half of the chains are in steady-state, but this 
% number is not necessarily to be true.
num = round(size(chain_q,1)/2)+1;
[vals,probs] = getDRAMMIMODensities(chain_q(num:end, :));

% Get the credible and prediction intervals.
% 500 is the rule of thumb number, and this number is suggested to be fixed.
nSample = 500;
[credLims,predLims] = ...
    getDRAMMIMOIntervals(data, model, modelParams, ...
                         chain_q(num:end,:),chain_cov_err(:,:,num:end),...
                         nSample);

%% Display the results.

disp('Presenting results...');

% Mean parameter estimation.
disp('Mean Parameter Estimation = ');
theta_est = mean(chain_q(num:end,:));
disp(mean(chain_q(num:end,:)));

set(0,'DefaultAxesFontSize',8,'DefaultTextFontSize',18,...
     'DefaultAxesFontName','Helvetica',...
    'DefaultTextFontName','Helvetica',...
     'DefaultAxesFontWeight','bold','DefaultTextFontWeight','bold',...
     'DefaultLineLineWidth',1,'DefaultLineMarkerSize',10,...
     'DefaultFigureColor','w','DefaultFigureResize','on')

% % new Plots

fs = 20;
nq = numel(paraName);
figure(2)
for i = 1:nq
    subplot(nq,1,i);
    hold on;
    plot(1:1:size(chain_q,1),chain_q(:,i),'b.');
    hold off;
    set(gca,'fontsize',fs,'xtick',[],...
    'xlim',[0,DRAMParams.numIterationsExpected],'ylim',[min(chain_q(:,i)),max(chain_q(:,i))]);
    box on;
    ylabel(paraName{i});
end
xlabel('Iterations');
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('figure2'),'-dsvg','-r600');
       

figure(20)
for i = 1:nq
    subplot(nq,1,i);
    hold on;
    plot(1:1:size(chain_q,1),chain_q(:,i),'b.');
    hold off;
    set(gca,'fontsize',fs,'xtick',[],...
    'xlim',[0,DRAMParams.numIterationsExpected],'ylim',[modelParams.table{i}{3},modelParams.table{i}{4}]);
    box on;
    ylabel(paraName{i});
end
xlabel('Iterations');
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('figure20'),'-dsvg','-r600');

       

 set(0,'DefaultAxesFontSize',16,'DefaultTextFontSize',24,...
     'DefaultAxesFontName','Helvetica',...
    'DefaultTextFontName','Helvetica',...
     'DefaultAxesFontWeight','bold','DefaultTextFontWeight','bold',...
     'DefaultLineLineWidth',1,'DefaultLineMarkerSize',10,...
     'DefaultFigureColor','w','DefaultFigureResize','on')



% Posterior densities.
figure(3)
for i = 1:nq
    subplot(1,nq,i);
    hold on;
    plot(vals(:,i),probs(:,i),'k','linewidth',3);
    hold off;
    box on;
    xlabel(paraName{i});
end
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('Posterior Densities_All'),'-dsvg','-r600');


figure(12)
start = 1;
p = nq; %number of parameters
for j=1:p
  for i= 1:j
    if(i==j)
        subplot(p,p,p*(j-1)+i);
        hist(chain_q(start:end,i),100);
%         set(gca,'xtick',[])
        set(gca,'ytick',[])
    else
        subplot(p,p,p*(j-1)+i);
        plot(chain_q(start:end,i),chain_q(start:end,j),'.');
        set(gca,'xtick',[])
        set(gca,'ytick',[])
    end
    if(i == 1)
        ylabel(paraName{j})
    end
    if(j==p)
       xlabel(paraName{i})
    end

  end  
end
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('FIGURE12'),'-dsvg','-r600');

start = 0.6*max(size(chain_q));
[vals,probs] = getDRAMMIMODensities(chain_q(start:end, :));

figure(13)
for i = 1:nq
    subplot(1,nq,i);
    hold on;
    plot(vals(:,i),probs(:,i),'k','linewidth',3);
    hold off;
    box on;
    xlabel(paraName{i});
end
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('FIGURE13'),'-dsvg','-r1200');

meanVal = mean(chain_q(start:end,:));
modeVal = mode(chain_q(start:end,:));
cova = cov(chain_q(start:end,:));
cova2 = corrcoef(chain_q(start:end,:));
vara = var(chain_q(start:end,:));

save finalChN.mat chain_q

meanValModel = getModelResponse5(meanVal,data.xdata,modelParams.extra)+getModelResponse6(meanVal,data.xdata,modelParams.extra)+getModelResponse7(meanVal,data.xdata,modelParams.extra);
modeValModel = getModelResponse5(modeVal,data.xdata,modelParams.extra)+getModelResponse6(modeVal,data.xdata,modelParams.extra)+getModelResponse7(modeVal,data.xdata,modelParams.extra);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create new set of credible and prediction intervals based on the 
% Gauss quadrature circuit output
nSample = 500;
model.fun = {@getModelResponse_quadrature};
model.errFun = {@getModelResponseError_quadrature};
data.xdata = {inputData7}; %needed so only one input goes into the one model function
% DRAMParams.previousResults.prior.psi_s = 1e-8; %has to be 1x1 instead of 8x8 b/c
%                                                %we only have one quantity
%                                                %of interest...can get it
%                                                %from data and known
%                                                %quadrature value

% [credLims,predLims] = ...
%     getDRAMMIMOIntervals_quadrature(data, model, modelParams, ...
%                          chain_q(num:end,:),chain_cov_err(:,:,num:end),...
%                          nSample);
var_quadrature = 5e-3*ones(1,1,length(chain_cov_err(:,:,num:end))); %we need to update this
                                       %with the true variance of the
                                       %quadrature integral error
[credLims,predLims] = ...
    getDRAMMIMOIntervals_quadrature(data, model, modelParams, ...
                         chain_q(num:end,:),var_quadrature,...
                         nSample);

% Credible and prediction interval for Data I.
% figNum = figNum+1;
% fh = figure(figNum);
% set(fh,'outerposition',96*[2,2,7,6]);
% set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.7,2]);
figure(14)
hold on;
h(1) = patch([data.xdata{1}',fliplr(data.xdata{1}')],...
      [predLims(1,:,1),fliplr(predLims(3,:,1))],...
      [1,0.75,0.5],'linestyle','none');
h(2) = patch([data.xdata{1}',fliplr(data.xdata{1}')],...
      [credLims(1,:,1),fliplr(credLims(3,:,1))],...
      [0.75,1,0.5],'linestyle','none');
h(3) = plot(inputData7,credLims(2,:,1),'k');
h(4) = plot(inputData6,outputData6+outputData7+outputData8,'bo'); %This should be the quadrature output
h(5) = plot(inputData6,meanValModel(1)*ones(1,length(inputData6)),'ks-')
h(6) = plot(inputData6,modeValModel(1)*ones(1,length(inputData6)),'kp-')
hold off;
box on;
lh = legend(h,'95% Pred Interval','95% Cred Interval','Quadrature Model','Quadrature Output','MeanVal','ModelVal','location','se');
lh.FontSize = 18;
legend boxoff;
xlabel('data samples of P'); 
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('FIGURE14'),'-dsvg','-r1200');

toc


D = chain_q;

save allChain.mat chain_q

ind = 1:length(D);
ind(chain_q(:,5) < 0.75) = [];
G = D(ind,:);
ind = 1:numel(ind);
ind(G(:,6) < -0.7) = [];
F = G(ind,:);
ind = 1:numel(ind);
ind(F(:,2) < 0.7) = [];
E = F(ind,:);

figure(21)
start = 1;
p = nq; %number of parameters
for j=1:p
  for i= 1:j
    if(i==j)
        subplot(p,p,p*(j-1)+i);
        hist(E(start:end,i),100);
%         set(gca,'xtick',[])
        set(gca,'ytick',[])
    else
        subplot(p,p,p*(j-1)+i);
        plot(E(start:end,i),E(start:end,j),'.');
        set(gca,'xtick',[])
        set(gca,'ytick',[])
    end
    if(i == 1)
        ylabel(paraName{j})
    end
    if(j==p)
       xlabel(paraName{i})
    end

  end  
end
set(gcf, 'Position',[10 10 1800 1350]);
print(gcf,sprintf('FIGURE21'),'-dsvg','-r300');

meanVal = mean(E);
covVal = cov(E);
corrVal = corrcoef(E);

save stat.mat meanVal covVal corrVal paraName

toc