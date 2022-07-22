clear all
close all
clc
format short

 set(0,'DefaultAxesFontSize',18,'DefaultTextFontSize',24,...
     'DefaultAxesFontName','Helvetica',...
    'DefaultTextFontName','Helvetica',...
     'DefaultAxesFontWeight','bold','DefaultTextFontWeight','bold',...
     'DefaultLineLineWidth',2,'DefaultLineMarkerSize',15,...
     'DefaultFigureColor','w','DefaultFigureResize','on')

% This code estimates first, second and total order sensitivity indices for
% cases with dependent variables. Originally based on the paper titled
% "estimation of global sensitivity indices for models with dependent
% variables" by Kucherenko 2012.

% code starts with loading parameter correlations (i.e., stat.mat) that
% has the mean vector (mu) and covariance matrix (sigma).

% Here we provide two options for model implementations. The function 
% modelOrig(var) is the original implementation of the quantum gate circuit that has 12
% variables (3 probabilities and 9 gate angles). However, in the current
% analysis, we utilize the function model(var), which is a simplified
% model. We have verified that they both provide the same results. 

%% FUNCTION DEFINITIONS

% the function [S,SSC,ST,D,f0,A,outA] = sobolCalc(n,Nums,mu,sigma,m)
% estimates the sensitivity indices. inbuilt function model calculates
% return the model output. update accordingly. 

% % function inputs
% n - number of parameters
% nums - sample size
% mu - mean vector
% sigma - covariance matrix
% m - number of processors

% % function outputs
% S - first order index
% SSC - second order closed sensitivity indices 
% ST - total order index
% other outputs - for book keeping

% function SS = ssc2SS(S,SSC) calculates the second order sensitivity index
% using first order and second order closed indices 

% SS_ij = SSC_ij - S_i - S_j

% % input variables 
% S = first order index
% SSC = second order closed index

% % output variables
% SS = second order index 

%% 

load stat.mat

mu = meanVal;
sigma = covVal;
vari = diag(sigma);
sigmaP = corrVal;

p = numel(mu);
n = p; % Number of parameters

Nums = 50000000; %sample size
m = 8; % Number of processes


[S,SSC,ST,D,f0,A,outA] = sobolCalc(n,Nums,mu,sigma,m);

SS = ssc2SS(S,SSC);

save data9.mat Nums S SSC ST SS

figure(1)
plot(S, 'ro','MarkerFaceColor','r')
hold on;
plot(ST,'ko','MarkerFaceColor','k')
xlim([0.5 p+0.5])
legend('First Order Indices','Total Order Indices','Location','NorthWest')
title('Sensitivity Indices')
set(gca,'XTick', [1:p])
set(gca, 'XTickLabel',paraName)
xlabel('Parameter')
set(gcf, 'Position',[10 10 1500 600]);
% print(gcf,sprintf('SobolOrigN'),'-dsvg','-r300');
saveas(gcf,sprintf('SobolOrigN'),'epsc');


figure(100)
plot(S, 'bs','MarkerFaceColor','b')
xlim([0.5 p+0.5])
title('First-Order Sensitivity Indices')
set(gca,'XTick', [1:p])
set(gca, 'XTickLabel',paraName)
xlabel('Parameter')
set(gcf, 'Position',[10 10 1500 600]);
% print(gcf,sprintf('SobolOrigN1st'),'-dsvg','-r300');
saveas(gcf,sprintf('SobolOrigN1st'),'epsc');
% 
% 
% figure(2)
% h = heatmap(paraName,paraName, round(sigma,5));
% h.Colormap = jet;
% h.MissingDataColor = 'w';
% h.GridVisible = 'off';
% h.FontSize = 16;
% colorbar
% xlabel('Parameters')
% ylabel('Parameters')
% title('\Sigma')
% set(gcf, 'Position',[10 10 1500 1125]);
% print(gcf,'SigmaOrig','-dsvg','-r300');
% 
% figure(3)
% h = heatmap(paraName,paraName, round(sigmaP,5));
% h.Colormap = jet;
% h.MissingDataColor = 'w';
% h.GridVisible = 'off';
% h.FontSize = 16;
% colorbar
% xlabel('Parameters')
% ylabel('Parameters')
% title('\Sigma_N')
% set(gcf, 'Position',[10 10 1500 1125]);
% print(gcf,'SigmaNormOrig','-dsvg','-r300');
% 
% figure(4)
% for j=1:p
%   for i= 1:j
%     if(i==j)
%         subplot(p,p,p*(j-1)+i);
%         hist(A(i,:),100);
% %         set(gca,'xtick',[])
%         set(gca,'ytick',[])
%     else
%         subplot(p,p,p*(j-1)+i);
%         plot(A(i,:),A(j,:),'.');
%         set(gca,'xtick',[])
%         set(gca,'ytick',[])
%     end
%     if(i == 1)
%         ylabel(paraName{j})
%     end
%     if(j==p)
%        xlabel(paraName{i})
%     end
% 
%   end  
% end
% set(gcf, 'Position',[10 10 1800 1350]);
% print(gcf,sprintf('paramCorrModelOrig'),'-dsvg','-r300');
% 
% 
% % figure(5)
% % for j=1:p
% %   for i= 1:j
% %     if(i==j)
% %         subplot(p,p,p*(j-1)+i);
% %         hist(A(i,:),100);
% %         set(gca,'xtick',[])
% %         set(gca,'ytick',[])
% %     else
% %         subplot(p,p,p*(j-1)+i);
% %         plot(A(i,:),outA,'.');
% %         set(gca,'xtick',[])
% %         set(gca,'ytick',[])
% %     end
% %     if(i == 1)
% %         ylabel(paraName{j})
% %     end
% %     if(j==p)
% %        xlabel(paraName{i})
% %     end
% % 
% %   end  
% % end
% % set(gcf, 'Position',[10 10 1800 1350]);
% % print(gcf,sprintf('paramCorrWithOutputOrig'),'-dsvg','-r300');
% 
% 
temp = SSC;
temp(temp==0) = nan;
figure(6)
h = heatmap(paraName,paraName, round(temp,3));
h.Colormap = jet;
h.MissingDataColor = 'w';
h.GridVisible = 'off';
h.FontSize = 20;
colorbar
h.Title = sprintf('Second Order Closed Sensitivity Indices');
h.XLabel = 'Parameters';
h.YLabel = 'Parameters';
set(gcf, 'Position',[10 10 1200 900]);
print(gcf,sprintf('SecondClosedOrder'),'-dsvg','-r1200');
saveas(gcf,sprintf('SecondClosedOrder'),'epsc');


temp = SS;
temp(temp==0) = nan;
figure(7)
h = heatmap(paraName,paraName, round(temp,3));
h.Colormap = jet;
h.MissingDataColor = 'w';
h.GridVisible = 'off';
h.FontSize = 20;
colorbar
h.Title = sprintf('Second Order Sensitivity Indices');
h.XLabel = 'Parameters';
h.YLabel = 'Parameters';
set(gcf, 'Position',[10 10 1200 900]);
% print(gcf,sprintf('SecondOrder'),'-dsvg','-r1200');
saveas(gcf,sprintf('SecondOrder'),'epsc');

%% Functions
% this is the unsimplified version of the Quantum circui model. It has 12
% inputs, 3 probabilities (i.e., p) for CNOT gate probabilities and 9
% quantum gate angles (i.e., theta). 
function Fout = modelOrig(varp)
    format long

    %% Calling parameters
    p = [1,1,1];
    theta = pi*ones(1,9);
    theta(1) = varp(1);
    theta(2) = varp(2);
    theta(4) = varp(3);
    theta(6) = varp(4);
    theta(7) = varp(5);
    theta(9) = varp(6);
    
    %% Part 1 // We seperate the whole cicuit into two parts. Each part involves 
    % operations on two qubits. This part is acting on q0 and q1
    % This part contains three Ry gates and one CNOT gate

    %% Initial state
    init_state = zeros(8,1);
    init_state(1) = 1;

    %% First Ry gate with param theta_1 on q0
    Id2 = eye(2,2);
    Ry_1 = [cos(theta(1)/2),-sin(theta(1)/2);sin(theta(1)/2),cos(theta(1)/2)];


    %% Second Ry gate with param theta_2 on q1

    Ry_2 = [cos(theta(2)/2),-sin(theta(2)/2);sin(theta(2)/2),cos(theta(2)/2)];

    step1 = kron(Id2,kron(Ry_2,Ry_1));

    state = step1 * init_state;
    %% CNOT gate with prop p ; in other case with prop (1-p), it does nothing
    Id = eye(4,4);
    CNOT = eye(4,4);
    CNOT(2,2) = cos(theta(3)/2);
    CNOT(4,4) = CNOT(2,2);


    CNOT(2,4) = -1i*sin(theta(3)/2);
    CNOT(4,2) = CNOT(2,4);


    num_rand_1 = rand(1);
    if num_rand_1 < p(1)
        step2 = kron(Id2,CNOT);
    else
        step2 = kron(Id2,Id);
    end
    state = step2 * state;

    %% Third Ry gate with param theta_3 on q1

    Ry_3 = [cos(theta(4)/2),-sin(theta(4)/2);sin(theta(4)/2),cos(theta(4)/2)];

    step3 = kron(Id2,kron(Ry_3,Id2));

    state = step3 * state;

    %% Part 2 // This part is acting on q1 and q2. Note that the WHOLE circuit 
    % doesn't equal to part 1 + part 2.
    % This part contains one Ry gate, two CNOT gates, and one X gate.

    %% CNOT gate with prop p ; in other case with prop (1-p), it does nothing
    Id = eye(4,4);
    CNOT = eye(4,4);

    CNOT(2,2) = cos(theta(5)/2);
    CNOT(4,4) = CNOT(2,2);


    CNOT(2,4) = -1i*sin(theta(5)/2);
    CNOT(4,2) = CNOT(2,4);


    num_rand_2 = rand(1);
    if num_rand_2 < p(2)
        step4 = kron(CNOT,Id2);
    else
        step4 = kron(Id,Id2);
    end
    state2 = step4 * state;

    %% X gate on q1

    X = [cos(theta(6)/2),-1i*sin(theta(6)/2);-1i*sin(theta(6)/2),cos(theta(6)/2)];
    step5 = kron(kron(Id2,X),Id2);

    state2 = step5 * state2;

    %% Ry gate on q2

    Ry_4 = [cos(theta(7)/2),-sin(theta(7)/2);sin(theta(7)/2),cos(theta(7)/2)];
    step6 = kron(kron(Ry_4,Id2),Id2);

    state2 = step6 * state2;

    %% CNOT gate with prop p ; in other case with prop (1-p), it does nothing
    Id = eye(4,4);
    CNOT = eye(4,4);

    CNOT(2,2) = cos(theta(8)/2);
    CNOT(4,4) = CNOT(2,2);


    CNOT(2,4) = -1i*sin(theta(8)/2);
    CNOT(4,2) = CNOT(2,4);

    num_rand_3 = rand(1);
    if num_rand_3 < p(3)
        step7 = kron(CNOT,Id2);
    else
        step7 = kron(Id,Id2);
    end
    state2 = step7 * state2;

    %% Ry gate on q2

    Ry_5 = [cos(theta(9)/2),-sin(theta(9)/2);sin(theta(9)/2),cos(theta(9)/2)];
    step8 = kron(kron(Ry_5,Id2),Id2);

    state2 = step8 * state2;

    Integral1 = sum(abs(state2(5:8).*state2(5:8)));      
    Fout = [Integral1];
end

%this is the simplified model for the quantum gate circuit. 

function Fout = model(theta)

    t1 = theta(1);
    t2 = theta(2); 
    t3 = pi;
    t4 = theta(3);
    t5 = pi;
    t6 = theta(4);
    t7 = theta(5);
    t8 = pi;
    t9 = theta(6);
    I = 1i;

     f4tmp = (-sin(t5/2)*sin(t6/2)*sin(t2/2 + t4/2)*cos(t7/2 + t9/2) ...
        - 1i*sin(t6/2)*sin(t2/2 + t4/2)*sin(t7/2 + t9/2)*cos(t5/2) ...
        + sin(t7/2 + t9/2)*cos(t6/2)*cos(t2/2 + t4/2))*cos(t1/2);
    
     f4tmp = abs(f4tmp)^2;    
    
     f5tmp = (1i*sin(t3/2)*sin(t5/2)*sin(t6/2)*cos(t2/2 - t4/2)*cos(t7/2 + t9/2) ...
        - sin(t3/2)*sin(t6/2)*sin(t7/2 + t9/2)*cos(t5/2)*cos(t2/2 - t4/2) ...
        - 1i*sin(t3/2)*sin(t2/2 - t4/2)*sin(t7/2 + t9/2)*cos(t6/2) ...
        - sin(t5/2)*sin(t6/2)*sin(t2/2 + t4/2)*cos(t3/2)*cos(t7/2 + t9/2) ...
        - 1i*sin(t6/2)*sin(t2/2 + t4/2)*sin(t7/2 + t9/2)*cos(t3/2)*cos(t5/2) ...
        + sin(t7/2 + t9/2)*cos(t3/2)*cos(t6/2)*cos(t2/2 + t4/2))*sin(t1/2);
        
    
    f5tmp = abs(f5tmp)^2;
    
    f6tmp = (-sin(t6/2)*sin(t8/2)*cos(t2/2 + t4/2)*cos(t7/2 - t9/2) ...
        - 1i*sin(t6/2)*sin(t7/2 + t9/2)*cos(t8/2)*cos(t2/2 + t4/2) ...
        + 1i*sin(t7/2)*sin(t9/2)*sin(t2/2 + t4/2)*sin(t5/2 - t8/2)*cos(t6/2) ...
        + sin(t7/2)*sin(t2/2 + t4/2)*cos(t6/2)*cos(t9/2)*cos(t5/2 - t8/2) ...
        + sin(t9/2)*sin(t2/2 + t4/2)*cos(t6/2)*cos(t7/2)*cos(t5/2 + t8/2) ...
        - 1i*sin(t2/2 + t4/2)*sin(t5/2 + t8/2)*cos(t6/2)*cos(t7/2)*cos(t9/2))*cos(t1/2);
        
    
    f6tmp = abs(f6tmp)^2;
    
    f7tmp = (I*sin(t3/2)*sin(t6/2)*sin(t8/2)*sin(t2/2 - t4/2)*cos(t7/2 - t9/2) ...
        - sin(t3/2)*sin(t6/2)*sin(t2/2 - t4/2)*sin(t7/2 + t9/2)*cos(t8/2) ...
        + sin(t3/2)*sin(t7/2)*sin(t9/2)*sin(t5/2 - t8/2)*cos(t6/2)*cos(t2/2 - t4/2) ...
        - I*sin(t3/2)*sin(t7/2)*cos(t6/2)*cos(t9/2)*cos(t2/2 - t4/2)*cos(t5/2 - t8/2) ...
        - I*sin(t3/2)*sin(t9/2)*cos(t6/2)*cos(t7/2)*cos(t2/2 - t4/2)*cos(t5/2 + t8/2) ...
        - sin(t3/2)*sin(t5/2 + t8/2)*cos(t6/2)*cos(t7/2)*cos(t9/2)*cos(t2/2 - t4/2) ...
        - sin(t6/2)*sin(t8/2)*cos(t3/2)*cos(t2/2 + t4/2)*cos(t7/2 - t9/2) ...
        - I*sin(t6/2)*sin(t7/2 + t9/2)*cos(t3/2)*cos(t8/2)*cos(t2/2 + t4/2) ...
        + I*sin(t7/2)*sin(t9/2)*sin(t2/2 + t4/2)*sin(t5/2 - t8/2)*cos(t3/2)*cos(t6/2) ...
        + sin(t7/2)*sin(t2/2 + t4/2)*cos(t3/2)*cos(t6/2)*cos(t9/2)*cos(t5/2 - t8/2) ...
        + sin(t9/2)*sin(t2/2 + t4/2)*cos(t3/2)*cos(t6/2)*cos(t7/2)*cos(t5/2 + t8/2) ...
        - I*sin(t2/2 + t4/2)*sin(t5/2 + t8/2)*cos(t3/2)*cos(t6/2)*cos(t7/2)*cos(t9/2))*sin(t1/2);
        
    
    f7tmp = abs(f7tmp)^2;
    
    Fout = f4temp+f5tmp+f6tmp+f7tmp;
end

%% In-bulit functions for Sobol Calculations

function [S,SS,ST,D,f0,A,outA] = sobolCalc(n,N,mu,sigma,m)
%% creating two independent random matrices with joint probability distribution
    tic
    for i = 1:n
        i
        temp = lhsnorm(0,1,2*N);
        t = temp';
        AN(i,:) = t(1:N); 
        BN(i,:) = t(N+1:2*N);
    end
    AA = cholD(sigma);
    A = AA*AN+mu';
    B = AA*BN+mu';
    
%     [A,B] = randomGJPDF2(n,N,mu, sigma);
%     initial function evaluations
    parfor (i = 1:N,m)
       outA(i) = model(A(:,i));
       outB(i) = model(B(:,i));
    end
    outC = [outA,outB];
    
%    Calculating f0 and D
    f0 = sum(outA)/(N);
    D = sum(outA.*outA)/(N) - f0^2;
    
    disp('f0 and D calculation complete')
    toc
    
%%  First order indic+es
     tic
    for i = 1:n
        disp('Calculation of S(i)')
        disp(i)
        
%       Random matrix generation based on conditional probability
        C = randomG1(n,N, mu, sigma, i,AN,BN);
%       Function evaluations
        parfor (j = 1:N,m)
            outC(j) = model(C(:,j));
        end
        
        temp = 0;
        for j = 1:N 
            temp = temp + outA(j)*(outC(j)-outB(j));
        end
        
        S(i) = temp/(D*N);
    end
    toc
%%   Second order indices
     tic
     clear i j temp
    for i = 1:n
        for j = i+1:n
            disp('Calculation of SS(ij)')
            in = [i,j];
            disp(in)
%             disp(j)
        
    %       Random matrix generation based on conditional probability
            C = randomGSS(n,N, mu, sigma, i,j,AN,BN);
    %       Function evaluations
            parfor (a = 1:N,m)
                outC(a) = model(C(:,a));
            end

            temp = 0;
            for a = 1:N 
                temp = temp + outA(a)*(outC(a)-outB(a));
            end

            SS(i,j,:) = temp;
        end
    end
    SS(n,:) = zeros(1,n);
    SS = SS/(D*N);
    toc
%% Total order indices
    tic
    for i = 1:n
        disp('Calculation of ST(i)')
        disp(i)
%       Random matrix generation based on conditional probability
        C = randomG2(n,N, mu, sigma, i,AN, BN);
%       Function evaluations
        parfor (j = 1:N,m)
            outC(j) = model(C(:,j));
        end
        
        temp = 0;
        for j = 1:N
            temp = temp + (outA(j)-outC(j))^2;
        end
        ST(i) = temp/(2*D*N);
    end
    toc
    
end


%% In-built functions for sample generation

function [yzBP] = randomGSS(n,N, mu, sigma, i,j,AN, BN)

    AA = cholD(sigma);
    x = AA*AN+mu';    
    ind = [i,j];
    % spliting x into y and z
    y = x(ind,:);

    %     transform wP to zB ~ N(0,1)
    k = 1:n;
    k(ind) = [];
    for l = 1:numel(k)
        zB(l,:) = BN(k(l),:);
    end
        
    %     Calculating muZC = muZ + sigmaYZ*inv(sigmaY)*(y-muY)
    [mu1,mu2] = decomMuSS(i,j,mu);
    [sigma1, sigma2, sigma12] = decomSigmaSS(i,j,sigma);

    muC = mu2'+sigma12'*inv(sigma1)*(y-mu1');
    sigmaC = sigma2'-sigma12'*inv(sigma1)*sigma12;

    %     cholesky decomposition
    Ac = cholD(sigmaC);
    zBP = Ac*zB+muC;

    yzBP = zeros(n,N);
%     for l = 1:n
%         if(l < i)
%             yzBP(l,:) = zBP(l,:);
%         elseif (l == i)
%             yzBP(l,:) = y(1,:);
%         elseif(l < i && l > j)
%             yzBP(l,:) = zBP(l-1,:);
%         elseif (l == j)
%             yzBP(l,:) = y(2,:);    
%         elseif(l > j)
%             yzBP(l,:) = zBP(l-2,:);
%         end
%     end
    indC = 1:n;
    indC(ind) = [];
    for l = 1:numel(ind)
        yzBP(ind(l),:) = y(l,:);
    end
    for l = 1:numel(indC)
        yzBP(indC(l),:) = zBP(l,:);
    end


end


function [yzBP] = randomG1(n,N, mu, sigma, i,AN, BN)

    AA = cholD(sigma);
    x = AA*AN+mu';    
    % spliting x into y and z
    y = x(i,:);

    %     transform wP to zB ~ N(0,1)
    k = 1:n;
    k(i) = [];
    for j = 1:n-1
        zB(j,:) = BN(k(j),:);
    end
        
    %     Calculating muZC = muZ + sigmaYZ*inv(sigmaY)*(y-muY)
    [mu1,mu2] = decomMu(i,mu,1);
    [sigma1, sigma2, sigma12] = decomSigma(i, sigma, 1);
    
    muC = mu2'+sigma12'*inv(sigma1)*(y-mu1);
    sigmaC = sigma2'-sigma12'*inv(sigma1)*sigma12;

    %     cholesky decomposition
    Ac = cholD(sigmaC);
    zBP = Ac*zB+muC;

    yzBP = zeros(n,N);
    for j = 1:n
        if(j < i)
            yzBP(j,:) = zBP(j,:);
        elseif (j == i)
            yzBP(j,:) = y;
        else
            yzBP(j,:) = zBP(j-1,:);
        end
    end

end


function [yzBP] = randomG2(n,N, mu, sigma, i,AN, BN)
AA = cholD(sigma);
    x = AA*AN+mu';    
    % spliting x into y and z
    y = x(i,:);
    temp = x;
    temp(i,:) = [];
    z = temp;

    %     transform vP to yB ~ N(0,1)
    yB = BN(i,:);

        
    %     Calculating muZC = muZ + sigmaYZ*inv(sigmaY)*(y-muY)
    [mu1,mu2] = decomMu(i,mu,2);
    [sigma1, sigma2, sigma12] = decomSigma(i, sigma, 2);
    
    muC = mu2'+sigma12'*inv(sigma1)*(z-mu1');
    sigmaC = sigma2'-sigma12'*inv(sigma1)*sigma12;

    %     cholesky decomposition
    Ac = cholD(sigmaC);
    yBP = Ac*yB+muC;

    yzBP = zeros(n,N);
    for j = 1:n
        if(j < i)
            yzBP(j,:) = z(j,:);
        elseif (j == i)
            yzBP(j,:) = yBP;
        else
            yzBP(j,:) = z(j-1,:);
        end
    end

end



function [muI,muNI] = decomMu(i,mu,k)
    if (k==1)
        muI = mu(i);
        temp = mu;
        temp(i) = [];
        muNI = temp;
    else
        muNI = mu(i);
        temp = mu;
        temp(i) = [];
        muI = temp;
    end
end

function [muI,muNI] = decomMuSS(i,j, mu)
        ind = [i,j];
        muI = mu(ind);
        temp = mu;
        temp(ind) = [];
        muNI = temp;
end

function [sigma1, sigma2, sigma12] = decomSigma(i, sigma, k)
    if(k==1)
        sigma1 = sigma(i,i);
        temp = sigma;
        temp(i,:) = [];
        temp(:,i) = [];
        sigma2 = temp;
        temp = sigma(i,:);
        temp(i) = [];
        sigma12 = temp;
        
    else
        sigma2 = sigma(i,i);
        temp = sigma;
        temp(i,:) = [];
        temp(:,i) = [];
        sigma1 = temp;
        temp = sigma(i,:);
        temp(i) = [];
        sigma12 = temp';
        sigma12 = temp';
        
    end
end

function [sigma1, sigma2, sigma12] = decomSigmaSS(i,j, sigma)
        sigma1(1,1) = sigma(i,i);
        sigma1(2,2) = sigma(j,j);
        sigma1(1,2) = sigma(i,j);
        sigma1(2,1) = sigma(j,i);

        ind = [i,j];
        temp = sigma;
        temp(ind,:) = [];
        temp(:,ind) = [];
        
        sigma2 = temp;
        temp = sigma(ind,:);
        temp(:,ind) = [];
        sigma12 = temp;
end



function L = cholD(A)
    LL = chol(A);
    L = LL';
end

function SS = ssc2SS(S,SSC)
    SS = zeros(size(SSC));
    for i = 1:numel(S)
        for j = i+1:numel(S)
            SS(i,j) = SSC(i,j) - S(i) - S(j);
        end
    end
end
