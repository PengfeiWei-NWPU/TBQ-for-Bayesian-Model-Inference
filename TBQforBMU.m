%%%%%this code implementes the Transitional Bayesian Quadrature (TBQ) algorithm for Bayesian Model Updating, where the estimates 
%%%%%of both posterior density and model evidence are of concern. Please refer to my MSSP paper for the theoretical development.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%P. Wei. Bayesian Model Inference with Complex Posteriors: Exponential-Impact-Informed Bayesian Quadrature. Mechanical Systems &
%%        Signal Processing, 2025 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this code implementes case 1 of the second example reported in the above reference.

clear
clc
%%define the problem, the model parameters are denoted by "x"
nz = 2;
U1 = @(x) log(exp(-0.5*(x(:,1)-2).^2/0.08^2)+exp(-0.5*(x(:,1)+2).^2/0.08^2)) -  0.5*((sqrt(sum(x(:,1:2).^2,2))-2)/0.2).^2; %%energy function of x_1 and x_2 with multimodal features
LogLikelihood = @(z)U1(z);%define the composite energy function 
Likelihood = @(z)exp(LogLikelihood(z));% define the likelihood function


%%%%%%%%%%%%%%%%%%%%%setting the algorithm parameters%%%%%%%%%%%%%%%%%%%%%%%%   
N0 = 12; %%Number of initial sample size
StopThresholdMid = 0.15;% Stopping threshold for all layer execept the last tempering stage 
StopThresholdLast = 0.03;%Stopping thershold for the last tempering stage, smaller to achieve high accuracy
NMC = 1e4; % size of MC/MCMC samples for each tempering stage    
NTest = 5e6; %size of sample for computing the reference solution of the model evidence using Monte Carlo simulation
CV = 1.2;%parameters for controlling the variation of estimator for each layer, suggested to be 1~2
StopLimit = 2;%Terminate the training of the current layer when the stopping criterion is satisfied consecutively for "StopLimit" times.
beta = norminv(0.75);%%%parameters used for balancing the local exploitation and global exploration
Lchain = 20; % length of MCMC chain
zeta = 0.2;%%required when the M-H sampler is used for conditional sampling, for defining the length of steps of the proposal distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%initialzie the algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TestSamp = unifrnd(-4,4,NTest,nz);%test samples used for estimating the reference value, supervising the quality of intermediate model evidence, and for generating the CI estimates of model evidence 
TTestSamp = Likelihood(TestSamp);
ZMC = mean(TTestSamp);%compute the reference value of model evidence 
CovZMC = std(TTestSamp)/sqrt(NTest)/ZMC;% the CoV of ZMC
MCSamp(:,:,1) = unifinv(lhsdesign(NMC,nz),-4,4);%use latin-hypercube to produce the MC sampels for the first tempering stage
Zfixed=[1];%initialize the model evidence
ZRef = [1];%initialize the reference value of model evidence (will be estimated by MC based on "TestSamp")
Z_CI = [1,1];%initialize the credible interval (CI) of model evidence
gamma_fixed=[0];%%initialize the value of gamma_1 as zero
PriorPDF=@(z)ones(size(z,1),1);
for i=1:nz% define the prior density
    PriorPDF=@(z)PriorPDF(z).*unifpdf(z(:,i),-4,4);
end
LikeSamp(:,1) = ones(NMC,1);% initialize the sample values of likelihood of the first stage computed at "TestSamp" 
Ttrain = unifinv(lhsdesign(N0,nz),-4,4);% Initial training points
Ytrain = LogLikelihood(Ttrain);% label the initial training points
j=1;%initialize the index of tempering stage
Ncall = N0;%used for recording the total number of model calls
NcallEachLayer(1)=N0;% used for recording the number of model calls consumed by each tempering stage 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%create "Nplot" points in each dimension for plotting the results of (intermediate) posetrior densities 
Nplot = 100;
Zplot = linspace(-4,4,Nplot);
[Xgrid, Ygrid] = meshgrid(Zplot,Zplot);

% figure%%required if you want to plot the MCMC sampels


options = optimoptions(@ga,'UseParallel',true,'UseVectorized',true,'PopulationSize',50); %%setting parameters for GA algorithm  
while 1==1
    Stopflag=0;
    while 1==1
        GPRmodel = fitrgp(Ttrain,Ytrain,'KernelFunction','ardsquaredexponential'...
                   ,'BasisFunction','constant','Sigma', 2e-5, 'ConstantSigma', true,'SigmaLowerBound', eps,'Standardize',false);  
        [PostMeanSamp,PostSDSamp,~] = predict(GPRmodel,MCSamp(:,:,j));

         weight = @(gamma)exp(gamma*PostMeanSamp)./LikeSamp(:,j);%%define the ratio of consecutive likelihoods for MCSamp(:,:,j-1)
         weightLim = weight(1);%% weights of samples computed at gamma=1
         if std(weightLim)/mean(weightLim)>CV% in case gamma_Active is smaller than one and needs to be updated
            gamma_Active = fminbnd(@(gamma)abs(std(weight(gamma))/mean(weight(gamma))-CV),gamma_fixed(j),1);
            StopThreshold = StopThresholdMid;
         elseif std(weightLim)/mean(weightLim)<CV% in case the maximum COV is lower than CV, set gamma as 1
            gamma_Active = 1;
            StopThreshold = StopThresholdLast;
         end
 

        DeltaGamma = gamma_Active-gamma_fixed(j);
        CI_Z_Active(j,:)=[mean(exp(gamma_Active*(PostMeanSamp-beta*PostSDSamp))./LikeSamp(:,j)),mean(exp(gamma_Active*(PostMeanSamp+beta*PostSDSamp))./LikeSamp(:,j))];
        UQ_CI(Ncall-N0+1) = (CI_Z_Active(j,2) - CI_Z_Active(j,1))/(CI_Z_Active(j,2) + CI_Z_Active(j,1));%%prediction uncertainty of evidence, normalzied
        fprintf('Active gamma value： %.4f\n', gamma_Active);
        fprintf('Active error： %.4f\n', UQ_CI(Ncall-N0+1));
        fprintf('Active number of model calls： %d\n', Ncall);
        if UQ_CI(Ncall-N0+1)<StopThreshold
            Stopflag = Stopflag+1;
        else
            Stopflag = 0;
        end
        if Stopflag>=StopLimit
            break
        else
%%%searching the next best training point by maximziing the acquisition function using the GA algorithm%%%%%%%%%%%%%%%%%%%%%
            [Tnew,~] = ga(@(z)AcqFunTypeI(z,GPRmodel,beta,gamma_Active,PriorPDF),nz,[],[],[],[],-4.0*ones(1,nz),4.0*ones(1,nz),[],options);
            Ncall = Ncall+1;
            Ttrain = [Ttrain;Tnew];
            Ytrain = [Ytrain;LogLikelihood(Tnew)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end%terminate the active learning for the (j+1)-th stage
    GPModel{j} = GPRmodel;
    gamma_fixed(j+1)=gamma_Active;
    DeltaGamma = gamma_fixed(j+1)-gamma_fixed(j);
    ZRef(j+1) = mean(TTestSamp.^gamma_Active);
    [TestMeanPred,TestSDPred,~] = predict(GPModel{j},TestSamp);
    Zfixed(j+1) = mean(exp(gamma_fixed(j+1)*TestMeanPred));%mean estimate of the model evidence of the (j+1)-th stage
    Z_CI(j+1,:) = [mean(exp(gamma_fixed(j+1)*(TestMeanPred-beta*TestSDPred))),mean(exp(gamma_fixed(j+1)*(TestMeanPred+beta*TestSDPred)))];%CI of the model evidence of the (j+1)-th stage
    PostPDF{j+1} = @(z)exp(gamma_fixed(j+1)*predict(GPModel{j},z)).*PriorPDF(z)/Zfixed(j+1);%%mean estimate of the normalzied posterior density of the (j+1)-th stage
    fprintf('Mean estimate of the intermediate model evidence of current tempering stage： %.8f\n', Zfixed(j+1));
    


%%%%%this section produces MCMC samples for the (j+1)-th stage using the R-MH algorithm reported as Algorithm 2 of the paper
    UnNormWeight = exp(gamma_fixed(j+1)*PostMeanSamp)./LikeSamp(:,j);%compute the weight of each sample 
    Weight = UnNormWeight/sum(UnNormWeight);%normalizing the weights
    SampInd = randsample(1:1:NMC,NMC,true,Weight);
    InitialSamp = MCSamp(SampInd,:,j);%resampling with replacement following the weights, and the resultant samples follow PostDensity{j+1}
    WeightedMean = sum(repmat(Weight,1,nz).* MCSamp(:,:,j),1);%weighted mean
    Deviation = sqrt(repmat(Weight,1,nz)).*(MCSamp(:,:,j) - repmat(WeightedMean,NMC,1));%weighted deviation of samples
    WeightedCOV = Deviation'*Deviation;% compute the covariance of samples
    PropSigma = zeta^2*WeightedCOV;% covariance matrix of the proposal distribution
    UnNormPDF = exp(gamma_fixed(j+1)*predict(GPModel{j},InitialSamp)).*PriorPDF(InitialSamp);
    clear UnNormWeight Weight SampInd_Record SampInd WeightedMean Deviation WeightedCOV
    for k=1:NMC
        Samp_kth_Chain{k} = InitialSamp(k,:);%the initial seed of the k-th chain
        PDF_kth_Chain{k} = UnNormPDF(k);
    end
    parfor k=1:NMC %for each sample, creat a chain of length Lchain
       for s=2:Lchain
           CandSamp = mvnrnd(Samp_kth_Chain{k}(s-1,:),PropSigma,1);%sampling from proposal distribution
           CandPDF = exp(gamma_fixed(j+1)*predict(GPModel{j},CandSamp)).*PriorPDF(CandSamp);
           AcceptProb = min(1,CandPDF/PDF_kth_Chain{k}(s-1));%%compute the acceptance rate
           u = rand;
           if AcceptProb>=u%accept "CandSamp" with probability "AcceptProb"
              Samp_kth_Chain{k}(s,:) = CandSamp;
              PDF_kth_Chain{k}(s,:) = CandPDF;
           else
              Samp_kth_Chain{k}(s,:) = Samp_kth_Chain{k}(s-1,:);
              PDF_kth_Chain{k}(s,:) = PDF_kth_Chain{k}(s-1,:);
           end
       end%terminate the k-th chain
    end%all chains produced
    for k=1:NMC% take the last state of each chain as the 
        MCSamp(k,:,j+1) = Samp_kth_Chain{k}(end,:);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%end of R-MH algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    LikeSamp(:,j+1) = exp(gamma_fixed(j+1)*predict(GPModel{j},MCSamp(k,:,j+1)));% compute the values of likeilihood scaled with gamma_fixed(j+1) for eahc produced samples
    clear Samp_kth_Chain PDF_kth_Chain InitialSamp

    %%%%%%for plotting the MCMC samples of eahc tempering stage
    % subplot(2,2,j)
    % scatter(MCSamp(:,1,j+1),MCSamp(:,2,j+1),'Marker','.','MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[231, 31, 24]/255);
    % xlim([-4,4])
    % ylim([-4,4])
    
    NcallEachLayer(j+1) = Ncall;% record the accumulated number of model calls for the (j+1)-th stage
    if gamma_fixed(j+1)>=1
        break;
    end
    j=j+1;
end
fprintf('Total number of model calls： %d\n', Ncall);
fprintf('Mean estimate of the model evidence： %.8f\n', Zfixed(j+1));
fprintf('gamma values:');
fprintf('%.4f  ',  gamma_fixed);
fprintf('\nCredible intervals of model evidence:');
fprintf('%.4f  ',  Z_CI(end,:));
fprintf('\nReference value of the model evidence： %.8f\n', ZRef(j+1));

filename = 'MCSampCase1.xlsx';
writematrix(MCSamp,filename)
writematrix(Ttrain,'TtrainCase1.xlsx')

for i=1:Nplot
    for k=1:j
        Postgrid(:,i,k) =PostPDF{k+1}([Xgrid(:,i),Ygrid(:,i)]);
    end
end



% compute the reference posterior density
for i=1:Nplot
   PostRef(:,i) = Likelihood([Xgrid(:,i),Ygrid(:,i)]).*unifpdf(Xgrid(:,i),-4,4).*unifpdf(Ygrid(:,i),-4,4)/ZRef(j+1);
end


C = [017 050 093
    034 065 110
    054 080 131
    080 092 142
    115 107 157 
    183 131 175
    210 150 135
    245 166 115
    252 219 114]/255;

figure
axes('Position',[0.1,0.73,0.38,0.22])
pcolor(Xgrid,Ygrid,Postgrid(:,:,1))
colormap(C)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('$\mathrm{(b).}\,\,j=2$','Fontsize',12,'Interpreter','latex')
plot(Ttrain(1:N0,1),Ttrain(1:N0,2),'d','LineStyle','none','MarkerSize',5,'MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[241 239 236]/255)
hold on
plot(Ttrain(N0+1:NcallEachLayer(2),1),Ttrain(N0+1:NcallEachLayer(2),2),'p','LineStyle','none','MarkerSize',5,'MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[241 239 236]/255)

axes('Position',[0.57,0.73,0.38,0.22])
pcolor(Xgrid,Ygrid,Postgrid(:,:,2))
colormap(C)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('$\mathrm{(b).}\,\,j=3$','Fontsize',12,'Interpreter','latex')

plot(Ttrain(NcallEachLayer(2)+1:NcallEachLayer(3),1),Ttrain(NcallEachLayer(2)+1:NcallEachLayer(3),2),'p','LineStyle','none','MarkerSize',5,'MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[241 239 236]/255)

axes('Position',[0.1,0.42,0.38,0.22])
pcolor(Xgrid,Ygrid,Postgrid(:,:,3))
colormap(C)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('$\mathrm{(c).}\,\,j=4$','Fontsize',12,'Interpreter','latex')

plot(Ttrain(NcallEachLayer(3)+1:NcallEachLayer(4),1),Ttrain(NcallEachLayer(3)+1:NcallEachLayer(4),2),'p','LineStyle','none','MarkerSize',5,'MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[241 239 236]/255)


axes('Position',[0.57,0.42,0.38,0.22])
pcolor(Xgrid,Ygrid,Postgrid(:,:,4))
colormap(C)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('$\mathrm{(d).}\,\,j=5,\,\, \gamma_5=1$','Fontsize',12,'Interpreter','latex')

plot(Ttrain(NcallEachLayer(4)+1:Ncall,1),Ttrain(NcallEachLayer(4)+1:Ncall,2),'p','LineStyle','none','MarkerSize',5,'MarkerEdgeColor',[231, 31, 24]/255,'MarkerFaceColor',[241 239 236]/255)



axes('Position',[0.1,0.1,0.38,0.22])
pcolor(Xgrid,Ygrid,PostRef)
colormap(C)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('(e). Target posterior','Fontsize',12,'Interpreter','latex')

axes('Position',[0.57,0.1,0.38,0.22])
pcolor(Xgrid,Ygrid,abs(PostRef-Postgrid(:,:,end)))
colormap(C)
colorbar
shading interp
hold on
xlabel('$\theta_1$','Interpreter','latex')
ylabel('$\theta_2$','Interpreter','latex')
title('(f). Absolute error','Fontsize',12,'Interpreter','latex')



function AcqFvalue = AcqFunTypeI(z,GPRmodel,beta,gamma1,PriorDen)
%% define the non-prospective acquisition function of second type
[MeanPred,STDPred,~] = predict(GPRmodel,z);
AcqFvalue = -(exp(gamma1*(MeanPred+beta*STDPred))-exp(gamma1*(MeanPred-beta*STDPred))).*PriorDen(z);
end

% function AcqFvalue = AcqFunTypeII(z,GPRmodel,beta,gamma1,PriorDen)
% %% define the non-prospective acquisition function of second type
% [MeanPred,STDPred,~] = predict(GPRmodel,z);
% AcqFvalue = -gamma1*(MeanPred+2*beta*STDPred)-log(PriorDen(z));
% end
