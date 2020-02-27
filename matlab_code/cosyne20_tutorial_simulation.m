% Simulation code for Cosyne 2020 Tutorial by Ann Hermundstad
% This script uses the following functions from the "matlab_code" folder: pmkmp.m, vmrand.m, circ_vmpdf.m
% Sashank Pisupati 02/02/2020
% Tested on Matlab R2019b

%% Environmental params
% Normalized color frequency
f = [0:1/99:1];
% Flower distribution
flowerDist = 'diffMeans';
% flowerDist = 'diffVars';


% Flower distribution parameters
switch flowerDist
    case 'diffMeans'
        envPars.field.pFlowerMu = 0.33;
        envPars.forest.pFlowerSig = 0.1;
        
        envPars.forest.pFlowerMu = 0.66;
        envPars.field.pFlowerSig = 0.1;
        
    case 'diffDist'
        envPars.field.pFlowerMu = 0.33;
        envPars.forest.pFlowerSig = 1;
        
        envPars.forest.pFlowerMu = 0.66;
        envPars.field.pFlowerSig = 0.1;
        
    case 'diffVars'
        envPars.field.pFlowerMu = 0.5;
        envPars.forest.pFlowerSig = 0.1;
        
        envPars.forest.pFlowerMu = 0.5;
        envPars.field.pFlowerSig = 0.5;
end

% Nectar function parameters
envPars.field.nectarMu = 0.5;
envPars.forest.nectarSig = 0.15;

envPars.forest.nectarMu = 0.5;
envPars.field.nectarSig = 0.15;

% Plot Flower distribution
figure(1);clf
set(gcf,'color','w');
subplot(3,2,1)
patch([0,f,f(end)],[0,pFlower(f,envPars,'field'),0],'r');hold on
patch([0,f,f(end)],[0,pFlower(f,envPars,'forest'),0],'b');box off
set(gca,'ytick','')
ylabel('probability P(f)')
xlabel('frequency f')
set(gca,'fontSize',12)
alpha(0.2)
title('Distribution of flowers')

% Plot nectar content
subplot(3,2,2)
plot(f,nectarFunc(f,envPars,'forest'),'b','LineWidth',1);hold on
plot(f,nectarFunc(f,envPars,'field'),'r','LineWidth',1);box off
set(gca,'ytick','')
ylabel('nectar content')
xlabel('frequency f')
set(gca,'fontSize',12)
alpha(0.2)
title('nectar content')




%% Part 1: Optimizing sensory encoding to maximize entropy
% Try changing flowerDist to 'diffVars' to see how it changes the results!

i = 2;
for env = {'forest','both'}
    % Max spike count
    beePars.(env{:}).kMax = 10;
    
    % Tuning parameters to maximize encoding entropy;
    params0=[0.5,10]+0.5*rand(1,2);
    paramsMaxEnt = fmincon(@(params)spikeNegEntropy(f,beePars.(env{:}).kMax,params,envPars,env{:}),params0);
    beePars.(env{:}).tuningOffset = paramsMaxEnt(1);
    beePars.(env{:}).tuningSlope = paramsMaxEnt(2);
    
    % % Plot sensory tuning curve
    subplot(3,2,2*i-1)
    hold on
    plot(f,tuningFun(f,beePars.(env{:})),'color','k','LineWidth',1);
    ylabel('Spike count')
    xlabel('frequency f')
    set(gca,'fontSize',12)
    ylim([0,beePars.(env{:}).kMax]);
    alpha(0.2)
    title(['Tuning optimized for ',env{:}])
    
    % % Plot spike count probability
    subplot(3,2,2*i)
    cla
    [counts,prob] = spikeProb(f,beePars.(env{:}),envPars,'forest');
    barh(counts,prob,'b')
    hold on
    [counts,prob] = spikeProb(f,beePars.(env{:}),envPars,'field');
    barh(counts,prob,'r');alpha(0.3);ylim([0.5,beePars.(env{:}).kMax+0.5]);set(gca,'ytick',[0,5,10]);
    set(gca,'fontSize',12)
    ylabel('Spike counts k')
    xlabel('probability p(k)')
    title('Spike count probability')
    i = i+1;
end
pause(2)



%% Part 2: Simulating online context inference
% "Regular" environments switch reliably every nSwitch trials, while
% "stochastic" ones switch every nSwitch trials *on average*
% Try changing flowerDist to 'diffVars' to see how it changes the results!

% Context switching dynamics
contextDyn = 'regular';
% contextDyn = 'stochastic';

tMax = 200; % Number of timesteps
nSwitch = 10; % Average number of encounters per context
h = 1/nSwitch; %Hazard rate in units of 1/encounters
T = [1-h, h; h, 1-h]; %Transition matrix
context = zeros(1,tMax);
fObs = zeros(1,tMax);

% Simulate context switching dynamics
env = {'field','forest'};
switch contextDyn
    case 'regular'
        %Periodically switching contexts
        for i = 1:round(tMax*h)/2
            context((2*i-1)*nSwitch+1:2*i*nSwitch)=1;
        end
    case 'stochastic'
        %Stochastic contexts with hazard rate h
        for i = 2:tMax
            if rand<h
                context(i) = ~context(i-1);
            else
                context(i) = context(i-1);
            end
        end
end

% Plot context switching dynamics
figure(2);clf
set(gcf,'color','w');
subplot(3,2,[1,2])
imagesc(-context);colormap(jet);alpha(0.8)
title('Context')
xlabel('No. of encounters')
set(gca,'ytick','')

% Initialize beliefs
p0 =[0.5;0.5];
pContext = zeros(2,tMax);
pContext(:,1) = p0;

%Perceptual inference
for t = 1:tMax
    % True context/environment
    envTrue = env{context(t)+1};
    % Prior before observation
    if t >1
        pContext(:,t) = T*pContext(:,t-1); %Apply transition matrix
    end
    % Observed color
    fObs(t) = normrnd(envPars.(envTrue).pFlowerMu,envPars.(envTrue).pFlowerSig);
    % Likelihood of observation
    lik = [pFlower(fObs(t),envPars,env{1});pFlower(fObs(t),envPars,env{2})];
    % Posterior after observation
    pContext(:,t) = pContext(:,t).*lik;
    pContext(:,t) = pContext(:,t)/sum(pContext(:,t));
end

% Plot running posterior
subplot(3,2,[3,4])
plot(pContext(2,:),'b','linewidth',1)
title('Posterior belief')
xlabel('No. of encounters')
ylabel('p(Forest)')


% Average posterior before & after forest->field switch
indsField = strfind(context,[1 1 1 1 1 0 0 0 0 0]);
pAvgField = zeros(1,10);
for i = 1:length(indsField)
    pAvgField = pAvgField + pContext(2,indsField(i):indsField(i)+9);
end
pAvgField = pAvgField/length(indsField);

% Plot average posterior around forest->field switches
subplot(3,2,5)
plot(pAvgField,'k-','linewidth',1)
ylim([0,1])
xlim([1,nSwitch])
title('Average p(Forest->Field)')
xlabel('No. of encounters')
ylabel('p(Forest)')
hold on
plot([5,5],[0,1],'k--')
hold off

% Average posterior before & after field->forest switch
indsForest = strfind(context,[0 0 0 0 0 1 1 1 1 1]);
pAvgForest = zeros(1,10);
for i = 1:length(indsForest)
    pAvgForest = pAvgForest + pContext(2,indsForest(i):indsForest(i)+9);
end
pAvgForest = pAvgForest/length(indsForest);

% Plot average posterior around field->forest switches
subplot(3,2,6)
plot(pAvgForest,'k-','linewidth',1)
hold on
plot([5,5],[0,1],'k--')
ylim([0,1])
xlim([1,nSwitch])
title('Average p(Field->Forest)')
xlabel('No. of encounters')
ylabel('p(Forest)')
hold off

pause(2)



%% Part 3: Simulating action policy & learning rule
% Try simulating for 20 repetitions of 20 timesteps each - alternatively
% try simulating for 1 repetition of 200 timesteps

tMax = 20; % Number of timesteps per trial
nReps = 20; %Number of trials/repetitions
showSim =1; % show step-by-step evolution
learning = 1; % allow stim->value weights to be learnt
contextSwitch = 0; % allow unsignaled context switch
linewidth = 0.1;
linecolor = [0 0 0];

% Bee RL parameters
beePars.valueFuncType = 'linear'; % Type of value function approximator
beePars.policyFuncType = 'both'; % Type of turns allowed in policy
beePars.softmaxTemp = 20;%Exploratoriness or "temperature"
beePars.softmaxThreshold = 0;%Point of value indifference
beePars.reorientMu = pi/2;%Mean reorienting angle
beePars.reorientKappa = 10;%cConcentration of reorienting angles
beePars.stepSize = 0.5;%Size of one movement
beePars.discountGamma = 0.6; %Temporal discounting factor "gamma"
beePars.learningRate = 0.5; %Learning rate "alpha"
init = vmrand(0,0.1); % initial orientation


% Initial/known weights
switch beePars.valueFuncType
    case 'linear' %Linear approximation
        if ~learning
            beePars.forest.w = [0 1];
            beePars.field.w = [1 -1];
        else
            beePars.forest.w = [0 0];
            beePars.field.w = [0 0];
        end
    case 'rbf' %Radial basis function approximation
        if ~learning
            beePars.forest.w = [0 1 2 3 4 5]/15;
            beePars.field.w = [5 4 3 2 1 0]/15;
        else
            beePars.forest.w = [0 0 0 0 0 0];
            beePars.field.w = [0 0 0 0 0 0];
        end
end


figure(3);clf;
set(gcf,'color','w');
% Simulate 2d environment with continuous frequency changesß
[X,Y] = meshgrid(linspace(-3,3,100));
F = -(X.^5-Y.^5-2*X*Y-2*X.^3-3*Y.^3).*(1.1*exp(-0.4*(X-1.5).^2-0.5*(Y-1.5).^2)-0.7*exp(-(X+2).^2-(Y+2).^2-0.8.*(X+2).*(Y+2))-0.7*exp(-(X+2).^2-0.5*(Y-2).^2+0.3.*(X+2).*(Y-2))-0.5*exp(-3*(X-2).^6-4*(Y+2).^2+4.*(X-2).*(Y+2).^2));
F = 1-(F- min(min(F)))/(max(max(F))-min(min(F)));

% Plot probability of reorienting
subplot(4,2,1)
plot([-1:0.01:1],1./(1+exp(beePars.softmaxTemp*([-1:0.01:1]-beePars.softmaxThreshold))),'k-','LineWidth',1)
xlabel('Change in value')
ylabel('p (Reorient)')
set(gca,'fontSize',12)
title('Policy - reorienting probability')

% Plot distribution of angle changes if reorienting  - requires circ_vmpdf.m
subplot(4,2,2)
patch(0:0.01:pi,circ_vmpdf(0:0.01:pi,beePars.reorientMu,beePars.reorientKappa),[0.5 0.5 0.5],'LineWidth',1)
xlim([0,pi])
set(gca,'XTick',[0 pi/2 pi],'XTickLabel',[0 90 180])
xlabel('Change in orientation')
ylabel('p (\Deltatheta)')
set(gca,'fontSize',12)
title('Policy - reorienting angle')

env = {'field','forest'};
colors = {'r','b'};

% Loop over contexts
for e = 1:2
    % Context specific weights? Otherwise unsignalled change from field->forest
    if ~contextSwitch
        w = beePars.(env{e}).w;
    end
    
    %Plot initial/known value estimate
    subplot(4,2,2+e)
    plot(f,valueFunc(f,w,beePars.valueFuncType),'-','color',colors{e},'LineWidth',1)
    xlabel('Frequency')
    ylabel('Value')
    set(gca,'fontSize',12)
    ylim([0,1])
    xlim([0,1])
    title(['Value in ',env{e}])
    
    % Loop over repetitions/trials
    for rep = 1:nReps
        x = zeros(2,tMax);
        y = zeros(2,tMax);
        thetaT = zeros(1,tMax);
        deltaTheta = zeros(1,tMax);
        
        %Plot environment - uses pmkmp.m for nice colors
        subplot(4,2,[4+e,6+e])
        cla
        contourf(X,Y,F,10)
        set(gca,'ytick','')
        set(gca,'xtick','')
        map = pmkmp(256,'CubicL');
        colormap(flip(map))
        set(gca,'FontSize',15)
        xlim([-3,3])
        ylim([-3,3])
        hold on
    
        
        % Initial location [x(1,1),y(1,1)]
        x(1,1) = 0;
        y(1,1) = 0;
        % Initial orientation i.e. prospective location [x(2,1),y(2,1)]-[x(1,1),y(1,1)]
        % - requires vmrand.m
        if strcmp(init, 'rand')
            thetaInit = vmrand(0,0.1);
        else
            thetaInit = init;
        end
        r = beePars.stepSize;
        thetaT(1) = thetaInit;
        deltaTheta(1) = 0;
        x(2,1) = x(1,1) + r*cos(thetaT(1));
        y(2,1) = y(1,1) + r*sin(thetaT(1));
        
        % Initial (discretized) indices on grid
        iCurr = find(X(1,:)<x(1,1),1,'last');
        jCurr = find(Y(:,1)<y(1,1),1,'last');
        iNext = find(X(1,:)<x(2,1),1,'last');
        jNext = find(Y(:,1)<y(2,1),1,'last');
        
        % Draw starting orientation
        subplot(4,2,[4+e,6+e])
        drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0);
        h = drawArrow(x(:,1),y(:,1));
        set(h,'color','b','LineWidth',3,'MaxHeadSize',1)
        
        
        % Loop over timesteps
        for t = 2:tMax
            
            % Planning : Compute expected value at current, prospective location
            vCurr = valueFunc(F(iCurr,jCurr),w,beePars.valueFuncType);
            vNext = valueFunc(F(iNext,jNext),w,beePars.valueFuncType);
            
            % Action selection: Choose to continue/reorient based on policy
            % replanning as needed to avoid boundaries
            plan = 1;
            while plan
                deltaTheta(t) = policyFunc(vNext,vCurr,beePars);
                thetaT(t) = thetaT(t-1)+ deltaTheta(t);
                
                % New location
                x(1,t) = x(1,t-1) + r*cos(thetaT(t));
                y(1,t) = y(1,t-1) + r*sin(thetaT(t));
                % New orientation/prospective location
                x(2,t) = x(1,t) + r*cos(thetaT(t));
                y(2,t) = y(1,t) + r*sin(thetaT(t));
                % New indices on grid
                iCurr = find(X(1,:)<x(1,t),1,'last');
                jCurr = find(Y(:,1)<y(1,t),1,'last');
                iNext = find(X(1,:)<x(2,t),1,'last');
                jNext = find(Y(:,1)<y(2,t),1,'last');
                
                % Replan if boundary encountered in current/prospective location
                if isempty(iNext) || iNext == length(F) || isempty(jNext) || jNext == length(F)
                    plan = 1;
                    vNext = -inf;
                else
                    plan = 0;
                end
            end
            
            % Plot movement to new location
            subplot(4,2,[4+e,6+e])
            hold on
            hMove = drawArrow([x(1,t-1),x(1,t)],[y(1,t-1),y(1,t)]);
            set(hMove,'color',linecolor,'LineWidth',linewidth,'MaxHeadSize',1)
            if showSim
                drawnow
            end
            
            % Observe sensory outcome of action, re-evaluate current location
            vPrev = vCurr;
            vCurr = valueFunc(F(iCurr,jCurr),w,beePars.valueFuncType);
            
            % Observe reward outcome of action, update weights based on learning rule
            reward = nectarFunc(F(iCurr,jCurr),envPars,env{e});
            
            if learning && ~isempty(vNext)
                % Temporally discounted reward prediction error
                delta = reward+beePars.discountGamma*vCurr-vPrev;
                switch beePars.valueFuncType
                    case 'linear'
                        % Gradient of value w.r.t. weights
                        gradfun = [1; F(iCurr,jCurr)];
                    case 'rbf'
                        gradfun = radbas([5*F(iCurr,jCurr);5*F(iCurr,jCurr)-1;5*F(iCurr,jCurr)-2;5*F(iCurr,jCurr)-3;5*F(iCurr,jCurr)-4;5*F(iCurr,jCurr)-5]);
                end
                % Weight update using Temporal Difference learning
                w = w + delta*beePars.learningRate*gradfun';
            end
            
            % Plot updated weights
            subplot(4,2,2+e)
            plot(f,valueFunc(f,w,beePars.valueFuncType),'--','color',colors{e},'LineWidth',1)
            xlabel('Frequency')
            ylabel('Value')
            set(gca,'fontSize',12)
            ylim([0,3])
            xlim([0,1])
            title(['Value in ',env{e}])
            
        end
        
    end
    % Learnt weights
    beePars.(env{e}).w = w;
end

%% PDF plotting help
hglobal = gcf;
hglobal.PaperPositionMode = 'auto';
fig_pos = hglobal.PaperPosition;
hglobal.PaperSize = [1.1*fig_pos(3) 1.1*fig_pos(4)];

%% Functions
% Distribution of flowers
function p = pFlower(f,envPars,env)
switch env
    case {'field','forest'}
        p = normpdf(f,envPars.(env).pFlowerMu,envPars.(env).pFlowerSig);
    case 'both'
        %         p = 0.5*normpdf(f,envPars.forest.pFlowerMu,envPars.forest.pFlowerSig)+0.5*normpdf(f,envPars.field.pFlowerMu,envPars.field.pFlowerSig);
        sig(:,:,1) = (envPars.field.pFlowerSig)^2;
        sig(:,:,2) = (envPars.forest.pFlowerSig)^2;
        mu = [envPars.field.pFlowerMu;envPars.forest.pFlowerMu];
        pEnv = [0.5 0.5];
        gm = gmdistribution(mu,sig,pEnv);
        p = pdf(gm,f');
end
% p = p/sum(p);
end

% Cumulative distribution of flowers
function c = cFlower(f,envPars,env)
switch env
    case {'field','forest'}
        c = normcdf(f,envPars.(env).pFlowerMu,envPars.(env).pFlowerSig);
    case 'both'
        %         p = 0.5*normpdf(f,envPars.forest.pFlowerMu,envPars.forest.pFlowerSig)+0.5*normpdf(f,envPars.field.pFlowerMu,envPars.field.pFlowerSig);
        sig(:,:,1) = (envPars.field.pFlowerSig)^2;
        sig(:,:,2) = (envPars.forest.pFlowerSig)^2;
        mu = [envPars.field.pFlowerMu;envPars.forest.pFlowerMu];
        pEnv = [0.5 0.5];
        gm = gmdistribution(mu,sig,pEnv);
        c = cdf(gm,f');
end
end

% Nectar function
function n = nectarFunc(f,envPars,env)
switch env
    case 'field'
        n = 1-normcdf(f,envPars.(env).nectarMu,envPars.(env).nectarSig);
    case 'forest'
        n = normcdf(f,envPars.(env).nectarMu,envPars.(env).nectarSig);
end
end

% Sensory coding function
function counts = tuningFun(f,beePars)
% rate = beePars.kMax*normcdf(f, beePars.tuningOffset,1/beePars.tuningSlope);
counts = round(beePars.kMax*normcdf(f, beePars.tuningOffset,1/beePars.tuningSlope));
end


% Spike count probability
function [cbin,pbin] = spikeProb(f,beePars,envPars,env)

% First, get counts as a function of f
counts = tuningFun(f,beePars);
cbin = unique(counts)';

% Next, get unique frequency bins using inverse tuning function
fbin = norminv([counts/beePars.kMax;(counts-1)/beePars.kMax], beePars.tuningOffset,1/beePars.tuningSlope);
fbin(isinf(fbin)&fbin>0)=1;
fbin(fbin<0)=0;
fbin(isnan(fbin))=0;
fbin = unique(fbin','rows');

% Finally, get probability mass in each bin
pbin = cFlower(fbin(:,1)',envPars,env)-cFlower(fbin(:,2)',envPars,env);
pbin = pbin/sum(pbin);
end


% Sensory coding (negative) entropy
function i = spikeNegEntropy(f,kMax,params,envPars,env)
% parameters
beePars.kMax = kMax;
beePars.tuningOffset = params(1);
beePars.tuningSlope = params(2);

% Spike count probability
[~,pbin] = spikeProb(f,beePars,envPars,env);

% Negative Entropy of encoding
i = sum(pbin(pbin>0).*log2(pbin(pbin>0)));
end


% Value coding function
function v = valueFunc(f,w,fun)
if ~exist('fun','var')
    fun = 'linear';
end

% Boundary condition
if isempty(f)
    v = -inf;
else
    switch fun
        case 'linear' %Linear basis
            v = w(2)*f + w(1);
        case 'rbf' %Radial basis
            h = radbas([5*f; 5*f-1;5*f-2;5*f-3;5*f-4;5*f-5]);
            v = w*h;
    end
end

end


% Policy function
function deltaTheta = policyFunc(vT,vTminus1,beePars)
% Expected value difference
delV = vT-vTminus1;

% Softmax decision to reorient
pReorient = 1/(1+exp(beePars.softmaxTemp*(delV-beePars.softmaxThreshold)));

if rand<pReorient
    % Von mises distributed reorientation angles if reorienting
    switch beePars.policyFuncType
        case 'cw'
            % Only Clockwise turn
            deltaTheta = vmrand(beePars.reorientMu,beePars.reorientKappa);
        case 'ccw'
            % Only Counterclockwise turn
            deltaTheta = -vmrand(beePars.reorientMu,beePars.reorientKappa);
        case 'both'
            % Randomly pick between clockwise and counterclockwise turns
            deltaTheta = sign(rand-0.5)*vmrand(beePars.reorientMu,beePars.reorientKappa);
    end
else
    % Continue straight if not reorienting
    deltaTheta = 0;
end
end


