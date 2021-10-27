% LORENZ system
% System identification: DelayDMDc

clear all, close all, clc

 
%% importing Data
% pathold='/Users/shaurya/Docs/Robo Track Competition/training_trajectories/great-bipedal-bumblebee_';
pathold='/Users/shaurya/Docs/Robo Track Competition/great-bipedal-bumblebee_all.csv';
datan=importdata(pathold);
d=datan.data;
x_state=d(201:400,2:3);
dx_state=d(201:400,4:5);
u=d(201:400,10:11);


%% SINDYc
% Parameters
dt=0.01;
ModelName = 'SINDYc'; iModel = 2;
Nvar = 2;  % no of state variables
polyorder = 2;
usesine = 0;
lambda = 0.1;
            
trainSINDYc


%% Prediction  over training phase
% if any(strcmp(InputSignalType,{'sine2', 'chirp','prbs', 'sphs'})==1)
%     [tSINDYc,xSINDYc]=ode45(@(t,x)sparseGalerkinControl(t,x,forcing(x,t),Xi(:,1:Nvar),polyorder,usesine),tspan,x0,options);  % approximate
% else 
p.ahat = Xi(:,1:Nvar);
p.polyorder = polyorder;
p.usesine = usesine;
p.dt = 0.01;%dt;
[N,Ns] = size(x);   % [10000,4]
xSINDYc = zeros(Ns,N); xSINDYc(:,1) = x(1,:);
for ct=1:N-1
    xSINDYc(:,ct+1) = rk4u(@sparseGalerkinControl_Discrete,xSINDYc(:,ct),u(ct,:),dt,1,[],p);
end
xSINDYc = xSINDYc';
% end

% %% Show validation
% tspan=0:dt:2;
% clear ph
% figure,box on,
% ccolors = get(gca,'colororder');
% ccolors_valid = [ccolors(1,:)-[0 0.2 0.2];ccolors(2,:)-[0.1 0.2 0.09];ccolors(3,:)-[0.1 0.2 0.09]];
% for i = 1:Nvar
%     ph(i) = plot(tspan,x(:,i),'-','Color',ccolors(i,:),'LineWidth',1); hold on
% end
% for i = 1:Nvar
%     ph(Nvar+i) = plot(tspan,xSINDYc(:,i),'--','Color',ccolors_valid(i,:),'LineWidth',2);
% end
% xlim([0 (length(tspan)-1)*dt]), ylim([-25 50])
% xlabel('Time')
% ylabel('xi')
% legend(ph([1,4]),'True',ModelName)
% set(gca,'LineWidth',1, 'FontSize',14)
% set(gcf,'Position',[100 100 300 200])
% set(gcf,'PaperPositionMode','auto')
% print('-depsc2', '-loose', '-cmyk', [figpath,'EX_',SystemModel,'_SI_',ModelName,'_',InputSignalType,'.eps']);

% %% Validation 3D
% filename = ['EX_',SystemModel,'_SI_',ModelName,'_',InputSignalType,'_train'];
% xModel = xSINDYc;
% xTRUTH = x;
% color_type = 'models';
% VIZ_3D_MODELvsTRUTH

% %% Prediction
% % Reference
% tspanV   = [10:dt:20];
% xA      = xv;
% tA      = tv;

% % Model
% if any(strcmp(InputSignalType,{'sine2', 'chirp','prbs', 'sphs'})==1)
%     [tB,xB]=ode45(@(t,x)sparseGalerkinControl(t,x,forcing(x,t),Xi(:,1:Nvar),polyorder,usesine),tspanV,x(end,:),options);  % approximate
%     xB = xB(2:end,:);
%     tB = tB(2:end); % rm IC
% else
%     [N,Ns] = size(xA);
%     xB = zeros(Ns,N); xB(:,1) = x(end,:)';
%     for ct=1:N
%         xB(:,ct+1) = rk4u(@sparseGalerkinControl_Discrete,xB(:,ct),uv(ct),dt,1,[],p);
%     end
%     xB = xB(:,1:N+1)';
%     tB = tspanV(1:end);
% end

% %% Prediction
% filename = ['EX_',SystemModel,'_SI_',ModelName,'_',InputSignalType,'_valid'];
% xModel = xB;
% xTRUTH = xA;
% color_type = 'models';
% VIZ_3D_MODELvsTRUTH

% %% Show training and prediction
% VIZ_SI_Validation

% %% Save Data
% Model.name = 'SINDYc';
% Model.polyorder = polyorder;
% Model.usesine = usesine;
% Model.Xi = Xi;
% Model.dt = dt;
% save(fullfile(datapath,['EX_',SystemModel,'_SI_',ModelName,'_',InputSignalType,'.mat']),'Model')
tspan = 0:dt:2-dt;
i=1;
plot(tspan,xSINDYc(:,1),'r',tspan,x(:,1),'b')