
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear

%% Simulation setting
Tsim = 5;      %simulation time length

%% Training settings
Niter = 1000;         % Number of learning iterations 
learning_rate = 0.01;     % learning rate to update W 0.0001
r = 0.01;               % weighting factor for DU 0.0001
Tsettling = 3;          % settling time
u_lim = [-12 12];       % input voltage limit

ts = 1/1000;

%------------------------------------------
% Initial condition for W = [w1 w2]'

Wo = 0.01*(rand(4,1));

%% Step reference 
pos_ref = pi/12;        % position step reference in rad
Tstep = 1;              % step reference instant

%% Plot settings
plot_iterations = 1;    % 1: plot learning performance
plot_interval = [1 50];% interval to be plotted while training
font_size = 16;         % font size for plots

%%  System Parameters
g  = 9.81;

Rm = 8.4;
kt = 0.042;
km = 0.042;
Jm = 4.0e-6;
Jh = 0.6e-6;

md = 0.053;
rd = 0.0248;

mr = 0.095;
Lr = 0.085;

mp = 0.024;
Lp = 0.129;

lr = Lr/2;
lp = Lp/2;

Jr_arm = (1/3)*mr*Lr^2;
Jd     = 0.5*md*rd^2;
Jr     = Jm + Jh + Jd + Jr_arm;

Jp     = (1/3)*mp*Lp^2;

Br = 0;
Bp = 0;

Ku = kt/Rm;
Be = kt*km/Rm;

Jt    = Jr + mp*Lr^2;
Delta = Jt*Jp - (mp*Lr*lp)^2;

Ac = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, (mp^2*g*lp^2*Lr)/Delta, -(Jp*(Br+Be))/Delta,  (mp*Lr*lp*Bp)/Delta;
     0,  -(mp*g*lp*Jt)/Delta,      (mp*Lr*lp*(Br+Be))/Delta, -(Jt*Bp)/Delta];

Bc = [0;
     0;
     (Jp*Ku)/Delta;
    -(mp*Lr*lp*Ku)/Delta];

Cc = [1 0 0 0;
     0 1 0 0];

ct_model = ss(Ac,Bc,Cc,0);

dt_model = c2d(ct_model,ts);

A = dt_model.A;
B = dt_model.B;
C = dt_model.C;


%% ANN Training data length 
time = 0:ts:Tsim;
N = length(time);
inv_N = 1/N;

%% prediction model for X_[2:N], Y_[2:N], and DU_[1:N-1]
[Phi,Phi_y,S] = prediction_model(A,B,C,N-1);

%% Step reference
Nstep = round(Tstep/ts);
Nstep = max(Nstep,1);
if (Nstep>N)
    Nstep = N;
end
% y1_ref = [0 0 ... 0 ref ref ... ref]'
y1_ref = [zeros(Nstep-1,1); pos_ref*ones(N-Nstep+1,1)];

% y2_ref = [0 0 ... 0 ... 0]'
y2_ref = zeros(N,1);

y_ref = [y1_ref'; y2_ref'];

% x_ref = [y1_ref(1) y1_ref(2) ... y1_ref(N) %position ref
%             0        0       ...   0      
%          y2_ref(1) y2_ref(2) ... y2_ref(N) %angle ref
%             0        0     ...   0      ]
x_ref = [y1_ref'; y2_ref'; zeros(1, N);  zeros(1, N)];

%% Desired Exponential convergence of the system output
%  Y*_[2:N] = [y*(2) y*(3) ... y*(N)]'
y1_ref_filtered =  first_order_filter(y1_ref, ts, Tsettling/5);

y_ref_filtered = [y1_ref_filtered'; y2_ref'];
Yref = reshape(y_ref_filtered(:,2:N),[],1);

%% ANN Training (Gradient descendant algorithm)
disp('ANN Training...')
W = Wo; %initialization 
J = zeros(1,Niter);
n =0 ;
while (n<Niter)
   n = n+1;
    % Predictions: run SFC-based closed-loop system model
    K_est = -W';
    [x,y,u] = run_SFC_CL_system(A,B,C,K_est,x_ref,u_lim);

    %% ANN Input data
    xe = x-x_ref;
    Xe = xe(:,1:N-1);

    %% ANN Output data
    Y_hat = reshape(y(:,2:N),[],1); % Y_hat_[2:N] = [y(2); y(3); ... y(N)]

    U_hat = u(1:N-1)';      % U_[1:N-1]  = [u(1) u(2) ... u(N-1)]'
    DU_hat= S*U_hat;        % DU_[1:N-1] = [Du(1) Du(2) ... Du(N-1)]'

    %% Loss function
    Jy  = inv_N*(Y_hat - Yref)'*(Y_hat - Yref);
    Jdu = inv_N*DU_hat'*DU_hat;
    J(n) = Jy + r*Jdu;

    %% Gradient of J w.r.t. y
    % replace this accordingly 
    DWy  = 2/N*Xe*Phi_y'*(Y_hat-Yref);
    
    %% Gradient of J w.r.t. DU
    % replace this accordingly 
    DWdu = 2/N*Xe*S'*DU_hat;
    
    %% W Update
    DW = DWy + r*DWdu;
    W = W - learning_rate*DW;

    % plot each iteration
    if (plot_iterations == 1 && n>=plot_interval(1) && n <=plot_interval(2))        
        plot_results;                                                                                                                                                                                                                        
    end
end

%% final plots
%plot_results;

%% Results
disp('********************************************')
disp('Resuls:')
disp(['Initial Estimate: K_ini = [', num2str(-Wo(1)), '  ', num2str(-Wo(2)),']'])
disp(['Final Estimate: K_est = [', num2str(K_est(1)), '  ', num2str(K_est(2)),']'])
disp(['Final loss value: ', num2str(J(n))])

%% Verification
xo = [0 0 0 0]';
sim('sim_ICS2026_inv_pendulum.slx')

%% Verification plot
% add your final plot here
figure(901)
plot(time,x(:,1),'b',time,x_ref,'--k')
ylabel('$x_c(k) [m]$', 'Interpreter', 'latex')
xlabel('Time [sec]')
legend({'$x_c(k)$', '$x_{\mathrm{ref}}$'}, ...
       'Interpreter', 'latex', ...
       'Location', 'best')

