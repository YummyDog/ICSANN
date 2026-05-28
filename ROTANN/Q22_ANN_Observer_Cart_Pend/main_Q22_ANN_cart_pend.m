% Assessment Task 5 
% Assignment 3 - Q2.2) ANN-based Observer design
% 
%
% 48580 Intelligent Control Studio
% University of Technology Sydney, Australia
% Autumn 2026
%
% A/Prof Ricardo P. Aguilera
%
% Enjoy it!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear

%% Simulation settings
Tsim = 3;      %simulation time length
ts = 1/1000;        %controller sampling time

% use estimated control gain from Q2.1)
K_est = 1.0e+03 * [-0.1517   -1.1774   -0.0842   -0.0921];

%% Noise setting
pos_noise_var = 1e-4;
alpha_sensor_var = 1e-4;
add_noise = 1;
    
%% Observer setting and design targets
Niter = 100;           % Number of learning iterations 
learning_rate = 0.1;   % learning rate to update W 0.0001

%% Observer design targets
Tsettling = 0.3;            % estimation settling time

%% Step reference 
pos_ref = pi/12; % position step reference in rad
Tstep = 1;      % step reference instant
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
     0, -(mp^2*g*lp^2*Lr)/Delta, -(Jp*(Br+Be))/Delta,  (mp*Lr*lp*Bp)/Delta;
     0,  (mp*g*lp*Jt)/Delta,      (mp*Lr*lp*(Br+Be))/Delta, -(Jt*Bp)/Delta];

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

%% Initial conditions
xc_o = pos_ref;
alpha_o = 0;
xo = [0 0 0 0]';
xo_hat = [0 0 0 0]';
%------------------------------------------
% Initial condition for W = [w1 w1]'
W1_o = [0.246301777945478 -0.013280510405406]';
W2_o = [-0.007465290689340 0.323673922377557]';
W3_o = [2.654105268984229 -0.635158375296772]';
W4_o = [0.026269868368885  4.927945666386226]';
Kf_o = [W1_o'; W2_o'; W3_o'; W4_o'];

%% Plot settings
plot_iterations = 1;    % 1: plot learning performance
plot_interval = [1 1];% interval to be plotted while training
font_size = 16;         % font size for plots

%%  System Parameters
% these parameters will depend on your student number


%% Data length 
time = 0:ts:Tsim;
N = length(time);
inv_N = 1/N;

%% prediction model for X_[2:N], Y_[2:N], and Vi_[1:N-1]
[Psi_1, Psi_2,Psi_3,Psi_4,S] = prediction_model_observer(A,B,C,N-1);

%% Step reference
Nstep = round(Tstep/ts);
if (Nstep >N)
    Nstep = N;
end
% y_ref = [0 0 ... 0 ref ref ... ref]'
y1_ref = [pos_ref*ones(Nstep,1);  2*pos_ref*ones(N-Nstep,1)];
% x_ref = [y_ref(1) y_ref(2) ... y_ref(N)
%             0        0     ...   0      ]; angular speed ref = 0
x_ref = [y1_ref.'; zeros(1, N); zeros(1, N); zeros(1, N)];

%% Run closed-loop system
[x,y,u] = run_SFC_CL_system(A,B,C,K_est,x_ref,xo);
Y = reshape(y(2:N), [], 1);

%% sensor noise
pos_noise = add_noise*sqrt(pos_noise_var)*randn(1, N);
alpha_noise =  add_noise*sqrt(alpha_sensor_var)*randn(1, N);
y_noise = [pos_noise; alpha_noise];

%% output target
Nset = round(Tsettling/ts); % sample when settling time is achieved 

y1_target = y(1,:);
y1_target(1:Nset) = first_order_filter(y1_target(1:Nset),ts,Tsettling/5);

y2_target = y(2,:);
y2_target(1:Nset) = first_order_filter(y2_target(1:Nset),ts,Tsettling/5);

y_target = [y1_target; y2_target];
Y_target = reshape(y_target(:,2:N), [], 1);

%% output target - use actual noisy measurements directly
%y_noisy_full = y + y_noise;
%Y_target = reshape(y_noisy_full(:, 2:N), [], 1);

%% state target - use actual states directly (no filtering)
%X_target = reshape(x(:, 2:N), [], 1);

%% state target
x_target = x;
x_target(1,:) = y1_target; % replace x1 by y1_filtered
X_target = reshape(x_target(:,2:N), [], 1); % X_[2:N] = [x(2); x(3); ...; x(N) ]

%% ANN Training (Gradient descendant algorithm)
disp('ANN Training...')
W1 = W1_o; %initialization 
W2 = W2_o; %initialization 
W3 = W3_o; %initialization 
W4 = W4_o; %initialization 
J = zeros(1,Niter);

count_plot = 0;
n =0 ;
while (n<Niter)
    n = n+1;
    %% Predictions: run observer to generate data
    Kf_est = [W1'; W2'; W3'; W4'];
    [x_hat,y_hat,v_hat] = run_observer(A,B,C,Kf_est,x,u,xo_hat,y_noise);

    %% ANN input Ye = [y_e(1) y_e(2) ... y_e(N-1)]
    y_noisy = y + y_noise;
    Ye = y_noisy(:,1:N-1)-y_hat(:,1:N-1);

    %% Observer output X_hat_[2:N] = [x_hat(2); x_hat(3);...; x_hat(N)]
    X_hat = reshape(x_hat(:,2:N),[],1); 

    %% Observation error
    Xe_hat = X_hat - X_target;

    %% Loss function
    Jx   = inv_N*Xe_hat'*Xe_hat ;
    J(n) = Jx;

    %% Gradient decent:

    %% Gradient DJx/DW1
    %-------------------------

    dJx_dW1 =  2/N*Ye*Psi_1'*Xe_hat;

    %% Gradient DJx/DW2
    %-------------------------

    dJx_dW2 =  2/N*Ye*Psi_2'*Xe_hat;

    %% Gradient DJx/DW3
    %-------------------------

    dJx_dW3 =  2/N*Ye*Psi_3'*Xe_hat;

    %% Gradient DJx/DW4
    %-------------------------

    dJx_dW4 =  2/N*Ye*Psi_4'*Xe_hat;
    
    %-------------------------

    grad_max = 1;

    %dJx_dW1 = max(min(dJx_dW1,grad_max),-grad_max);
    %dJx_dW2 = max(min(dJx_dW2,grad_max),-grad_max);
    %dJx_dW3 = max(min(dJx_dW3,grad_max),-grad_max);
    %dJx_dW4 = max(min(dJx_dW4,grad_max),-grad_max);



    DW1_x = dJx_dW1;
    DW2_x = dJx_dW2;
    DW3_x = dJx_dW3;
    DW4_x = dJx_dW4;

    

    %% W Update
    W1 = W1 - learning_rate*DW1_x; 
    W2 = W2;
    W3 = W3;
    W4 = W4;

    % plot each iteration
    if (J(n)>1e10 || isnan(J(n)))
        %re-start W
        W1 = W1_o;
        W2 = W2_o;
        W3 = W3_o;
        W4 = W4_o;
        n = 0;
    elseif (plot_iterations == 1)        
        if (n>=plot_interval(1) && n <=plot_interval(2))
            plot_results
            drawnow  
        end
        count_plot = count_plot + 1;
        if count_plot==round(Niter/100)
            plot_results
            drawnow  
            count_plot = 0;
        end
        
    end
    
end
Kf_est = [W1'; W2'; W3'; W4'];

%% final plots
plot_results
drawnow

%% Results
disp('********************************************')
disp('Resuls:')
disp(' ')
disp(['Initial Estimate: Kf_ini = [', num2str(Kf_o(1,:))])
disp(['                            ', num2str(Kf_o(2,:))])
disp(['                            ', num2str(Kf_o(3,:))])
disp(['                            ', num2str(Kf_o(4,:)),']'])
disp(' ')
disp(['Final Estimate:   Kf_est = [', num2str(Kf_est(1,:))])
disp(['                            ', num2str(Kf_est(2,:))])
disp(['                            ', num2str(Kf_est(2,:))])
disp(['                            ', num2str(Kf_est(2,:)),']'])
disp(' ')
disp(['Final loss value: ', num2str(J(n))])
eig_AL = eig(A-Kf_est*C);
disp(['Observer closed-loop poles: [',num2str(eig_AL(1)),' ', ...
    num2str(eig_AL(2)),' ',num2str(eig_AL(3)),' ', ...
    num2str(eig_AL(4)),']'])

%% Verification
Tf = 20;
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
