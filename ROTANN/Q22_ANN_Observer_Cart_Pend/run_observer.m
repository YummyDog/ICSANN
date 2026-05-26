%%  Run a closed-loop SFC system
%
%   48580 - Intelligent Control Studio
%   UTS, Australia
%
%   Your favorite academic,
%   A/Prof Ricardo P. Aguilera
%   May 2025
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_hat,y_hat,v_hat] = run_observer(A,B,C,Kf,x,u,xo_hat,y_noise)
N = length(x);  %data size

n = size(A)*[1 0]'; %number of states  x \in \R^n
m = size(B)*[0 1]'; %number of inputs  u \in \R^m

x_hat(:,1) = xo_hat;    %initial condition xo

for k = 1:N-1
    y_k = C*x(:,k)+y_noise(:,k);
    y_hat(:,k) = C*x_hat(:,k);
    v_hat(:,k) = Kf*(y_k-y_hat(:,k));
    x_hat(:,k+1) = A*x_hat(:,k)+B*u(:,k)+v_hat(:,k);
    x_hat(:,k+1) = min(max(x_hat(:,k+1), -1000), 1000);
end
y_hat(:,N) = C*x_hat(:,N);
v_hat(:,k+1) = v_hat(:,k);
end