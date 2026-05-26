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
function [x,y,u] = run_SFC_CL_system(A,B,C,K,x_ref,u_lim)
N = length(x_ref);  %data size
n = size(A)*[1 0]';
x(:,1) = zeros(n,1);    %initial condition xo
y(:,1) = C*x(:,1);      %initial output yo
for k = 1:N-1
    xe = (x(:,k)-x_ref(:,k));   %tracking error
    u(:,k) = -K*xe;             %SFC

    %input saturation
    if (u(:,k)>u_lim(2))
        u(:,k) = u_lim(2);
    end
    if (u(:,k)<u_lim(1))
        u(:,k) = u_lim(1);
    end
    
    % apply input to the system model
    x(:,k+1) = A*x(:,k)+B*u(:,k);
    y(:,k+1) = C*x(:,k+1);
end
%add a final input to have the same vector dimension
u(:,k+1) = u(:,k); 
end