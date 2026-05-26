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
function [x,y,u] = run_SFC_CL_system(A,B,C,K,x_ref,xo)
N = length(x_ref);  %data size
x(:,1) = xo;    %initial condition xo
y(:,1) = C*x(:,1);      %initial output yo
for k = 1:N-1
    xe = (x(:,k)-x_ref(:,k));   %tracking error
    u(:,k) = -K*xe;             %SFC

    %input saturation
    if (u(:,k)>12)
        u(:,k) = 12;
    end
    if (u(:,k)<-12)
        u(:,k) = -12;
    end
    
    % apply input to the system model
    x(:,k+1) = A*x(:,k)+B*u(:,k);
    y(:,k+1) = C*x(:,k+1);
end
%add a final input to have the same vector dimension
u(:,k+1) = u(:,k); 
end