function [Ac,Bc,Cc] = my_cart_pendulum_model(Mc,mp,l,bc,bp,g,Kv)

% Add your system model here
Ac(1,:) = [0 1 0 0];
Ac(2,:) = [0 -bc/Mc mp*g/Mc bp/(Mc*l)];
Ac(3,:) = [0 0 0 1];
Ac(4,:) = [0 bc/(Mc*l) -((Mc+mp)*g)/(Mc*l) -(bp*(Mc+mp))/(Mc*mp*l^2)];

Bc(1,1) = 0;
Bc(2,1) = Kv/Mc;
Bc(3,1) = 0;
Bc(4,1) = -Kv/(Mc*l);

Cc = [1 0 0 0;
      0 0 1 0];
