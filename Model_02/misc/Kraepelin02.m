t=0:0.001:10;

tau1=0.1;
tau2=0.1;
k=-2;
l=0.5;
m=3;
r=0;
x0=[-1;0];

eq=@(t,x)scz(t,x,tau1,tau2,k,l,m);
[t,x]=ode45(eq, t, x0);
iso = isoclines(-1:0.01:2,k,l,m);
[P,Q] = meshgrid(-1:0.1:2);
dP = (-P.^3-k*P.^2-l*P-Q)/tau1;
dQ = (m*P-Q)/tau2;

figure(1);
plot(t, x)

figure(2);
hold on
plot(-1:0.01:2,iso(1,:),'red')
plot(-1:0.01:2,iso(2,:),'green')
plot(x(:,1),x(:,2))
quiver(P,Q,dP,dQ)
hold off

function dxdt = scz(t,x,tau1,tau2,k,l,m)
    dxdt=[(-x(1)^3-k*x(1)^2-l*x(1)-x(2))/tau1;
          (m*x(1)-x(2))/tau2];
end

%{
function dxdt = scz(t,x,tau1,tau2,k,l,m,r)
    dxdt=[(-x(1)*(x(1)-k)+r-x(2))/tau1;
          (l*x(1)-x(2))/tau2];
end
%}

function iso = isoclines(V,k,l,m)
    iso = [-V.^3-k*V.^2-l*V;
           m*V];
end