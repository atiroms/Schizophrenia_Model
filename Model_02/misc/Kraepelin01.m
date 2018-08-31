t=0:0.001:10;

tau1=0.1;
tau2=0.2;
k=1;
l=2;
m=1;
r=0;
x0=[0;0];

eq=@(t,x)scz(t,x,tau1,tau2,k,l,m,r);
[t,x]=ode45(eq, t, x0);
iso = isoclines(-1:0.01:2,k,l,r);
[P,Q] = meshgrid(-1:0.1:2);
dP = (-P.^2+k*P+r-Q)/tau1;
dQ = (l*P-Q)/tau2;

figure(1);
plot(t, x)

figure(2);
hold on
plot(-1:0.01:2,iso(1,:),'red')
plot(-1:0.01:2,iso(2,:),'green')
plot(x(:,1),x(:,2))
quiver(P,Q,dP,dQ)
hold off


function dxdt = scz(t,x,tau1,tau2,k,l,m,r)
    dxdt=[(-x(1)*(x(1)-k)+r-x(2))/tau1;
          (l*x(1)-x(2))/tau2];
end

function iso = isoclines(V,k,l,r)
    iso = [-V.^2+k*V+r;
           l*V];
end