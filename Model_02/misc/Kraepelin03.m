%{
t=0:0.0001:1;
tau1=0.02;
tau2=0.1;
k=-1;
l=0;
m=1.4;
r=0.1;
x0=[-0.1;0];
plot_lim=[-0.4:0.01:1.1];
%}

t=0:0.0001:1;
tau1=0.02;
tau2=0.1;
k=-1;
l=0;
m=1.4;
r=0.2;
x0=[-0.1;0];
plot_lim=[-0.4:0.01:1.1];

eq=@(t,x)scz(t,x,tau1,tau2,k,l,m,r);
[t,x]=ode45(eq, t, x0);
null = nullclines(plot_lim,k,l,m,r);
[x_quiver,y_quiver] = meshgrid(-0.4:0.1:1);
dx_quiver = (-5*(x_quiver.^3+k*x_quiver.^2+l*x_quiver)-y_quiver)/tau1;
dy_quiver = (m*x_quiver-r-y_quiver)/tau2;

figure(1);
plot(t, x)
ylim([-0.4 1.2])

figure(2);
clf
hold on
plot(plot_lim,null(1,:),'red')
plot(plot_lim,null(2,:),'green')
plot(x(:,1),x(:,2))
quiver(x_quiver,y_quiver,dx_quiver,dy_quiver)
xlim([-0.5 1.1])
ylim([-0.5 1.1])
hold off

function dxdt = scz(t,x,tau1,tau2,k,l,m,r)
    dxdt=[(-5*(x(1)^3+k*x(1)^2+l*x(1))-x(2))/tau1;
          (m*x(1)-r-x(2))/tau2];
end

%{
function dxdt = scz(t,x,tau1,tau2,k,l,m,r)
    dxdt=[(-x(1)*(x(1)-k)+r-x(2))/tau1;
          (l*x(1)-x(2))/tau2];
end
%}

function null = nullclines(V,k,l,m,r)
    null = [-5*(V.^3+k*V.^2+l*V);
           m*V-r];
end