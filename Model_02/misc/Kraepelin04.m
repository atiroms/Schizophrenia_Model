%{
t=0:0.0001:10;
tau1=0.02;
tau2=0.1;
k1=-5;
k2=5;
k3=0;
l=1.4;
m=0.05;
r=0.1;
x0=[-0.1;0];
plot_lim=[-0.4:0.01:1.1];
%}

t=0:0.0001:10;
tau1=0.02;
tau2=0.1;
k1=-5;
k2=5;
k3=0;
l=1.4;
m=0.05;
r=0.2;
x0=[-0.1;0];
plot_lim=[-0.4:0.01:1.1];

eq=@(t,x)scz(t,x,tau1,tau2,k1,k2,k3,l,r);
[t,x]=ode45(eq, t, x0);
n=m*cumtrapz(t,x(:,1));

null = nullclines(plot_lim,k1,k2,k3,l,r);
[x_quiver,y_quiver] = meshgrid(-0.4:0.1:1);
dx_quiver = (k1*x_quiver.^3+k2*x_quiver.^2+k3*x_quiver-y_quiver)/tau1;
dy_quiver = (l*x_quiver-r-y_quiver)/tau2;

figure(1);
clf
hold on
plot(t, x)
plot(t, n)
ylim([-0.4 1.2])
hold off

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

function dxdt = scz(t,x,tau1,tau2,k1,k2,k3,l,r)
    dxdt=[(k1*x(1)^3+k2*x(1)^2+k3*x(1)-x(2))/tau1;
          (l*x(1)-r-x(2))/tau2];
end

function null = nullclines(V,k1,k2,k3,l,r)
    null = [k1*V.^3+k2*V.^2+k3*V;
           l*V-r];
end