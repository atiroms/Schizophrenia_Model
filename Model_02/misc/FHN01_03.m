t=0:0.1:1000;
%{
a=0.08;
b=0.7;
c=0.8;
I=0.5;
x0=[0;0];
%}
a=0.08;
b=2;
c=0.1;
I=10;
x0=[-2.5;0];

eq=@(t,x)fhn(t,x,a,b,c,I);
[t,x]=ode45(eq, t, x0);
iso = isoclines(-2.5:0.1:2.5,I,b,c);
[V,W] = meshgrid(-2.5:0.25:2.5);
dV = V-V.^3/3-W;
dW = a*(b+V-c*W);

figure(1);
plot(t, x)

figure(2);
hold on
plot(-2.5:0.1:2.5,iso(1,:),'red')
plot(-2.5:0.1:2.5,iso(2,:),'green')
plot(x(:,1),x(:,2))
quiver(V,W,dV,dW)
xlim([-2.6 2.6])
ylim([-2.6 5])
hold off

function dxdt = fhn(t,x,a,b,c,I)
    dxdt=[x(1)-(x(1)^3)/3-x(2)+I;
          a*(b+x(1)-c*x(2))];
end

function iso = isoclines( V,I,b,c )
    iso = [V-(V.^3)/3+I;(b+V)/c];
end