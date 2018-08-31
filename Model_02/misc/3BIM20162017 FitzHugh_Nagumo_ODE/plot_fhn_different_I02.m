t=0:0.1:1000;
%{
a=0.08;
b=0.7;
c=0.8;
I=0.5;
x0=[0;0];
%}
a=0.08;
b=0.7;
c=0.8;
I=-0.5;
x0=[-1;-1];

function dxdt = fhn(t,x,a,b,c,I)
    dxdt=[x(1)-(x(1)^3)/3-x(2)+I;
          a*(b+x(1)-c*x(2))];
end

function iso = isoclines( V,I,b,c )
    iso = [V-(V.^3)/3+I;(b+V)/c];
end

function dydt = jacobian( x,a,c )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
dydt=[1-x(1)^2,-1;a,-a*c];
end

%Il = @(t)In(t,50);

eq=@(t,x)fhn(t,x,a,b,c,I);
[t,x]=ode45(eq, t, x0);
iso = isoclines(-2:0.1:2,I,b,c);
figure(1);
plot(t, x)
figure(2);
hold on
plot(-2:0.1:2,iso(1,:),'red')
plot(-2:0.1:2,iso(2,:),'green')
plot(x(:,1),x(:,2))
hold off

%hold on
%{
j = 1
for I=[0,0.25,1.0,2.0]
    subplot(2,2,j)
	eq=@(t,x)fhn(t,x,a,b,c,I);
	[t,x]=ode45(eq, t, x0);
	%plot(t, x)
    iso = isoclines(-2:0.1:2,I,b,c);
    plot(-2:0.1:2,iso(1,:),'red')
    hold on
    plot(-2:0.1:2,iso(2,:),'green')
    plot(x(:,1),x(:,2))
    j = j+1;
end
%}

%{
x = transpose([-2:0.04:2; -2:0.04:2])
J=zeros(100,4);
for i=1:101
    j = jacobian(x(i,:),a,c);
	J(i,1:2)= j(1:2);
    J(i,3:4)= j(3:4);
end
plot(x(1:101,1),J)
hold off
%}

%https://www.intechopen.com/source/html/40762/media/image13.png
%{
hold on
for I=0:0.02:5
	eq=@(t,x)fhn(t,x,a,b,c,I);
	[t,x]=ode45(eq, t, x0);
    f=@(x)fhn(0,x,a,b,c,I);
	eqx=fsolve(f,x0);
	spec=eig(jacobian(eqx,a,c));
	respec = real(spec);
	plot(I, respec,'o');
end
hold off
%}
%{
I=1.5;
iso = isoclines(-2:0.1:2,I,b,c);
hold on
plot(-2:0.1:2,iso(1,:))
plot(-2:0.1:2,iso(2,:))
f=@(x)fhn(0,x,a,b,c,I);
eqx=fsolve(f,[1;1]);
plot(eqx(1),eqx(2),'o')
%}

%{
I=1.5;
[V,W] = meshgrid(-10:2:10);
[V,W];
dV = V-V^3/3-W+1;
dW = a*(b+V-c*W);
hold on
quiver(V,W,dV,dW)
eq=@(t,x)fhn(t,x,a,b,c,I);
[t,x]=ode45(eq, t, [-5;6]);
plot(x(:,1),x(:,2))
[t,x]=ode45(eq, t, [5;6]);
plot(x(:,1),x(:,2))
[t,x]=ode45(eq, t, [-6;1]);
plot(x(:,1),x(:,2))
[t,x]=ode45(eq, t, [1;-6]);
plot(x(:,1),x(:,2))
[t,x]=ode45(eq, t, [-5;-6]);
plot(x(:,1),x(:,2))
[t,x]=ode45(eq, t, [10;5]);
plot(x(:,1),x(:,2))
[t,x]=ode45(eq, t, [6;-2]);
plot(x(:,1),x(:,2))
%}
