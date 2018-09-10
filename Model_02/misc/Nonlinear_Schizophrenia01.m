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

%time in year
t=14:.001:60;
%t=0:0.0001:10;
%populational distribution 0-1 orderd by risk of onset (only 0-0.02 will be cauculated)
sample=0:.00001:0.02;

%reproduced from Hafner Arch Clin Psychiatry 2002
%t_real=[13,17,22,27,32,37,42,47,52,57];
t_real=[10,13,17,22,27,32,37,42,47,52,57,62];
%cumulative onset
%onset_real=[0.00000795,0.00052325,0.00236027,0.00398756,0.00502651,0.00569886,0.00598945,0.00643747,0.00676400,0.00700000];
onset_real=[0,0.00000795,0.00052325,0.00236027,0.00398756,0.00502651,0.00569886,0.00598945,0.00643747,0.00676400,0.00700000,0.00700000];

tau1=0.02;
tau2=0.1;
k1=-5;
k2=5;
k3=0;
l=1.4;
m=0.05;
r=0.2;
x0=[-0.1;0;0];
plot_lim=[-0.4:0.01:1.1];


%genetic risk defined as x with 100*sample percentile value in standard normal distribution snd(x)
genetic_risk=2^(1/2)*erfinv(1-2*sample);
figure(1);
plot(sample,genetic_risk);
title('Genetic Risk')
xlabel('Sample')
ylabel('Risk')

dval=polyfit(t_real,onset_real,5);
onset=polyval(dval,t);
figure(2);
plot(t_real,onset_real,'o',t,onset);
title('Cumulative Onset')
xlabel('Age')
ylabel('Cumulative Onset')

%onset defined as total risk = genetic risk + aging risk>0
aging_risk=-2^(1/2)*erfinv(1-2*onset);
figure(3);
plot(t,aging_risk)
title('Aging Risk')
xlabel('Age')
ylabel('Risk')

risk=repelem(aging_risk,numel(genetic_risk),1)+repelem(genetic_risk',1,numel(aging_risk));
risk_transparency=(0.5*(risk>=0)+0.5);
figure(4);
surf(t,sample,risk,'EdgeColor','none','CData',risk,'FaceColor','flat','AlphaData',risk_transparency,'AlphaDataMapping','none','FaceAlpha','flat')
title('Total Risk')
xlabel('Age')
ylabel('Sample')
zlabel('Risk')
colormap('jet')
view([-1,1,1])



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