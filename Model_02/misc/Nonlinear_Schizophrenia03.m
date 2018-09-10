%time in year
dt=0.001;
t=14:dt:60;
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
m=0.9;
origin=[-0.1;0;0];
plot_lim=-0.4:0.01:1.1;


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

risk_positive=risk;
risk_positive(isnan(risk) | isinf(risk))=0;
risk_positive=max(0,risk_positive);

qpn=zeros(3, size(t,2));
qpn(:,1)=origin;

for i=2:size(t,2)
    qpn(1,i)=qpn(1,i-1)+(k1*qpn(1,i-1)^3+k2*qpn(1,i-1)^2+k3*qpn(1,i-1)-qpn(2,i-1))/tau1*dt;
    qpn(2,i)=qpn(2,i-1)+(l*qpn(1,i-1)-risk_positive(500,i-1)-qpn(2,i-1))/tau2*dt;
    qpn(3,i)=1-(1-qpn(3,i-1))*m^(qpn(1,i-1)*dt);
end

figure(5);
clf
plot(t, qpn)
ylim([-0.4 1.2])

nullcline=k1*plot_lim.^3+k2*plot_lim.^2+k3*plot_lim;

figure(6);
clf
hold on
plot(plot_lim,nullcline)
plot(qpn(1,:),qpn(2,:))
hold off