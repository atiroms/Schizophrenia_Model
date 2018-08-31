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

dt=0.001;
t=0:dt:10;
tau1=0.02;
tau2=0.1;
k1=-5;
k2=5;
k3=0;
l=1.4;
m=0.9;
r=1*ones(size(t,2),1);
origin=[-0.01;0;0];
plot_lim=-0.4:0.01:1.1;

qpn=zeros(3, size(t,2));
qpn(:,1)=origin;

for i=2:size(t,2)
    qpn(1,i)=qpn(1,i-1)+(k1*qpn(1,i-1)^3+k2*qpn(1,i-1)^2+k3*qpn(1,i-1)-qpn(2,i-1))/tau1*dt;
    qpn(2,i)=qpn(2,i-1)+(l*qpn(1,i-1)-r(i-1)-qpn(2,i-1))/tau2*dt;
    qpn(3,i)=1-(1-qpn(3,i-1))*m^(qpn(1,i-1)*dt);
end

nullcline=k1*plot_lim.^3+k2*plot_lim.^2+k3*plot_lim;

figure
subplot(1,2,1)
plot(t, qpn)
title('Symptoms')
xlabel('Time')
ylabel('Symptom Intensity')
ylim([-0.4 1.2])


subplot(1,2,2)
hold on
plot(plot_lim,nullcline)
plot(qpn(1,:),qpn(2,:))
hold off