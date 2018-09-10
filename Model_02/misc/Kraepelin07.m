%{
dt=0.001;
t=0:dt:2;
tau1=0.02;
tau2=0.1;
k1=-5;
k2=5;
k3=0;
l=1.4;
m=0.9;
rlist=[-0.01 0.01 0.1 0.2 1.5];
origin=[0.08;0;0];
plot_lim=-0.4:0.01:1.1;
%}

dt=0.001;
t=0:dt:2;
tau1=0.02;
tau2=0.1;
k1=-5;
k2=5;
k3=0;
l=1.4;
m=0.9;
rlist=[-0.01 0.01 0.1 0.2 1.5];
origin=[0.08;0;0];
plot_lim=-0.4:0.01:1.1;

figure(1);
title('Schizophrenia Severity in Continuum')
for j=1:size(rlist,2)
    r=rlist(j)*ones(size(t,2),1);
    qpn=zeros(3, size(t,2));
    qpn(:,1)=origin;
    for i=2:size(t,2)
        qpn(1,i)=qpn(1,i-1)+(k1*qpn(1,i-1)^3+k2*qpn(1,i-1)^2+k3*qpn(1,i-1)-qpn(2,i-1))/tau1*dt;
        qpn(2,i)=qpn(2,i-1)+(l*qpn(1,i-1)-r(i-1)-qpn(2,i-1))/tau2*dt;
        qpn(3,i)=1-(1-qpn(3,i-1))*m^(qpn(1,i-1)*dt);
    end
    nullcline=[k1*plot_lim.^3+k2*plot_lim.^2+k3*plot_lim;l*plot_lim-r(1)];
    [x_quiver,y_quiver] = meshgrid(-0.4:0.1:1);
    dx_quiver = (k1*x_quiver.^3+k2*x_quiver.^2+k3*x_quiver-y_quiver)/tau1;
    dy_quiver = (l*x_quiver-r(1)-y_quiver)/tau2;
    
    fig1=subplot(2,size(rlist,2),j);
    title('Symptoms')
    %set(fig1, 'defaultAxesColorOrder',[[1,0,0]; [0,0,1]])

    xlabel('Time')
    yyaxis left
    plot(t, qpn(2,:),'red')
    ylabel('Symptom Intensity')
    ylim([-0.4 1.2])
    yyaxis right
    plot(t, qpn(3,:),'blue')
    ax.YAxis(2).Direction = 'reverse';
    ylim([-0.1 0.2])
    legend('Positive','Negative')
    ax = gca;
    ax.YAxis(1).Color = 'red';
    ax.YAxis(2).Color = 'blue';
    
    fig2=subplot(2,size(rlist,2),size(rlist,2)+j);
    cla()
    hold on
    plot(plot_lim,nullcline)
    plot(qpn(1,:),qpn(2,:))
    quiver(x_quiver,y_quiver,dx_quiver,dy_quiver)
    xlim([-0.5 1.1])
    ylim([-0.5 1.1])
    title('Attractor')
    xlabel('Factor Q')
    ylabel('Positive Symptoms')
    legend('X Nullcline','Y Nullcline', 'Symptoms')
    hold off
end