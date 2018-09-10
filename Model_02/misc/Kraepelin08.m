%{
dt=0.001;
t=0:dt:10;
tau1=0.04;
tau2=0.8;
k1=-5;
k2=5;
k3=0;
k4=1.2;
l=0.99;
m=0.08;
r_list=[0 0.005 0.015 0.1 0.9];
boundary=[0;0;0];
plot_lim=-0.4:0.01:1.1;
%}

dt=0.001;
t=0:dt:10;
tau1=0.04;
tau2=0.8;
k1=-5;
k2=5;
k3=0;
k4=1.2;
l=0.99;
m=0.08;
r_list=[0 0.005 0.015 0.1 0.9];
boundary=[0;0;0];
plot_lim=-0.4:0.01:1.1;

figure(1);
clf()
for j=1:size(r_list,2)
    r=r_list(j)*ones(1,size(t,2))+m*randn(1,size(t,2));
    opn=zeros(3, size(t,2));
    opn(:,1)=boundary;
    for i=2:size(t,2)
        opn(1,i)=opn(1,i-1)+(k1*opn(1,i-1)^3+k2*opn(1,i-1)^2+k3*opn(1,i-1)-opn(2,i-1))/tau1*dt;
        opn(2,i)=opn(2,i-1)+(k4*opn(1,i-1)-r(i-1)-opn(2,i-1))/tau2*dt;
        opn(3,i)=1-(1-opn(3,i-1))*l^(opn(1,i-1)*dt);
    end
    nullcline=[k1*plot_lim.^3+k2*plot_lim.^2+k3*plot_lim;k4*plot_lim-r_list(j)];
    [o_quiver,p_quiver] = meshgrid(-0.4:0.1:1);
    do_quiver = (k1*o_quiver.^3+k2*o_quiver.^2+k3*o_quiver-p_quiver)/tau1;
    dp_quiver = (k4*o_quiver-r_list(j)-p_quiver)/tau2;
    
    fig1=subplot(2,size(r_list,2),j);
    title(strcat('Symptoms R = ',num2str(r_list(j),3)))
    xlabel('Time')
    yyaxis left
    hold on
    plot(t, opn(2,:),'r')
    plot(t, opn(1,:),'--k')
    hold off
    ylabel('Symptom Intensity')
    ylim([-0.4 1.2])
    yyaxis right
    plot(t, opn(3,:),'b')
    ylim([-0.1 0.3])
    legend('Positive','Factor O','Negative')
    ax = gca;
    ax.YAxis(1).Color = 'red';
    ax.YAxis(2).Color = 'blue';
    
    fig2=subplot(2,size(r_list,2),size(r_list,2)+j);
    hold on
    plot(plot_lim,nullcline)
    plot(opn(1,:),opn(2,:))
    quiver(o_quiver,p_quiver,do_quiver,dp_quiver)
    xlim([-0.5 1.1])
    ylim([-0.5 1.1])
    title(strcat('Phase Portrait R = ',num2str(r_list(j),3)))
    xlabel('Factor O')
    ylabel('Positive Symptoms')
    legend('O-Nullcline','P-Nullcline', 'Symptoms')
    hold off
end