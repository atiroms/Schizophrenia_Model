%time in year
dt=.001;
t=14:dt:60;
%populational distribution 0-1 orderd by risk of onset (only 0-0.02 will be calculated)
dsample=.00001;
sample=0:dsample:.02;

%onset data reproduced from Hafner Arch Clin Psychiatry 2002
t_real=[13,17,22,27,32,37,42,47,52,57];
%t_real=[10,13,17,22,27,32,37,42,47,52,57,62];
%cumulative onset
onset_real=[0.00000795,0.00052325,0.00236027,0.00398756,0.00502651,0.00569886,0.00598945,0.00643747,0.00676400,0.00700000];
%onset_real=[0,0.00000795,0.00052325,0.00236027,0.00398756,0.00502651,0.00569886,0.00598945,0.00643747,0.00676400,0.00700000,0.00700000];

%parameters for the dynamic component
tau1=0.04; tau2=0.8;
k1=-1.6; k2=3.2; k3=0; k4=1.5;
tau3=300; 
%n_limit=0.3; n_progression=0.98;
m1=0.01; m2=0.1;
boundary=[0 0 0];

%representative samples for plotting
%sample_list=[0.008 0.0077 0.0076 0.0075];
sample_list=[0.009 0.007 0.004 0.0001];

%plotting parameters
plot_lim=-0.8:0.02:2.2;

%genetic risk defined as x with 100*sample percentile value in standard normal distribution snd(x)
genetic_risk=2^(1/2)*erfinv(1-2*sample);

%fit cumulative onset to polynomial
onset_fit=fit(t_real', onset_real', 'smoothingspline', 'SmoothingParam', 0.9);
onset=zeros(1,size(t,2));
for i=1:size(t,2)
    onset(i)=onset_fit(t(i));
end
%dval=polyfit(t_real,onset_real,5);
%onset=polyval(dval,t);

%onset defined as total risk = genetic risk + aging risk>0
%aging risk calculated from onset and genetic risk
aging_risk=-2^(1/2)*erfinv(1-2*onset);

%total risk matrix
risk=repelem(genetic_risk',1,numel(aging_risk))+repelem(aging_risk,numel(genetic_risk),1);
cumulative_randomness=zeros(size(risk));
cumulative_randomness(:,1)=t(1)*randn(size(risk,1),1);
for i=2:size(risk,2)
    randomness=dt*randn(size(risk,1),1);
    cumulative_randomness(:,i)=cumulative_randomness(:,i-1)+dt*randn(size(risk,1),1);
end
risk=risk+m1*cumulative_randomness+m2*randn(size(risk));
%risk=risk+m*randn(size(risk,1),size(risk,2));

%set negative, infinite, and nan values to zero
risk_pos=risk;
risk_pos(isnan(risk) | isinf(risk))=0;
risk_pos=max(0,risk_pos);

figure(1);
clf();
fig1=subplot(2,2,1);
plot(sample,genetic_risk);
title('Genetic Risk (pre-defined)')
xlabel('Sample')
ylabel('Risk')

fig2=subplot(2,2,2);
plot(t_real,onset_real,'o',t,onset);
title('Cumulative Onset (reproduced from Hafner 2002)')
xlabel('Age')
ylabel('Cumulative Onset')

fig3=subplot(2,2,3);
plot(t,aging_risk)
title('Aging Risk (calculated from Genetic Risk and Cumulative Onset)')
xlabel('Age')
ylabel('Risk')

opn=zeros(size(sample,2),size(t,2),3);
opn(:,1,:)=repelem(permute(boundary, [1 3 2]),size(sample,2),1);
for i=2:size(t,2)
    %van der Pol oscillator increments
    %factor O
    opn(:,i,1)=opn(:,i-1,1)+(k1*opn(:,i-1,1).^3+k2*opn(:,i-1,1).^2+k3*opn(:,i-1,1)-opn(:,i-1,2))/tau1*dt;
    %positive symptoms
    opn(:,i,2)=opn(:,i-1,1)+(k4*opn(:,i-1,2)-risk_pos(:,i-1)-opn(:,i-1,2))/tau2*dt;
    %negative and cognitive symptoms
    opn(:,i,3)=opn(:,i-1,3)+opn(:,i-1,1)/tau3*dt;
    %opn(3,i)=n_limit-(n_limit-opn(3,i-1))*n_progression^(opn(1,i-1)*dt);
end

figure(2);
clf()
fig1=subplot(2,2,1);
hold on
%surf(t,sample,risk,'EdgeColor','none','CData',risk,'FaceColor','flat','AlphaData',risk_transparency,'AlphaDataMapping','none','FaceAlpha','flat')
surf(t,sample,risk,'EdgeColor','none','CData',risk,'FaceColor','flat')
surf([t(1,1) t(1,end)], [sample(1,1) sample(1,end)], zeros(2,2), 'EdgeColor','none', 'FaceColor', 'k', 'FaceAlpha', 0.5)
hold off
title('Total Risk')
xlabel('Age')
ylabel('Sample')
zlabel('Risk')
colormap('jet')
ax=gca;
ax.XDir='reverse';
view([1,1,1])

fig2=subplot(2,2,2);
surf(t,sample,opn(:,:,2),'EdgeColor','none','CData',opn(:,:,2),'FaceColor','flat')
title('Positive Symptoms')
xlabel('Age')
ylabel('Sample')
zlabel('Severity')
colormap('jet')
ax=gca;
ax.XDir='reverse';
view([1,1,1])

fig3=subplot(2,2,3);
surf(t,sample,opn(:,:,3),'EdgeColor','none','CData',opn(:,:,3),'FaceColor','flat')
title('Negative and Cognitive Symptoms')
xlabel('Age')
ylabel('Sample')
zlabel('Severity')
colormap('jet')
ax=gca;
ax.XDir='reverse';
view([1,1,1])

%{
figure(3);
clf()
for j=1:size(sample_list,2)
    sample_ID=find(sample>(sample_list(1, j)-dsample/2) & sample<(sample_list(1, j)+dsample/2));
    
    nullcline=[k1*plot_lim.^3+k2*plot_lim.^2+k3*plot_lim;k4*plot_lim-risk_pos(sample_ID,size(t,2))];
    
    %parameters for vector field plotting
    [o_quiver,p_quiver] = meshgrid(-0.8:0.2:2.2);
    do_quiver = (k1*o_quiver.^3+k2*o_quiver.^2+k3*o_quiver-p_quiver)/tau1;
    dp_quiver = (k4*o_quiver-sample_list(j)-p_quiver)/tau2;
    
    fig1=subplot(2,size(sample_list,2),j);
    title(strcat('Sample = ',num2str(sample_list(j),3), 'Symptoms'))
    xlabel('Time')
    yyaxis left
    hold on
    plot(t, opn(2,:),'r')
    plot(t, opn(1,:),'--k')
    hold off
    ylabel('Symptom Intensity')
    ylim([-0.6 2.2])
    yyaxis right
    plot(t, opn(3,:),'b')
    ylim([-0.1 0.3])
    lgd1=legend('Positive','Factor O','Negative');
    lgd1.Location='southeast';
    ax = gca;
    ax.YAxis(1).Color = 'red';
    ax.YAxis(2).Color = 'blue';
    
    fig2=subplot(2,size(sample_list,2),size(sample_list,2)+j);
    hold on
    plot(plot_lim,nullcline)
    plot(opn(1,:),opn(2,:))
    quiver(o_quiver,p_quiver,do_quiver,dp_quiver)
    xlim([-0.9 2.3])
    ylim([-0.9 2.3])
    title(strcat('Sample = ',num2str(sample_list(j),3), ' Phase Portrait'))
    xlabel('Factor O')
    ylabel('Positive Symptoms')
    leg2=legend('O-Nullcline','P-Nullcline', ' Symptoms');
    leg2.Location='southeast';
    hold off
end
%}
