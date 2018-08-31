t=14:.01:60;
sample=0:.0001:0.02;
genetic_risk=2^(1/2)*erfinv(1-2*sample);
figure(1);
plot(sample,genetic_risk);
title('Genetic Risk')
xlabel('Sample')
ylabel('Risk')

t_real=[10,13,17,22,27,32,37,42,47,52,57,62];
onset_real=[0,0.00000795,0.00052325,0.00236027,0.00398756,0.00502651,0.00569886,0.00598945,0.00643747,0.00676400,0.00700000,0.00700000];
dval=polyfit(t_real,onset_real,5);
onset=polyval(dval,t);
figure(2);
plot(t_real,onset_real,'o',t,onset);
title('Cumulative Onset')
xlabel('Age')
ylabel('Cumulative Onset')

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
view([-1,1,1])