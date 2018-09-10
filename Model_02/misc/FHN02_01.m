figure('Position',[1 200 700 700]); % window size
global I;
for I = [0.2:0.005:1.2];  % varies I
f = @(t,y) [ 3*y(1) - y(1).^3 + 3*y(2) + 3*I; (1/3)*(0.7 - y(1) - 0.8*y(2)) ];
g = @(y) f(0,y);
fp = fsolve(g,[0 0]);       % Find the fixed point
Vss = fp(1); Wss = fp(2);   % Get the steady-state V and W values from "fp"
J = [ [3 - 3*Vss^2, 3]; [(-1/3), ((-0.8)/3)]];  % The Jacobian
Lambda = eig(J);                          % Eigenvalues of Jacobian
V = [-2:0.1:2];
VN = -V + V.^3/3 - I;   % V-nullcline
WN = (0.7 - V) / 0.8;  % W-nullcline
%%%%%%%%%%%%%%%%%%%%%%%     PANEL  1     %%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,1);
hold off;
plot(V, WN, 'k-', 'linewidth', 2);    % plot the W-nullcline
hold on;
plot(V, VN, 'k-.', 'linewidth', 2);   % plot the V-nullcline
[T,Y] = ode45(f, [0 80], [0, 0]);   % suggested [Vss+0.2, Wss]?
plot(Y(:,1), Y(:,2), 'm-', 'linewidth', 1.5);
title(['Phase plane, I=', num2str(I)], 'fontsize', 16);
xlabel('V(t)'); ylabel('W(t)');
axis([-2 3 -2 1]);
%%%%%%%%%%%%%%%%%%%%%%%     PANEL  2     %%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,2); hold off;
plot(T, Y(:,1), 'm-', 'linewidth', 1.5); hold on;
plot([0 80],[Vss Vss],'k:', 'linewidth', 1.5);
title('Voltage vs time', 'fontsize', 16);
xlabel('time'); ylabel('V(t)');
axis tight;
%%%%%%%%%%%%%%%%%%%%%%%     PANEL  3     %%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,3); hold on;
if real(Lambda(1)) > 0
    color = [1 0 0];   % Red color  ([R G B] values)
else 
    color = [0 0 1];   % Blue color ([R G B] values)
end;
plot(real(Lambda(1)),imag(Lambda(1)), '.', 'color', color);
plot(real(Lambda(2)),imag(Lambda(2)), '.', 'color', color);
title(['Real(\lambda)=',num2str(real(Lambda(1)))], 'fontsize', 16);
axis tight;
%%%%%%%%%%%%%%%%%%%%%%%     PANEL  4     %%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,4); hold on;
[T Y] = ode45(f, [0 100], Y(length(T),:) );  % Run some more in order
[T Y] = ode45(f, [0 100], Y(length(T),:) );  % to settle at equilibrium
plot(I, min(Y(:,1)), 'm.');       % minimum of V(t): a magenta point
plot(I, max(Y(:,1)), 'm.');       % maximum of V(t): a magenta point
plot(I,Vss,'.','color',color);    % Fixed point: a red or blue point
title('Bifurcation diagram (V_{min}, V_{max}, V*)', 'fontsize', 12);
xlabel('I'); ylabel('V equilibrium');
axis tight;
drawnow;
end;