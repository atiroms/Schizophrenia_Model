function dydt = jacobian( x,a,c )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
dydt=[1-x(1)^2,-1;a,-a*c];
end