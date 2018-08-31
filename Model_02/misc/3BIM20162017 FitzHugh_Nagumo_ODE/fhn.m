function dxdt = fhn(t,x,a,b,c,I)
    dxdt=[x(1)-(x(1)^3)/3-x(2)+I;
          a*(b+x(1)-c*x(2))];
end