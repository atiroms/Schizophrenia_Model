function iso = isoclines( V,I,b,c )
    iso = [V-(V.^3)/3+I;(b+V)/c];
end