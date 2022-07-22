function modelResponse = getModelResponse6(theta, xdata, extra)
    % This model computes the probability of state |110>.

    [t1,t2,t3,t4,t5,t6,t7,t8,t9] = transfer(theta);
    
    f1tmp = (-sin(t6/2)*sin(t8/2)*cos(t2/2 + t4/2)*cos(t7/2 - t9/2) ...
        - 1i*sin(t6/2)*sin(t7/2 + t9/2)*cos(t8/2)*cos(t2/2 + t4/2) ...
        + 1i*sin(t7/2)*sin(t9/2)*sin(t2/2 + t4/2)*sin(t5/2 - t8/2)*cos(t6/2) ...
        + sin(t7/2)*sin(t2/2 + t4/2)*cos(t6/2)*cos(t9/2)*cos(t5/2 - t8/2) ...
        + sin(t9/2)*sin(t2/2 + t4/2)*cos(t6/2)*cos(t7/2)*cos(t5/2 + t8/2) ...
        - 1i*sin(t2/2 + t4/2)*sin(t5/2 + t8/2)*cos(t6/2)*cos(t7/2)*cos(t9/2))*cos(t1/2);
        
    
    f1tmp = abs(f1tmp)^2;
 
    modelResponse = (f1tmp)*ones(length(xdata),1); %hardcoded 2000!
end
