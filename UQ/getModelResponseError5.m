function modelResponseError = getModelResponseError5(theta, xdata, ydata, extra)
    % Get the model response based on the parameter values in theta.
    modelResponse = getModelResponse5(theta,xdata,extra);
    % Get the error between model prediction and data.
%     size(modelResponse)
%     size(ydata)
    modelResponseError = modelResponse-ydata;
end