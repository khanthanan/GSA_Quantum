function modelResponseError = getModelResponseError4(theta, xdata, ydata, extra)
    % Get the model response based on the parameter values in theta.
    modelResponse = getModelResponse4(theta,xdata,extra);
    % Get the error between model prediction and data.
    modelResponseError = modelResponse-ydata;
end