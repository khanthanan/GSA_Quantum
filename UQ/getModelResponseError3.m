function modelResponseError = getModelResponseError3(theta, xdata, ydata, extra)
    % Get the model response based on the parameter values in theta.
    modelResponse = getModelResponse3(theta,xdata,extra);
    % Get the error between model prediction and data.
    modelResponseError = modelResponse-ydata;
end