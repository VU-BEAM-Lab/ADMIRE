% This script is based off of the Verasonics example script
% SetUpC5_2_128RyLns.m, and it uses the C5-2 probe. The sequence obtains 
% P.num_beams beams using P.numTx elements for each transmit event. The aperture is 
% walked by one element for each A-line, so the number of A-lines is determined
% by the number of transmit elements. Out of the 128 elements that receive
% data, only the P.numTx elements that correspond to each transmit event 
% are used for receive beamforming. The transmit origin of each A-line is
% centered about the transmit aperture. The last modifications to this
% script were made by Christopher Khan in 2020;


% Clear the workspace
clear all;

% Define P structure
P.startDepth = 0;                 % Start depth for acquisition (wavelengths)
P.endDepth = 192;                 % End depth for acquisition (wavelengths)
P.txFocus = 100;                  % Initial transmit focus (wavelengths)
P.numTx = 65;                     % Number of transmit elements in TX aperture (where possible)
P.num_beams = 128 - P.numTx + 1;  % Number of beams (walked aperture)
P.numFrames = 1;                  % Number of frames in RcvBuffer

% Define initial values for adjustable ADMIRE parameters 
alpha = 0.9;                      % Alpha for elastic-net regularization
lambda_scaling_factor = 0.0189;   % Lambda scaling factor used to scale lambda for elastic-net regularization
max_iterations = 1E5;             % Maximum number of cyclic coordinate descent iterations to perform (convergence criterion)
tolerance = 0.1;                  % Maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values between iterations of cyclic coordinate descent (convergence criterion)
 
% Specify system parameters
Resource.Parameters.numTransmit = 128;     % Number of transmit channels
Resource.Parameters.numRcvChannels = 128;  % Number of receive channels
Resource.Parameters.speedOfSound = 1540;   % Set speed of sound in m/sec before calling computeTrans
Resource.Parameters.verbose = 2;
Resource.Parameters.initializeOnly = 0;
Resource.Parameters.simulateMode = 0;

% Specify Trans structure array
Trans.name = 'C5-2';
Trans.units = 'wavelengths';  % Required in Gen3 to prevent default to mm units
Trans.frequency = 3.125;      % Transducer center frequency (MHz)
Trans = computeTrans(Trans);  % C5-2 transducer is 'known' transducer so we can use computeTrans
Trans.maxHighVoltage = 50;    % Set maximum high voltage limit for pulser supply
radius = Trans.radius;
scanangle = P.num_beams*Trans.spacing/radius; % Total scan angle (radians)
dtheta = scanangle/P.num_beams;      % Angle (radians) between successive beams
theta = -(scanangle/2) + 0.5*dtheta; % Angle (radians) to left edge from centerline
Angle = theta:dtheta:(-theta);       % Contains the angle values (radians) that the beams correspond to

% Compute the maximum receive path length (wavelengths), using the law of cosines
transducer_angle_span = Trans.numelements*Trans.spacing/radius;  % Angle span (radians) of the whole transducer
theta_first_element = -(transducer_angle_span/2) + 0.5*dtheta;   % Angle (radians) that corresponds to the first element
P.maxAcqLength = ceil(sqrt((P.endDepth+radius)^2 + radius^2 - ...
    2*(P.endDepth+radius)*radius*cos(-2*theta_first_element)));

% Define the conversion factor (mm/wavelength) to convert from wavelengths to mm 
wls2mm = Resource.Parameters.speedOfSound/1000/Trans.frequency;

% Specify Resources
Resource.RcvBuffer.datatype = 'int16';
Resource.RcvBuffer.rowsPerFrame = 128*ceil(P.maxAcqLength*2*4/128)*P.num_beams;
Resource.RcvBuffer.colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer.numFrames = 1;

% Specify Transmit waveform structure
TW(1).type = 'parametric';
TW(1).Parameters = [Trans.frequency,.67,2,1];

% Specify TX structure array
% Need P.num_beams transmit specifications
TX = repmat(struct('waveform', 1, ...
                   'Origin', [0.0,0.0,0.0], ...
                   'focus', P.txFocus, ...
                   'Steer', [0.0,0.0], ...
                   'Apod', zeros(1,Trans.numelements), ...
                   'Delay', zeros(1,Trans.numelements)), 1, P.num_beams);
               
% Set event specific TX attributes
% Define the first and last elements of the transmit aperture for the first
% transmit event
left_element = 1;
right_element = P.numTx; 

% Set the transmit origin and transmit apodization for each transmit event
for n = 1:P.num_beams
    % Set transmit waveform
    TX(n).waveform = 1;
    
    % Set each transmit origin to the center of the transmit aperture    
    TX(n).Origin = [radius*sin(Angle(n)), 0.0, radius*cos(Angle(n))-radius];
    
    % Set transmit apodization
    TX(n).Apod(left_element:right_element) = 1.0;
    TX(n).Delay = computeTXDelays(TX(n));
    left_element = left_element + 1;
    right_element = right_element + 1;
end
                 
% Specify Receive structure arrays
% Need P.num_beams Receives for each frame                 
Receive = repmat(struct('Apod', ones(1,Trans.numelements), ...
                        'startDepth', P.startDepth, ...
                        'endDepth', P.maxAcqLength, ...
                        'TGC', 1, ...
                        'bufnum', 1, ...
                        'framenum', 1, ...
                        'acqNum', 1, ...
                        'sampleMode','NS200BW', ...
                        'mode', 0, ...
                        'callMediaFunc', 0), 1, P.num_beams*Resource.RcvBuffer(1).numFrames);
                    
% Set event specific Receive attributes
for i = 1:Resource.RcvBuffer(1).numFrames
    Receive(P.num_beams*(i-1)+1).callMediaFunc = 1;
    for j = 1:P.num_beams
        Receive(P.num_beams*(i-1)+j).framenum = i;
        Receive(P.num_beams*(i-1)+j).acqNum = j;
    end
end

% Specify TGC Waveform structure
TGC.CntrlPts = [0,375,411,425,468,681,862,944];
TGC.rangeMax = P.endDepth;
TGC.Waveform = computeTGCWaveform(TGC);

% Specify Process structure array
Process(1).classname = 'External';
Process(1).method = 'gpu_processing_curvilinear_probe';
Process(1).Parameters = {'srcbuffer','receive',...
                         'srcbufnum',1,...
                         'srcframenum',1,...
                         'dstbuffer','none'};                 

% Specify SeqControl structure arrays
% Jump back to the first sequence event
SeqControl(1).command = 'jump';
SeqControl(1).argument = 1;

% Time between acquisitions in usec
SeqControl(2).command = 'timeToNextAcq';
SeqControl(2).argument = ceil(((P.maxAcqLength*2*wls2mm/1000)/Resource.Parameters.speedOfSound)*1E6)+10;

% Return to Matlab
SeqControl(3).command = 'returnToMatlab';

% Define the variable that contains the next sequence control number
nsc = 4;

% Specify Event structure arrays
n = 1;
for i = 1:Resource.RcvBuffer(1).numFrames
    % Perform P.num_beams transmit/receive events for each frame
    for j = 1:P.num_beams               
        Event(n).info = 'Acquisition';
        Event(n).tx = j;                      % Transmit event
        Event(n).rcv = P.num_beams*(i-1)+j;   % Receive event
        Event(n).recon = 0;                   % No reconstruction
        Event(n).process = 0;                 % No processing
        Event(n).seqControl = 2;              % seqControl
        n = n+1;
    end
    
    % Set the last event's seqControl to 0 
    Event(n-1).seqControl = 0;
    
    % Transfer the data frame to the RcvBuffer
    Event(n).info = 'Transfer frame to host';
    Event(n).tx = 0;                          % No transmit
    Event(n).rcv = 0;                         % No receive
    Event(n).recon = 0;                       % No reconstruction
    Event(n).process = 0;                     % No processing
    Event(n).seqControl = nsc;                % seqControl
       SeqControl(nsc).command = 'transferToHost'; % transfer frame to host buffer
       nsc = nsc+1;
    n = n+1;

    % Process the data frame with ADMIRE and synchronize the hardware and
    % software sequencers    
    Event(n).info = 'GPU ADMIRE processing'; 
    Event(n).tx = 0;                               % No transmit
    Event(n).rcv = 0;                              % No receive
    Event(n).recon = 0;                            % No reconstruction
    Event(n).process = 1;                          % ADMIRE GPU processing
    Event(n).seqControl = [nsc, nsc + 1, nsc + 2]; % seqControl
        SeqControl(nsc).command = 'waitForTransferComplete';
        SeqControl(nsc).argument = nsc - 1;
        SeqControl(nsc + 1).command = 'markTransferProcessed';
        SeqControl(nsc + 1).argument = nsc - 1;
        SeqControl(nsc + 2).command = 'sync';
        nsc = nsc + 3;
    n = n+1;
end

% Jump back to the first sequence event
Event(n).info = 'Jump back';
Event(n).tx = 0;              % No transmit
Event(n).rcv = 0;             % No receive
Event(n).recon = 0;           % No reconstruction
Event(n).process = 0;         % No processing
Event(n).seqControl = 1;      % seqControl


% User specified UI Control Elements
% Transmit focus change
UI(1).Control = {'UserB5','Style','VsSlider','Label',['TX Focus (','mm',')'],...
                 'SliderMinMaxVal',[50,300,P.txFocus]*wls2mm,'SliderStep',[10/250,20/250],'ValueFormat','%3.0f'};
UI(1).Callback = text2cell('%TxFocusCallback');

% Change alpha for elastic-net regularization in ADMIRE
UI(2).Control = {'UserB4','Style','VsSlider','Label','Alpha','SliderMinMaxVal', ...
                 [0,1,alpha],'SliderStep',[0.01,0.1],'ValueFormat','%0.2f'};
UI(2).Callback = text2cell('%AlphaCallback');

% Change the lambda scaling factor for elastic-net regularization in ADMIRE
UI(3).Control = {'UserB3','Style','VsSlider','Label','Lambda Scaling',...
                 'SliderMinMaxVal',[0,1,lambda_scaling_factor],'SliderStep',[1E-4,5E-3],'ValueFormat','%0.4f'};
UI(3).Callback = text2cell('%LambdaScalingFactorCallback');

% Change the maximum number of iterations for cyclic coordinate descent in ADMIRE
UI(4).Control = {'UserB2','Style','VsSlider','Label',' Max. Iter. CCD',...
                 'SliderMinMaxVal',[0,1E6,max_iterations],'SliderStep',[10/(1E6),(1E4)/(1E6)],'ValueFormat','%f'};
UI(4).Callback = text2cell('%MaximumIterationsCallback');

% Change the maximum coefficient change tolerance for cyclic coordinate descent in ADMIRE
UI(5).Control = {'UserB1','Style','VsSlider','Label','Toler. CCD',...
                 'SliderMinMaxVal',[0,100,tolerance],'SliderStep',[(1E-4)/100,1/100],'ValueFormat','%0.4f'};
UI(5).Callback = text2cell('%ToleranceCallback');

% Save all the structures to a .mat file.
save('C5-2_RyLns_ADMIRE');
filename = 'C5-2_RyLns_ADMIRE'
VSX 
return

% **** Callback routines to be converted by text2cell function. ****
%TxFocusCallback - TX focus change
simMode = evalin('base','Resource.Parameters.simulateMode');
% No focus change if in simulate mode 2.
if simMode == 2
    set(hObject,'Value',evalin('base','P.txFocus'));
    return
end
Trans = evalin('base','Trans');
Resource = evalin('base','Resource');
scaleToWvl = Trans.frequency/(Resource.Parameters.speedOfSound/1000);

P = evalin('base','P');
P.txFocus = UIValue .* scaleToWvl;
assignin('base','P',P);

TX = evalin('base', 'TX');
for n = 1:P.num_beams   % P.num_beams transmit events
    TX(n).focus = P.txFocus;
    TX(n).Delay = computeTXDelays(TX(n));
end
assignin('base','TX', TX);

% Set Control command to update TX
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'TX'};
assignin('base','Control', Control);
return
%TxFocusCallback

%AlphaCallback
alpha = evalin('base','alpha');
alpha = UIValue;
assignin('base','alpha',alpha);
return
%AlphaCallback

%LambdaScalingFactorCallback
lambda_scaling_factor = evalin('base','lambda_scaling_factor');
lambda_scaling_factor = UIValue;
assignin('base','lambda_scaling_factor',lambda_scaling_factor);
return
%LambdaScalingFactorCallback

%MaximumIterationsCallback
max_iterations = evalin('base','max_iterations');
max_iterations = UIValue;
assignin('base','max_iterations',max_iterations);
return
%MaximumIterationsCallback

%ToleranceCallback
tolerance = evalin('base','tolerance');
tolerance = UIValue;
assignin('base','tolerance',tolerance);
return
%ToleranceCallback
