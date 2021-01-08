% This script is based off of the Verasonics example script
% SetUpL7_4_128RyLns.m, and it uses the L7-4 probe. The sequence obtains 
% P.num_beams beams using P.numTx elements for each transmit event. The aperture is 
% walked by one element for each beam, so the number of beams is determined
% by the number of transmit elements. Out of the 128 elements that receive
% data, only the P.numTx elements that correspond to each transmit event 
% are used for receive beamforming. The transmit origin of each beam is
% centered about the transmit aperture. The last modifications to this
% script were made by Christopher Khan in 2020.


% Clear the workspace
clear all;

% Define P structure
P.startDepth = 0;                 % Start depth for acquisition (wavelengths)
P.endDepth = 300;                 % End depth for acquisition (wavelengths)
P.txFocus = 236;                  % Initial transmit focus (wavelengths)
P.numTx = 65;                     % Number of transmit elements in TX aperture (where possible)
P.num_beams = 128 - P.numTx + 1;  % Number of beams (walked aperture)
P.numFrames = 1;                  % Number of frames in RcvBuffer

% Define initial values for adjustable ADMIRE parameters 
alpha = 0.9;                      % Alpha for elastic-net regularization
lambda_scaling_factor = 0.0189;   % Lambda scaling factor used to scale lambda for elastic-net regularization
max_iterations = 1E5;             % Maximum number of cyclic coordinate descent iterations to perform (convergence criterion)
tolerance = 0.1;                  % Maximum weighted (observation weights are all 1 in this case) sum of squares of the changes in the fitted values between iterations of cyclic coordinate descent (convergence criterion)

% Define system parameters
Resource.Parameters.numTransmit = 128;      % Number of transmit channels
Resource.Parameters.numRcvChannels = 128;   % Number of receive channels
Resource.Parameters.speedOfSound = 1540;    % Set speed of sound in m/sec before calling computeTrans
Resource.Parameters.verbose = 2;
Resource.Parameters.initializeOnly = 0;
Resource.Parameters.simulateMode = 0;

% Specify Trans structure array
Trans.name = 'L7-4';
Trans.units = 'wavelengths';  % Explicit declaration avoids warning message when selected by default
Trans.frequency = 5.2083;     % Transducer center frequency (MHz)
Trans = computeTrans(Trans);  % L7-4 transducer is 'known' transducer so we can use computeTrans
Trans.maxHighVoltage = 50;    % Set maximum high voltage limit for pulser supply

% Set max acquistion length
P.maxAcqLength = ceil(sqrt(P.endDepth^2 + ((Trans.numelements-1)*Trans.spacing)^2));

% Define the conversion factor (mm/wavelength) to convert from wavelengths to mm 
wls2mm = Resource.Parameters.speedOfSound/1000/Trans.frequency;

% Specify Resources
Resource.RcvBuffer(1).datatype = 'int16';
Resource.RcvBuffer(1).rowsPerFrame = 128*ceil(P.maxAcqLength*2*4/128)*P.num_beams;
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames = P.numFrames;       

% Specify TW structure array
TW(1).type = 'parametric';
TW(1).Parameters = [Trans.frequency,0.67,2,1];

% Specify TX structure arrays 
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

% Calculate the transmit origin for the first transmit event
element_pos_vector = Trans.ElementPos(:, 1);
if mod(P.numTx, 2) == 1
    origin_start_ind = ((P.numTx - 1)/2) + 1;
    origin_TX = element_pos_vector(origin_start_ind);
elseif mod(P.numTx, 2) == 0
    origin_start_ind_1 = P.numTx/2;
    origin_start_ind_2 = origin_start_ind_1 + 1;
    origin_TX = mean([element_pos_vector(origin_start_ind_1), ...
        element_pos_vector(origin_start_ind_2)]);
end

% Set the transmit origin and transmit apodization for each transmit event
for n = 1:P.num_beams   % P.num_beams transmit events
    % Set transmit origin to the center of the transmit aperture
    TX(n).Origin = [origin_TX, 0.0, 0.0];
    
    % Move the lateral position of the transmit origin to obtain the
    % lateral position of the transmit origin for the next transmit event
    origin_TX = origin_TX + Trans.spacing;
    
    % Set transmit apodization
    TX(n).Apod(left_element:right_element) = 1.0;
    TX(n).Delay = computeTXDelays(TX(n));
    left_element = left_element + 1;
    right_element = right_element + 1;
end

% Specify Receive structure arrays 
% Need P.num_beams Receives for every frame
Receive = repmat(struct('Apod', ones(1,Trans.numelements), ...
                        'startDepth', P.startDepth, ...
                        'endDepth', P.maxAcqLength, ...
                        'TGC', 1, ...
                        'bufnum', 1, ...
                        'framenum', 1, ...
                        'acqNum', 1, ...
                        'sampleMode', 'NS200BW',...
                        'mode', 0, ...
                        'callMediaFunc', 0), 1, P.num_beams*Resource.RcvBuffer(1).numFrames);
                    
% Set event specific Receive attributes
for i = 1:Resource.RcvBuffer(1).numFrames
    k = P.num_beams*(i-1);
    Receive(k+1).callMediaFunc = 1;
    for j = 1:P.num_beams
        Receive(k+j).framenum = i;
        Receive(k+j).acqNum = j;
    end
end

% Specify TGC Waveform structure
TGC.CntrlPts = [0,138,260,287,385,593,674,810];
TGC.rangeMax = P.endDepth;
TGC.Waveform = computeTGCWaveform(TGC);

% Specify Process structure array                     
Process(1).classname = 'External';
Process(1).method = 'gpu_processing_linear_probe';
Process(1).Parameters = {'srcbuffer','receive',...
                         'srcbufnum',1,...
                         'srcframenum',1,...
                         'dstbuffer','none'};
        
% Specify SeqControl structure arrays
% Time between acquisitions in usec
t1 = ceil(((P.maxAcqLength*2*wls2mm/1000)/Resource.Parameters.speedOfSound)*1E6)+10;
SeqControl(1).command = 'timeToNextAcq';
SeqControl(1).argument = t1;

% Return to Matlab
SeqControl(2).command = 'returnToMatlab';

% Jump back to the first sequence event
SeqControl(3).command = 'jump';
SeqControl(3).argument = 1;

% Define the variable that contains the next sequence control number
nsc = 4;

% Specify Event structure arrays
n = 1;
for i = 1:Resource.RcvBuffer(1).numFrames
    % Perform P.num_beams transmit/receive events for each frame
    for j = 1:P.num_beams                    
        Event(n).info = 'Aqcuisition';
        Event(n).tx = j;                     % Transmit event
        Event(n).rcv = P.num_beams*(i-1)+j;  % Receive event
        Event(n).recon = 0;                  % No reconstruction
        Event(n).process = 0;                % No processing
        Event(n).seqControl = 1;             % seqControl 
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
       SeqControl(nsc).command = 'transferToHost'; % Transfer frame to host buffer
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
Event(n).tx = 0;                % No transmit
Event(n).rcv = 0;               % No receive
Event(n).recon = 0;             % No reconstruction
Event(n).process = 0;           % No processing
Event(n).seqControl = 3;        % seqControl


% User specified UI Control Elements            
% Transmit focus change
UI(1).Control = {'UserB5','Style','VsSlider','Label',['TX Focus (','mm',')'],...
                 'SliderMinMaxVal',[20,320,P.txFocus]*wls2mm,'SliderStep',[0.1,0.2],'ValueFormat','%3.0f'};
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
save('L7-4_RyLns_ADMIRE');
filename = 'L7-4_RyLns_ADMIRE'
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

% Update the transmit focus
P = evalin('base','P');
P.txFocus = UIValue .* scaleToWvl;
assignin('base','P',P);

% Update the TX structure
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
