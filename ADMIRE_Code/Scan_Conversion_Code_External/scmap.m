% Copyright Stephen Hsu: Permission was obtained from Stephen Hsu to
% include this code in this repository

% Code was modified by Christopher Khan: The modification was that the
% speed of sound in m/s was added as a function input. The modification was
% made in 2020.

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the license at

%     http://www.apache.org/licenses/LICENSE-2.0

% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and 
% limitations under the License.


% Description of scmap.m:
% This function is used in order to calculate the scan conversion map that
% is used to perform scan conversion of ultrasound images


function [idx,i00,dr,dth,lat,ax] = scmap(dim,mode,min_phi,span_phi,apex,dsfactor,freq,c,varargin)
%
% SCAN_CONVERT
%
% BEWARE OF UNITS !!!! BEWARE OF UNITS !!!! 2DBEWARE OF UNITS !!!!
% converts A-line data and geometric information to a scan-converted image
%
% EXAMPLES:
%
% All coordinates and default spacing (.2mm)
% [idx,i00,dr,dth,lat,ax]=scmap(dim,'sector',min_phi,span_phi,apex,1)
%
% User defined coordinates, and spacing
% [idx,i00,dr,dth,lat,ax]=scan_convert(dim,'sector',min_phi,span_phi,apex,1,[ax_min ax_max, ax_inc, lat_min, lat_max, lat_inc]); %
% INPUT:
%  dim = dimension of input array [rows columns]  
%  mode = scan type ('sector')
%  min_phi = minimum angle (degrees)
%  span_phi = angle span (degrees)
%  apex = distance to radial center (cm)
%  dsfactor = downsampling factor
%  c = speed of sound (m/s)
%  vargin = [ ax_min ax_max ax_inc lat_min lat_max lat_inc ] (meters)
%
% OUTPUT:
%  idx = array of indicies in non-zero elements in the output image
%  i00 = indicies of upper left point mapped from original image
%  dr = normalized radial distance to upper left point
%  dth = normalized theta distance to upper left point
%  lat = final lateral output coordinates
%  ax = final axial output coordinates
%

if (strcmp(mode,'sector'))

  row = dim(1);
  col = dim(2);

  % initialize input indices array 
  in = reshape(1:row*col,[row col]);

  % convert apex to mm
  apex_mm = -apex*10;

  % convert coordinates: 0 degrees = left, 90 degrees = straight down.
  min_phi = 90+min_phi;

  % sampling frequency
  fs = freq; %5e6; %warning('hardcoded fs manually changed'); 40e6;

  % radial resolution (mm)
  dr = c/2/fs*1000;
  samples_per_mm = 1/dr;

  % assign radial coordinates
  r=(0:(size(in,1)-1))/samples_per_mm+apex_mm;

  % assign theta coordinates
  theta = min_phi + (0:size(in,2)-1)*span_phi/(size(in,2)-1);

  % theta resolution
  dth = theta(2)-theta(1);
  samples_per_degree = 1/dth;

  % define recta-linear coordinate limits
  xmin = -max(r)*cos(min_phi*pi/180);
  xmax = -max(r)*cos(max(theta)*pi/180);

  ymin = min(r)*min(sin(theta*pi/180));
  ymax = max(r)*max(sin(theta*pi/180));

  % define default increment (mm)
  inc = 0.05;

  % default parameters
  param = [ymin ymax inc xmin xmax inc];

  % allow user defined limits/increments 0==default
  if (nargin>6) 
    q = varargin{6-5};

  % allows code to be compatible with earlier versions
    if (length(q)==5)
      q(4:6)=q(3:5);
      q(3) = q(6);
    end;

    idx = find(q~=0);
    q = q+[apex_mm apex_mm 0 0 0 0]/1000;
    param(idx) = q(idx)*1000;
  end;

  % reassign values
  ymin = param(1);
  ymax = param(2);
  yinc = param(3);
  xmin = param(4);
  xmax = param(5);
  xinc = param(6);

  % define recti-linear grid
  y = ymin:yinc:ymax;
  x = xmin:xinc:xmax;

  [X,Y] = meshgrid(x,y);

  % create output image
  out = zeros(size(X));
  dR = out;
  dTh = out;

  % create polar grid
  R = sqrt(X.^2+Y.^2);
  Th = atan2(Y,-X)*180/pi;

  % find indicies outside image boundaries
  idx = union(find(R>max(r)),find(R<min(r)));   
  idx2 = union(find(Th>max(theta)),find(Th<min(theta)));
  idx = union(idx,idx2);

  % zero those values
  R(idx) = 0;
  Th(idx) = 0;

  % find indicies inside image boundaries
  idx = find(R~=0);

  % find upper, left image pixel indicies
  r_index = floor((R(idx)-apex_mm)*samples_per_mm)+1;
  th_index = floor((Th(idx)-min(theta))*samples_per_degree)+1;

  % calculate distance from upper left pixel to mapped pixel
  dr = ((R(idx)-apex_mm)*samples_per_mm+1)-r_index;
  dth = ((Th(idx)-min(theta))*samples_per_degree+1)-th_index;

  % convert 2D indicies into 1D indicies
  i00 = (length(r)+1)*(th_index-1)+r_index;

  % assign recto-linear output coordinates (cm)
  y = y-apex_mm;
  ax = y/10;
  lat = x/10;

end;

