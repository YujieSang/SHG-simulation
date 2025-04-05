
close all; clc;

%from top-left pixel (1,1) to bottom-right pixel (2,3).
roi_row_start = 2;
roi_col_start = 2;
roi_row_end   = 8;
roi_col_end   = 8;
N               = 16;     % resolution (# pixels along x or y)
num_superpixels = 64;     % total superpixels on the SLM
num_phases      = 8;       % # of phase steps tested per superpixel
% Refractive indices
n1 = 1.8;  % fundamental
n2 = 1.8;  % second harmonic
w0 = 0.7e-3 * 5;  % beam waist (m)
I0 = 2e8;         % intensity (W/m^2)
if roi_row_start > roi_row_end
    tmp = roi_row_start;
    roi_row_start = roi_row_end;
    roi_row_end = tmp;
end
if roi_col_start > roi_col_end
    tmp = roi_col_start;
    roi_col_start = roi_col_end;
    roi_col_end = tmp;
end

%% 2) SIMULATION PARAMETERS
% Constants
c        = 3e8;         % speed of light (m/s)
epsilon0 = 8.854e-12;   % permittivity of free space (F/m)


if mod(N, sqrt(num_superpixels)) ~= 0
    error('N must be divisible by sqrt(num_superpixels) for equal-sized superpixels.');
end

L  = 10e-3;       % physical size in x and y (meters)
dx = L / N;       % grid spacing (m)

% Coordinates
x = linspace(-L/2, L/2, N);
y = linspace(-L/2, L/2, N);
[X, Y] = meshgrid(x, y);

% Propagation in z
Nz = 200;            % # of propagation steps
Lz = 0.01;           % total propagation distance (m)
dz = Lz / Nz;        % step (m)
z  = linspace(0, Lz, Nz);

% Wavelengths
lambda1 = 1064e-9;       % fundamental (m)
lambda2 = lambda1 / 2;   % second harmonic (m)



% Wave vectors
k1 = 2 * pi * n1 / lambda1;
k2 = 2 * pi * n2 / lambda2;

%% Phase mismatch (set to zero or something else if desired)
Delta_k = 1e6;
%% 

% Nonlinear coefficient
deff = 1.4e-12; % (m/V)



E0 = sqrt(2 * I0 / (n1 * epsilon0 * c)); % amplitude from intensity

% Move X, Y to GPU for speed
X = gpuArray(X);
Y = gpuArray(Y);

%% Define initial beam profile here
% Gaussian 
R0       = 0.005;        % Focal distance or radius of curvature (m)
tilt     = 1e3;          % Tilt coefficient (1/m)
k        = 2*pi / lambda1;  % wave number for fundamental
w0       = 0.5e-3;       % waist for the amplitude Gaussian

A = exp( - (X.^2 + Y.^2) / w0^2 );

phi_focusing = (k/(2*R0)) .* (X.^2 + Y.^2);

phi_tilt = k .* tilt .* X;

phi_total = phi_focusing + phi_tilt;

E1_baseline = E0 .* A .* exp(1i * phi_total);


% Hermite Gaussian
%E1_baseline = E0 * (2 * X / w0) .* exp(- (X.^2 + Y.^2) / w0^2);

% Bessel Gauss 
% k_r = 2*pi / w0;  % radial wavevector
% r = sqrt(X.^2 + Y.^2);
% E1_baseline = E0 * besselj(0, k_r * r) .* exp(-r.^2 / w0^2);


%%


E1_baseline = E1_baseline .* exp(1i * 2*pi * rand(N, N));

E2_baseline = gpuArray.zeros(N, N);

% Spatial frequencies for linear propagation
fx = (-N/2 : N/2 - 1) / L;
[FX, FY] = meshgrid(fx, fx);
FX = gpuArray(FX); 
FY = gpuArray(FY);
kxx = 2*pi * FX;
kyy = 2*pi * FY;
k_perp2 = kxx.^2 + kyy.^2;

% Propagation constants
c1    = 1i * dz / (2 * k1);
c2    = 1i * dz / (2 * k2);
omega1 = 2*pi * c / lambda1;
omega2 = 2*pi * c / lambda2;
NLc1  = 1i * dz * (2 * omega1^2 * deff) / (k1 * c^2);
NLc2  = 1i * dz * (omega2^2 * deff) / (k2 * c^2);

% Allocate for storing fields
E1_store_baseline = gpuArray.zeros(N, N, Nz);
E2_store_baseline = gpuArray.zeros(N, N, Nz);
I2_total_baseline = gpuArray.zeros(1, Nz);

% --- Split-step method for baseline
for nn = 1 : Nz
    % Linear (Fourier) step for E1
    E1f = fftshift(fft2(E1_baseline));
    E1f = E1f .* exp(c1 * k_perp2);
    E1_baseline = ifft2(ifftshift(E1f));
    
    % Linear step for E2
    E2f = fftshift(fft2(E2_baseline));
    E2f = E2f .* exp(c2 * k_perp2);
    E2_baseline = ifft2(ifftshift(E2f));

    % Phase mismatch
    E2_baseline = E2_baseline .* exp(1i * Delta_k * dz);

    % Nonlinear step
    E1_baseline = E1_baseline + NLc1 * E1_baseline .* conj(E2_baseline);
    E2_baseline = E2_baseline + NLc2 * E1_baseline.^2;

    % Store
    E1_store_baseline(:, :, nn) = E1_baseline;
    E2_store_baseline(:, :, nn) = E2_baseline;

    % Summed SHG intensity over entire plane
    I2_temp = (n2 * epsilon0 * c / 2) * abs(E2_baseline).^2; 
    I2_total_baseline(nn) = sum(I2_temp(:)) * dx * dx;
end

%% 4) DEFINE SUPERPIXELS

superpixel_size_x = N / sqrt(num_superpixels);
superpixel_size_y = N / sqrt(num_superpixels);

superpixel_indices = cell(num_superpixels, 1);
for sp = 1 : num_superpixels
    row = floor((sp-1) / sqrt(num_superpixels)) + 1;
    col = mod((sp-1), sqrt(num_superpixels)) + 1;
    
    x_start_sp = (col-1) * superpixel_size_x + 1;
    x_end_sp   = col * superpixel_size_x;
    y_start_sp = (row-1) * superpixel_size_y + 1;
    y_end_sp   = row * superpixel_size_y;
    
    superpixel_indices{sp} = struct('x_start', x_start_sp, 'x_end', x_end_sp, ...
                                    'y_start', y_start_sp, 'y_end', y_end_sp);
end

% Discrete phase steps we will test in each superpixel
phase_steps = linspace(0, 2*pi, num_phases+1);
phase_steps(end) = [];

%% 5) ADAPTIVE OPTIMIZATION FOR ROI
% We'll measure only in the user-chosen ROI [row,col] region at the output.
phase_mask      = gpuArray.zeros(N, N);
best_phase_mask = phase_mask;

for sp = 1 : num_superpixels
    sp_region = superpixel_indices{sp};
    max_I2_for_sp     = 0;
    best_phase_for_sp = 0;
    
    for p = 1 : num_phases
        current_phase = phase_steps(p);
        
        % Temporarily apply this 'current_phase' to the superpixel
        temp_phase_mask = best_phase_mask;
        temp_phase_mask(sp_region.y_start : sp_region.y_end, ...
                        sp_region.x_start : sp_region.x_end) = ...
            temp_phase_mask(sp_region.y_start : sp_region.y_end, ...
                            sp_region.x_start : sp_region.x_end) + current_phase;
        
        % Propagate with this temporary mask
        E1_temp = E0 * exp( - (X.^2 + Y.^2) / w0^2 ) .* exp(1i * temp_phase_mask);
        E2_temp = gpuArray.zeros(N, N);

        for nn = 1 : Nz
            % Linear step E1
            E1f = fftshift(fft2(E1_temp));
            E1f = E1f .* exp(c1 * k_perp2);
            E1_temp = ifft2(ifftshift(E1f));

            % Linear step E2
            E2f = fftshift(fft2(E2_temp));
            E2f = E2f .* exp(c2 * k_perp2);
            E2_temp = ifft2(ifftshift(E2f));

            % Phase mismatch
            E2_temp = E2_temp .* exp(1i * Delta_k * dz);

            % Nonlinear step
            E1_temp = E1_temp + NLc1 * E1_temp .* conj(E2_temp);
            E2_temp = E2_temp + NLc2 * E1_temp.^2;
        end

        % Measure SHG only in the user-specified ROI by pixel indices
        I2_final = (n2 * epsilon0 * c / 2) * abs(E2_temp).^2;
        ROI_I2 = sum( I2_final(roi_row_start : roi_row_end, ...
                               roi_col_start : roi_col_end), 'all') * dx * dx;

        % Keep track of which phase yields max ROI intensity
        if ROI_I2 > max_I2_for_sp
            max_I2_for_sp     = ROI_I2;
            best_phase_for_sp = current_phase;
        end
    end
    
    % Apply the best phase to the superpixel permanently
    best_phase_mask(sp_region.y_start : sp_region.y_end, ...
                    sp_region.x_start : sp_region.x_end) = ...
        best_phase_mask(sp_region.y_start : sp_region.y_end, ...
                        sp_region.x_start : sp_region.x_end) + best_phase_for_sp;
end

%% 6) FINAL PROPAGATION WITH OPTIMIZED PHASE MASK
E1_optimized = E0 * exp( - (X.^2 + Y.^2) / w0^2 ) .* exp(1i * best_phase_mask);
E2_optimized = gpuArray.zeros(N, N);

E1_store_optimized = gpuArray.zeros(N, N, Nz);
E2_store_optimized = gpuArray.zeros(N, N, Nz);
I2_total_optimized = gpuArray.zeros(1, Nz);

for nn = 1 : Nz
    E1f = fftshift(fft2(E1_optimized));
    E1f = E1f .* exp(c1 * k_perp2);
    E1_optimized = ifft2(ifftshift(E1f));

    E2f = fftshift(fft2(E2_optimized));
    E2f = E2f .* exp(c2 * k_perp2);
    E2_optimized = ifft2(ifftshift(E2f));

    E2_optimized = E2_optimized .* exp(1i * Delta_k * dz);

    E1_optimized = E1_optimized + NLc1 * E1_optimized .* conj(E2_optimized);
    E2_optimized = E2_optimized + NLc2 * E1_optimized.^2;

    E1_store_optimized(:, :, nn) = E1_optimized;
    E2_store_optimized(:, :, nn) = E2_optimized;

    I2_temp = (n2 * epsilon0 * c / 2) * abs(E2_optimized).^2;
    I2_total_optimized(nn) = sum(I2_temp(:)) * dx * dx;
end

%% 7) MOVE GPU -> CPU
E1_store_baseline   = gather(E1_store_baseline);
E2_store_baseline   = gather(E2_store_baseline);
I2_total_baseline   = gather(I2_total_baseline);
E1_store_optimized  = gather(E1_store_optimized);
E2_store_optimized  = gather(E2_store_optimized);
I2_total_optimized  = gather(I2_total_optimized);
x = gather(x);  % if you still want x, y for plotting
y = gather(y);
z = gather(z);
best_phase_mask = gather(best_phase_mask);


% (A) Compare baseline vs. optimized cross-sections (x vs z)
figure;
subplot(1,2,1);
imagesc(z*1e3, x*1e3, abs(squeeze(E1_store_baseline(round(N/2), :, :))).^2);
xlabel('z (mm)'); ylabel('x (mm)');
title('Fundamental Intensity, Baseline'); colorbar; axis tight;

subplot(1,2,2);
imagesc(z*1e3, x*1e3, abs(squeeze(E2_store_baseline(round(N/2), :, :))).^2);
xlabel('z (mm)'); ylabel('x (mm)');
title('SH Intensity, Baseline'); colorbar; axis tight;

figure;
subplot(1,2,1);
imagesc(z*1e3, x*1e3, abs(squeeze(E1_store_optimized(round(N/2), :, :))).^2);
xlabel('z (mm)'); ylabel('x (mm)');
title('Fundamental Intensity, Optimized'); colorbar; axis tight;

subplot(1,2,2);
imagesc(z*1e3, x*1e3, abs(squeeze(E2_store_optimized(round(N/2), :, :))).^2);
xlabel('z (mm)'); ylabel('x (mm)');
title('SH Intensity, Optimized'); colorbar; axis tight;

% % (B) Compare total SHG vs z
% figure;
% plot(z*1e3, I2_total_baseline, 'b-', 'LineWidth', 1.5); hold on;
% plot(z*1e3, I2_total_optimized,'r--','LineWidth', 1.5);
% xlabel('Propagation Distance (mm)');
% ylabel('Total SH Intensity (W)');
% legend('Baseline','Optimized','Location','best');
% title('Total Second-Harmonic Intensity vs z');
% grid on;



%% Interactive Visualization with Slider
figure('Units','normalized','Position',[0.1,0.1,0.8,0.6]);
idx = 1;
hAx1 = subplot(2,2,1);
hImg1 = imagesc(hAx1, x*1e3, y*1e3, abs(E2_store_optimized(:,:,idx)).^2);
xlabel(hAx1,'x (mm)'); ylabel(hAx1,'y (mm)');
title(hAx1,['SH @ z=' num2str(z(idx)*1e3,'%.2f') ' mm (Mask)']);
colorbar(hAx1); axis(hAx1,'square');

hAx2 = subplot(2,2,2);
hImg2 = imagesc(hAx2, x*1e3, y*1e3, abs(E2_store_baseline(:,:,idx)).^2);
xlabel(hAx2,'x (mm)'); ylabel(hAx2,'y (mm)');
title(hAx2,['SH @ z=' num2str(z(idx)*1e3,'%.2f') ' mm (No Mask)']);
colorbar(hAx2); axis(hAx2,'square');

hAx3 = subplot(2,2,[3,4]);
hLine1 = plot(hAx3, z*1e3, I2_total_baseline, 'b-', 'LineWidth',1.5); hold on;
hLine2 = plot(hAx3, z*1e3, I2_total_optimized, 'r--', 'LineWidth',1.5);
xlabel(hAx3,'z (mm)');
ylabel(hAx3,'Total SH Intensity (W)');
title(hAx3,'Total SH Intensity vs Propagation');
grid(hAx3,'on');
legend(hAx3,{'No Mask','Mask'},'Location','best');

%* Compute common y-axis limits for both curves
ymin = min([min(I2_total_baseline(:)), min(I2_total_optimized(:))]);
ymax = max([max(I2_total_baseline(:)), max(I2_total_optimized(:))]);
ylim(hAx3, [ymin, ymax]);  %* Set common y-limits

hold(hAx3,'on');
hMarker = plot(hAx3, z(idx)*1e3, I2_total_optimized(idx), 'ro', 'MarkerSize',8, 'LineWidth',2);
hold(hAx3,'off');

data.E2_store_optimized = E2_store_optimized;
data.E2_store_baseline = E2_store_baseline;
data.I2_total_optimized = I2_total_optimized;
data.I2_total_baseline = I2_total_baseline;
data.hImg1 = hImg1;
data.hImg2 = hImg2;
data.hAx1 = hAx1;
data.hAx2 = hAx2;
data.hAx3 = hAx3;
data.hMarker = hMarker;
data.z = z;
data.Nz = Nz;

sliderPos = [0.4,0.02,0.2,0.03];
hSlider = uicontrol('Style','slider','Units','normalized','Position',sliderPos,...
    'Value',1,'Min',1,'Max',Nz,'SliderStep',[1/(Nz-1),10/(Nz-1)],...
    'Callback',{@sliderCallback, data});

function sliderCallback(src,~,data)
    idx = round(get(src,'Value'));
    idx = max(1, min(idx, data.Nz));
    
    set(data.hImg1, 'CData', abs(data.E2_store_optimized(:,:,idx)).^2);
    title(data.hAx1, ['SH @ z=' num2str(data.z(idx)*1e3,'%.2f') ' mm (Mask)']);
    clim(data.hAx1, [min(abs(data.E2_store_optimized(:)).^2), max(abs(data.E2_store_optimized(:)).^2)]);
    
    set(data.hImg2, 'CData', abs(data.E2_store_baseline(:,:,idx)).^2);
    title(data.hAx2, ['SH @ z=' num2str(data.z(idx)*1e3,'%.2f') ' mm (No Mask)']);
    clim(data.hAx2, [min(abs(data.E2_store_baseline(:)).^2), max(abs(data.E2_store_baseline(:)).^2)]);
    
    set(data.hMarker, 'XData', data.z(idx)*1e3, 'YData', data.I2_total_optimized(idx));
    drawnow;
end
figure;
imagesc(x * 1e3, y * 1e3, mod(best_phase_mask, 2*pi)); 
xlabel('x (mm)');
ylabel('y (mm)');
title('Optimized Phase Mask (Wrapped 0 to 2\pi)');
colorbar;
axis equal tight;  % Make pixels square and fit tightly
