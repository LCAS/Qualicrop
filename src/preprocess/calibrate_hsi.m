function [cube_cal, rgbimage] = calibrate_hsi(rawCube, whiteRef, darkRef, lambda, brightness, outRgbFile, normalize_spec)

    [H, W, B] = size(rawCube);

    rawCube(~isfinite(rawCube)) = 0;

    lambda = double(lambda(:))';
    if numel(lambda) ~= B
        error('Length of lambda (%d) must match number of bands in rawCube (%d).', numel(lambda), B);
    end

    whiteRef(~isfinite(whiteRef)) = 0;
    darkRef(~isfinite(darkRef)) = 0;

    % calibration
    denom = whiteRef - darkRef;
    denom(~isfinite(denom)) = eps;
    denom(abs(denom) < eps) = eps;

    cube_cal = (rawCube - darkRef) ./ denom;

    cube_cal(~isfinite(cube_cal)) = 0;

    % clip calibrated values
    cube_cal = max(cube_cal, 0);
    cube_cal = min(cube_cal, 1);

    if normalize_spec
        cube_cal = normalize_by_spectrum(cube_cal,'L2');
        cube_cal(~isfinite(cube_cal)) = 0;
    end

    if strlength(outRgbFile) > 0
        [outDir, name, ext] = fileparts(outRgbFile);
        baseDir = fileparts(outDir);

        if normalize_spec
            calibDir = fullfile(baseDir, 'calibrated', 'norm_reflectance');
            rgbDir   = fullfile(baseDir, 'rgb', 'norm_reflectance');
        else
            calibDir = fullfile(baseDir, 'calibrated', 'reflectance');
            rgbDir   = fullfile(baseDir, 'rgb', 'reflectance');
        end

        if ~exist(calibDir, 'dir')
            mkdir(calibDir);
        end
        if ~exist(rgbDir, 'dir')
            mkdir(rgbDir);
        end

        calibFile = fullfile(calibDir, [name '.raw']);
        rgbFile   = fullfile(rgbDir, [name ext]);

        multibandwrite(single(cube_cal), calibFile, 'bil');
    end

    % create RGB image using your wavelength-based band selection
    rgbimage = makeRGBimage(cube_cal, lambda, brightness);

    % save RGB
    if strlength(outRgbFile) > 0
        imwrite(im2uint8(max(min(rgbimage,1),0)), rgbFile);
    end
end

function [rgbimage] = makeRGBimage(imageCube, lambda, brightness)

    if ~exist('brightness', 'var') || isempty(brightness)
        brightness = 1.25;
    end

    imageCube(~isfinite(imageCube)) = 0;

    % Find RGB bands
    out = find_rgb_bands(lambda);

    % Create unnormalised RGB image
    rgbimage = cat(3, ...
        imageCube(:,:,out(1)), ...
        imageCube(:,:,out(2)), ...
        imageCube(:,:,out(3)));

    % Normalise each channel
    for i = 1:3
        channel = double(rgbimage(:,:,i));
        channel(~isfinite(channel)) = 0;

        channel = channel - min(channel(:));

        chmax = max(channel(:));
        if ~isfinite(chmax) || chmax <= 0
            channel = zeros(size(channel));
        else
            channel = (brightness * channel) / chmax;
        end

        channel(~isfinite(channel)) = 0;
        rgbimage(:,:,i) = channel;
    end

    rgbimage = max(min(rgbimage, 1), 0);
end


function [out] = find_rgb_bands(lambda)

    lambda = double(lambda(:))';

    lambda(~isfinite(lambda)) = 0;

    % If wavelengths are in nm, convert to micrometers
    if max(lambda) > 10
        lambda = lambda / 1000;
    end

    % Target wavelengths
    red   = 0.6329;
    green = 0.5510;
    blue  = 0.454528;

    B = find(lambda <= blue, 1, 'last');
    if isempty(B)
        B = 1;
    end

    R = find(lambda <= red, 1, 'last');
    if isempty(R)
        R = numel(lambda);
    end

    G = find(lambda <= green, 1, 'last');
    if isempty(G)
        G = min(2, numel(lambda));
    end

    out = [R G B];
end


function Cube_out = normalize_by_spectrum(Cube,integral_type)
eps_val = 1e-12;
scale_factor=100;

[H, W, B] = size(Cube);
Cube = max(Cube,0);
if (integral_type=='L1')
SpecInt = sum(Cube, 3);
elseif(integral_type=='L2')
SpecInt=sqrt(sum(Cube.^2,3));
end
Den = max(SpecInt, eps_val);
Den = repmat(Den, [1 1 B]);
Cube_out = Cube ./ Den;
Cube_out = Cube_out * scale_factor;

end