s = load('splambda.mat');

H = 800;
W = 1024;
B = 448;
brightness=1.5;
normalise=1;

imageDir = fullfile('data','images');
calibDir = fullfile('data','calibration');
rgbDir   = fullfile('data','rgb');

whiteFile = fullfile(calibDir,'white_ref_2024-01-26_16-00-30.raw');
darkFile  = fullfile(calibDir,'dark_ref_shutter_2024-01-26_16-03-56.raw');

whiteRef  = double(multibandread(whiteFile, [H W B], 'uint16', 0, 'bil', 'ieee-le'));
darkRef   = double(multibandread(darkFile,  [H W B], 'uint16', 0, 'bil', 'ieee-le'));

files = dir(fullfile(imageDir,'*.raw'));

for i = 1:numel(files)

    rawFile = fullfile(imageDir, files(i).name);

    rawCube = double(multibandread(rawFile, [H W B], 'uint16', 0, 'bil', 'ieee-le'));

    [~, name, ~] = fileparts(rawFile);
    outFile = fullfile(rgbDir, [name '.png']);

    calibrate_hsi(rawCube, whiteRef, darkRef, splambda, brightness, outFile,normalise);

end