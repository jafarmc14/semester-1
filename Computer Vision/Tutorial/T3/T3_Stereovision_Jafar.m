% read pair of images
I1 = imread('left/left-0003.png');
I2 = imread('right/right-0003.png');

% rectify images using calculated stereo parameters
[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);

imtool(cat(3, J1, J2, J2));

% disparity range
dispRange = [80, 192];
disparityMap = disparity(J1, J2, ...
    'DisparityRange', dispRange, ...    
    'BlockSize', 17, ...
    'ContrastThreshold', 0.5, ...
    'UniquenessThreshold', 17 );
	
% show disparity map
figure 
imshow(disparityMap, dispRange);
colormap(gca,jet) 
colorbar

points3D = reconstructScene(disparityMap, stereoParams);

% expand gray image to three channels (simulate RGB)
J1_col = cat(3, J1, J1, J1);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', J1_col);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', 'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);