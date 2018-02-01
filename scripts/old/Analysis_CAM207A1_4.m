clear all;
%dirname=['/Users/shaystern/Desktop/short/newillumination3/' result2];
mkdir('E:\Results\CAM207A1');

aviFiles = dir('*CAM207*.avi');
file_start=1;
file_end=272;
numavi = length(aviFiles);
shuttleVideo2 = VideoReader(aviFiles(file_start).name);
[corrd rect] = imcrop(read(shuttleVideo2,1));

%Mask=uint8(roipoly(imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)])));
Mask=uint8(roipoly(imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)])));
result = input('How many deleted objects?');
%Mask_2=uint8(roipoly(imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)])));
for a=1:result
Mask_2=roipoly(imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)]));
Mask_2=uint8(times(Mask_2,-1)+1);
%Mask_3=roipoly(imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)]));
%Mask_3=times(Mask_3,-1)+1;
Mask=uint8(times(Mask_2,Mask));
end;

%Mask3=times(Mask_2,-1)+1;.lkuywtyfdhfa`1
%Mask=times(Mask_1,Mask3);
Mask2(:,:,1)=Mask;
Mask2(:,:,2)=Mask;
Mask2(:,:,3)=Mask;
frame_window=30;
frame_window2=1;

% C=imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)]);
% A=uint8(roipoly(imcrop(read(shuttleVideo2,1),[rect(1),rect(2),rect(3),rect(4)])));
% B=times(A,C(:,:,1));

% thresh=0.35;
% WormMin=100;
% WormMax=1000;




for d=file_start:file_end+8
    shuttleVideo = VideoReader(aviFiles(d).name);
    Mean_zeros(1:4000,1:4000,1:3)=uint8(0);
    Mean_mat2=imcrop(Mean_zeros,[rect(1),rect(2),rect(3),rect(4)]);
    numf=shuttleVideo.NumberOfFrames;
    numf2=round(numf/8-1);

    for p=1:8
    
    Mean_mat{p}=imcrop(read(shuttleVideo,p*numf2),[rect(1),rect(2),rect(3),rect(4)]);
    Mean_mat2=Mean_mat2+((Mean_mat{p}-Mean_mat2)./p);
    Mean_mat2=times(Mask2,Mean_mat2);
    end;

    
    Norm_array{d}=Mean_mat2;
    display(d);
    
    
end;

tic;
for n = file_start:file_end
if n<100
    thresh=0.37;
    WormMin=30;
    WormMax=1000;
else
    thresh=0.35;
    WormMin=100;
    WormMax=1500;
end;
    
ind=1;
shuttleVideo = VideoReader(aviFiles(n).name);
display(aviFiles(n).name);


%bmpFiles = dir('*.bmp'); 
%numfiles = 300;
numfiles=shuttleVideo.NumberOfFrames;
m=mod(numfiles,frame_window);
numfiles2=(numfiles-m)/frame_window;
mydata = cell(1, numfiles);
%mydata_raw = cell(1, numfiles);
%mydata_all = cell(1, numfiles);
%mydata_bw = cell(1, numfiles);
%Norm=rgb2gray(imread(bmpFiles(1).name));

Norm=Norm_array{n+8};
% med=Norm(:,:,1);
% med2=median(med);
% med3=median(med2);
% med(ze)=med3;
% Norm(:,:,1)=med;
% med=Norm(:,:,2);
% med2=median(med);
% med3=median(med2);
% med(ze)=med3;
% Norm(:,:,2)=med;
% med=Norm(:,:,3);
% med2=median(med);
% med3=median(med2);
% med(ze)=med3;
% Norm(:,:,3)=med;






% outputVideo = VideoWriter('L2_raw_contour_0_35.avi');
% outputVideo.Quality=100;
% %open(outputVideo_bw);
% open(outputVideo);

   
for i = 1:numfiles2
    
    flag=0;
    display(aviFiles(n).name);
    %img = imcrop(read(shuttleVideo,(i*frame_window)),[374,142,900,900]);
    img = imcrop(read(shuttleVideo,i*frame_window-frame_window+1),[rect(1),rect(2),rect(3),rect(4)]);
    display(i);
    img2=times(img,Mask2)+100-Norm;
    %img3=imadjust(img2)
    %I=imread(bmpFiles(k).name);
    %I2=I+100-Norm;
    J=imcrop(img2,[1,1,4096,2160]);
    %J=imcrop(img2(:,:,3),[1,1,2160,4096]);
    I3=imcrop(img,[1,1,4096,2160]);
  
  
    Q=imfill(((im2bw(J, thresh))-1).*-1,'holes');
    CC = bwconncomp(Q);
    S = regionprops(CC,'Centroid');
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    if CC.NumObjects==0 || biggest<WormMin || biggest>WormMax 
        display('error');
        S2(1)=350;
        S2(2)=350;
        Norm2=imcrop(Norm,[S2(1)-350,S2(2)-350,700,700]);
        flag=1;
            
    else
    S = regionprops(CC,'Centroid');
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    S2=S(idx).Centroid;
    
    if S2(1)<350
        left_upper_corner(1)=rect(1);
    else
        left_upper_corner(1)=rect(1)+S2(1)-350;
    end;
    
    if S2(2)<350
        left_upper_corner(2)=rect(2);
    else
        left_upper_corner(2)=rect(2)+S2(2)-350;
    end;
  
    Norm2=imcrop(Norm,[S2(1)-350,S2(2)-350,700,700]);
  end;
    st=i*frame_window-frame_window+1;
    en=i*frame_window;
    
    for q=st:frame_window2:en
        
    img = imcrop(times(imcrop(read(shuttleVideo,q),[rect(1),rect(2),rect(3),rect(4)]),Mask2),[S2(1)-350,S2(2)-350,700,700]);
    %Mask3=imcrop(Mask2,[S2(1)-350,S2(2)-350,700,700]);
    img2=img+100-Norm2;
    %img2=times(img,Mask2)+100-Norm;
    %img3=imadjust(img2)
    %I=imread(bmpFiles(k).name);
    %I2=I+100-Norm;
    J=imcrop(img2,[1,1,4096,2160]);
    %J=imcrop(img2(:,:,3),[1,1,2160,4096]);
    I3=imcrop(img,[1,1,4096,2160]);
  
  
    Q=imfill(((im2bw(J, thresh))-1).*-1,'holes');
    CC = bwconncomp(Q);
    S = regionprops(CC,'Centroid');
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    if CC.NumObjects==0 || biggest<WormMin || biggest>WormMax || flag==1
   
        display('error');
        x_y_coor(ind,1:2)=[1,1];
        big(ind)=1;
        ind=ind+1;

    else
    S = regionprops(CC,'Centroid');
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    S2_2=S(idx).Centroid;
    mydata{ind}=imcrop(img2,[S2_2(1)-75,S2_2(2)-75,150,150]); 
    %mydata_all{ind}=read(shuttleVideo,q);
    
    
  big(ind)=biggest;
  
    
    %x_y_coor(ind,1:2)= [rect(1)+S2(1)-350+S2_2(1),rect(2)+S2(2)-350+S2_2(2)];
    x_y_coor(ind,1:2)= [left_upper_corner(1)+S2_2(1),left_upper_corner(2)+S2_2(2)];
    ind=ind+1;
    %mydata_contour{q}=imcrop(read(shuttleVideo,q),[S2(1)-200,S2(2)-200,400,400]);
%   zer=find(myda=ta_contour{i}==1);
%   I4=imcrop(I3,[S2(1)-100,S2(2)-100,200,200]);
%   I4(zer)=255;
%   mydata{i}=I4;
%   big(i)=biggest;
  %norm4{i}=img2;
  %writeVideo(outputVideo,mydata{k});
 
  display(q);

    % Write out to a JPEG file (img1.jpg, img2.jpg, etc.)
    %imwrite(img,fullfile(workingDir,'images',sprintf('img%d.jpg',ii)));
    end;
    end;
    
    
end;
   
   
    st=(i*frame_window)+1;
    en=numfiles;
    %en2=round((en-st)/frame_window2)*frame_window2+st;
    
    for q=st:frame_window2:en
    img = imcrop(times(imcrop(read(shuttleVideo,q),[rect(1),rect(2),rect(3),rect(4)]),Mask2),[S2(1)-350,S2(2)-350,700,700]);
    %Mask3=imcrop(Mask2,[S2(1)-350,S2(2)-350,700,700]);
    Norm2=imcrop(Norm,[S2(1)-350,S2(2)-350,700,700]);
    img2=img+100-Norm2;
        
    %img = imcrop(imcrop(read(shuttleVideo,q),[rect(1),rect(2),rect(3),rect(4)]),[S2(1)-350,S2(2)-350,700,700]);
    
    % img2=img+100-Norm2;
    %img2=times(img,Mask2)+100-Norm;
    %img3=imadjust(img2)
    %I=imread(bmpFiles(k).name);
    %I2=I+100-Norm;
    J=imcrop(img2,[1,1,4096,2160]);
    %J=imcrop(img2(:,:,3),[1,1,2160,4096]);
    I3=imcrop(img,[1,1,4096,2160]);
  
  
    Q=imfill(((im2bw(J, thresh))-1).*-1,'holes');
    CC = bwconncomp(Q);
    S = regionprops(CC,'Centroid');
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    if CC.NumObjects==0 || biggest<WormMin || biggest>WormMax || flag==1
    
        display('error');
        x_y_coor(ind,1:2)=[1,1];
        big(ind)=1;
        ind=ind+1;
    else
    S = regionprops(CC,'Centroid');
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    S2_2=S(idx).Centroid;
    
    
    mydata{ind}=imcrop(img2,[S2_2(1)-75,S2_2(2)-75,150,150]);  
    %mydata_all{ind}=read(shuttleVideo,q);
    
    
    big(ind)=biggest;
  
    %mydata_contour{q}=imcrop(imcrop(read(shuttleVideo,q),[rect(1),rect(2),rect(3),rect(4)]),[S2_2(1)-50,S2_2(2)-50,100,100]);
    x_y_coor(ind,1:2)= [left_upper_corner(1)+S2_2(1),left_upper_corner(2)+S2_2(2)];
    %x_y_coor(ind,1:2)= [rect(1)+S2(1)-350+S2_2(1),rect(2)+S2(2)-350+S2_2(2)];
    ind=ind+1;
    %mydata_contour{q}=imcrop(read(shuttleVideo,q),[S2(1)-200,S2(2)-200,400,400]);
%   zer=find(mydata_contour{i}==1);
%   I4=imcrop(I3,[S2(1)-100,S2(2)-100,200,200]);
%   I4(zer)=255;
%   mydata{i}=I4;
%   big(i)=biggest;
  %norm4{i}=img2;
  %writeVideo(outputVideo,mydata{k});
 
  display(q);
    end;
    end;

%toc;
%outputVideo_bw = VideoWriter('test_dinolite_gray.avi');
% outputVideo = VideoWriter('L1_RGB_8fps_060614_exp_337.avi');
% outputVideo.Quality=100;
% outputVideo.FrameRate=40;
% %open(outputVideo_bw);
% open(outputVideo);

filename2='E:\Results\CAM207A1\corrdCAM207A1';
filename3=[filename2 aviFiles(n).name];
filename3=filename3(1:end-4);
save(filename3,'x_y_coor');

Trace=read(shuttleVideo,1);
for i=1:ind-1
    Trace(round(x_y_coor(i,2)),round(x_y_coor(i,1)),1)=255;
end;

%filename2=sprintf('/Users/shaystern/Desktop/short/newillumination/Trajectory%d',n);
filename2='E:\Results\CAM207A1\TrajectoryCAM207A1';
filename3=[filename2 aviFiles(n).name];
filename3=filename3(1:end-4);
save(filename3,'Trace');

%filename=sprintf('/Users/shaystern/Desktop/short/newillumination/short%d.avi',n);
filename2='E:\Results\CAM207A1\shortCAM207A1';
filename3=[filename2 aviFiles(n).name];
filename3=filename3(1:end-4);
save(filename3,'mydata');

filename2='E:\Results\CAM207A1\WormSizeCAM207A1';
filename3=[filename2 aviFiles(n).name];
filename3=filename3(1:end-4);
save(filename3,'big');
%outputVideo = VideoWriter(filename,'Uncompressed AVI');
% outputVideo = VideoWriter(filename3);
% outputVideo.Quality=100;
% outputVideo.FrameRate=9;
% open(outputVideo);
% 
% for k = 1:ind-1
%    %writeVideo(outputVideo,mydata_all{k});
%    %writeVideo(outputVideo,double(mydata{k}));
%    writeVideo(outputVideo,mydata{k});
%    %writeVideo(outputVideo_bw,double(mydata_bw{k}));
%    display(k);
% end;
% 
% close(outputVideo);

%close(outputVideo);
%close(outputVideo_bw);
%filename2=sprintf('/Users/shaystern/Desktop/short/newillumination/corrd%d',n);

clear big;
clear mydata;
clear x_y_coor;

end;

%close(outputVideo);

%close(outputVideo_bw);
toc;


















