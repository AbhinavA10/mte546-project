clc;
clear;
close all;
gps = csvread("gps_data.csv");
gt = csvread("gt_data.csv");

figure()
hold on
scatter(gt(:,3),gt(:,2),1,'black')
scatter(gps(:,3),gps(:,2),1,gps(:,1))
axis equal
colorbar
xlabel('East [m]')
xlabel('North [m]')