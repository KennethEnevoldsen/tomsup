%This script is used to compare the python implementation 
% with the matlab implementation step-by-step. We suggest
% using a debugger for this. Note that the VBA package 
% for matlab needs to be added to the matlab path for the
% matlab scripts to work

clear all
close all
clc

format long

% set the learning styles and interaction types of the first player
styles1 = {'1-ToM'};
modes1 = {'comp'};

% set the learning styles and interaction types of the second player
styles2 = {'RB'};
modes2 = {'comp'};

% game payof tables (competitive and cooperative interaction types)
compGame = cat(3,[1,-1;-1,1],[-1,1;1,-1]); % competitive game
%compGame = cat(3,[-1,1;1,-1],[1,-1;-1,1]); % competitive game
coopGame = cat(3,[1,-1;-1,1],[1,-1;-1,1]); % cooperative game

%Simulation settings
nt = 11; % number of trials

for i=1:length(styles1)
    % prepare player 1
    if isequal(modes1{i},'comp')
        info1.payoffTable = compGame;
    else
        info1.payoffTable = coopGame;
    end
    [info1] = prepare_agent(styles1{i},info1.payoffTable,1);
    for j=1:length(styles2)
        % prepare player 2
        if isequal(modes2{j},'comp')
            info2.payoffTable = compGame;
        else
            info2.payoffTable = coopGame;
        end
        [info2] = prepare_agent(styles2{j},info2.payoffTable,2);
        % simulate game
        [rew,y1,y2] = VBA_behavior_comparison_helper(info1,info2,nt,0);
    end
end



