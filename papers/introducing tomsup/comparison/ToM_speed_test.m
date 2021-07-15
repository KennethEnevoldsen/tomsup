clear all
close all
clc

% set the learning styles and interaction types of the first player
styles1 = {'5-ToM'};
modes1 = {'comp'};

% set the learning styles and interaction types of the second player
styles2 = {'RB'};
modes2 = {'comp'};

% game payof tables (competitive and cooperative interaction types)
compGame = cat(3,[1,-1;-1,1],[-1,1;1,-1]); % competitive game
coopGame = cat(3,[1,-1;-1,1],[1,-1;-1,1]); % cooperative game

%Simulation settings
ntests = 20; % number of speedtests to run
Nmc = 8; % number of simulations 
nt = 60; % number of trials

%ntests = 2; % number of speedtests to run
%Nmc = 2; % number of simulations 
%nt = 10; % number of trials


elapsedTimes = zeros(ntests,1);

for run = 1:ntests
    
    %run
    
    tic

    Perf = zeros(length(styles1),length(styles2),Nmc);
    for imc = 1:Nmc
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
                [rew,y1,y2] = runGame_2players(info1,info2,nt,0);
            end
        end
    end

    elapsedTimes(run) = toc;
    
end


% Get the mean and standard deviation for runtimes
mean(elapsedTimes)
std(elapsedTimes)
