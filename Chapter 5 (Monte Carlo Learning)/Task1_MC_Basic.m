% 在西湖大学赵世钰老师提供的MATLAB demo上继续编写代码(https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)：

clear 
close all

target_state = [4, 3];
obstacle_state = [2, 2; 2, 3; 3, 3; 4, 2; 4, 4; 5, 2];
x_length = 5;
y_length = 5;
state_space = x_length * y_length; 

reward_forbidden = -10;
reward_target = 1;
reward_step = 0;
gamma = 0.9; % 折扣因子

% Define actions: up, right, down, left, stay
actions = {[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]}; % 这里与原demo的逻辑不同，对顺序进行了修改

% Initialize a cell array to store the action space for each state
action_space = cell(state_space, 1);

% Populate the action space
for i = 1:state_space       
    action_space{i} = actions;
end

number_of_action = 5;

policy = zeros(state_space, number_of_action); % 行为state，列为action，值为对应state，action下的概率

% 初始化策略（全部为↑向上）
for i = 1:state_space
    policy(i,1) = 1;
end

% 初始化v
v = zeros(x_length, y_length); % 只在作图的时候使用，算法内不涉及

% 初始化q—table
q_table = zeros(state_space, number_of_action);

% 创建图形窗口并获取句柄
figure_handle = figure; % 通过传递figure_handle，可以确保每次调用figure_plot时都在同一个图形窗口中更新图形，而不是创建新的窗口

figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, target_state, actions, 0);

k_iteration = 10; % 迭代次数
l_episode = 100; % 收集数据的回合长度，改变这个值会影响最后得到的value
num_episode = 1; % 该环境是deterministic的，在同一个位置做一个动作，结果永远一样，因此在该环境中，采样1次和采样100次得到的结果相同

for k = 1:k_iteration
    clf(figure_handle) % 清除当前图形窗口内容
    
    for s = 1:state_space
        x_s = ceil(s / y_length);
        y_s = mod(s - 1, y_length) + 1; % 计算状态在网格中对应的行和列
        state = [x_s, y_s]; 

        for a = 1:number_of_action
            action = actions{a};
            total_G = 0;
            for i = 1:num_episode
                G = 0;
                current_state = state;
                current_action = action;
                for j = 1:l_episode
                    [new_state, reward] = next_state_and_reward(current_state, current_action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
                    G = G + gamma ^ (j-1) * reward; % 根据action value的原始定义
                    current_state = new_state;
                    current_state_idx = (current_state(1)-1) * y_length + current_state(2); % 当前状态在policy数组中的索引
                    [~, current_action_idx] = max(policy(current_state_idx,:)); 
                    current_action = actions{current_action_idx};
                end
                total_G = total_G + G;
            end
            q_table(s,a) = 1/num_episode * total_G; % 即G的期望
        end
       
        v(x_s, y_s) = max(q_table(s,:));  % 为了让画图能动，需要在这里更新v
        [maxValue, index] = max(q_table(s,:)); % 最大价值动作

        % 策略评估
        for i = 1:number_of_action
            if index == i
                policy(s,i) = 1;
            else
                policy(s,i) = 0;   
            end
        end
    end
    figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, target_state, actions, k); % 传递句柄
    pause(0.5); 
end

function [new_state, reward] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step)
    new_x = state(1) + action(1);
    new_y = state(2) + action(2);
    new_state = [new_x, new_y];

    % Check if the new state is out of bounds
    if new_x < 1 || new_x > x_length || new_y < 1 || new_y > y_length
        new_state = state;
        reward = reward_forbidden;
    elseif ismember(new_state, obstacle_state, 'rows') % 按行匹配
        % If the new state is an obstacle
        reward = reward_forbidden;
    elseif isequal(new_state, target_state)
        % If the new state is the target state
        reward = reward_target;
    else
        % If the new state is a normal cell
        reward = reward_step;
    end
end