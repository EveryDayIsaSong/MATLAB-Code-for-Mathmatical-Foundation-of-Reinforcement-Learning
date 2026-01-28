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
v = zeros(x_length, y_length);
v_pre_out = v + 1; % 随便给v_pre一个初值，使得其能满足第一次不跳出循环的条件

% 初始化q—table
q_table = zeros(state_space, number_of_action);

% 创建图形窗口并获取句柄
figure_handle = figure; % 通过传递figure_handle，可以确保每次调用figure_plot时都在同一个图形窗口中更新图形，而不是创建新的窗口
iteration = 0;
figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, target_state, actions, iteration);

while abs(norm(v) - norm(v_pre_out)) > 0.01
    clf(figure_handle) % 清除当前图形窗口内容
    iteration = iteration + 1;
    v_pre_out = v;

    % 策略评价：
    v_pre_in = v + 0.01;
    while abs(norm(v) - norm(v_pre_in)) > 0.01
        v_pre_in = v;
        for s = 1:state_space
            x_s = ceil(s / y_length);
            y_s = mod(s - 1, y_length) + 1; % 计算状态在网格中对应的行和列
            state = [x_s, y_s];
            for a = 1:number_of_action
                if policy(s,a) == 1
                    action = actions{a};
                    [new_state, reward] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
                    v(x_s, y_s) =  reward + gamma * v(new_state(1),new_state(2));
                    break
                end
            end
        end
    end
    
    % 策略改进：
    for s = 1:state_space
        x_s = ceil(s / y_length);
        y_s = mod(s - 1, y_length) + 1; % 计算状态在网格中对应的行和列
        state = [x_s, y_s]; 

        for a = 1:number_of_action
            action = actions{a};
            [new_state, reward] = next_state_and_reward(state, action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
            q_table(s,a) = reward + gamma * v(new_state(1),new_state(2)); % 计算q值
        end
        [maxValue, index] = max(q_table(s,:)); % 最大价值动作

        for i = 1:number_of_action
            if index == i
                policy(s,i) = 1;
            else
                policy(s,i) = 0;   
            end
        end
    end
    figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, target_state, actions, iteration); % 传递句柄
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