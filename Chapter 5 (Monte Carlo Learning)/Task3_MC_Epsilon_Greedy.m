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
epsilon = 0.3; % 初始值，在后续循环内逐渐变小

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

% 初始化策略（随机）
for s = 1:state_space
    random_a = randi(number_of_action);
    policy(s, random_a) = 1;
end

% 初始化v
v = zeros(x_length, y_length); % 只在作图的时候使用，算法内不涉及

q_table = zeros(state_space, number_of_action); % 初始化q—table
returns = zeros(state_space, number_of_action); % q—table中每个动作状态对的总回报
numbers = zeros(state_space, number_of_action); % q—table中每个动作状态对的总次数

% 创建图形窗口并获取句柄
figure_handle = figure; % 通过传递figure_handle，可以确保每次调用figure_plot时都在同一个图形窗口中更新图形，而不是创建新的窗口

figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, target_state, actions, 0);

k_iteration = 100000; % 迭代次数
l_episode = 5000; % 收集数据的回合长度，改变这个值会影响最后得到的value
num_episode = 1; % 该环境是deterministic的，在同一个位置做一个动作，结果永远一样，因此在该环境中，采样1次和采样100次得到的结果相同

for k = 1:k_iteration

    s_start = randi(state_space);
    x_s = ceil(s_start / y_length);
    y_s = mod(s_start - 1, y_length) + 1; % 计算状态在网格中对应的行和列
    state = [x_s, y_s];

    probalities = policy(s_start, :); % 获取当前状态的动作概率分布π(a|s)
    a_start = randsample(number_of_action, 1, true, probalities); % 1-5中抽1个数，true表示放回/独立，权重为probalities
    action = actions{a_start};  

    % 回合生成
    trajectory = zeros(l_episode, 3); % 每一行代表状态、动作、奖励
    current_state = state;
    current_state_idx = s_start;
    current_action = action;
    current_action_idx = a_start; % Exploring Starts
    for i = 1:l_episode
        trajectory(i,1) = current_state_idx; % s_t对应的索引
        trajectory(i,2) = current_action_idx; % a_t对应的索引

        [new_state, reward] = next_state_and_reward(current_state, current_action, x_length, y_length, target_state, obstacle_state, reward_forbidden, reward_target, reward_step);
        
        trajectory(i,3) = reward; % r_t+1
        l_real = i;

        % 到达目标终止，因为发现如果到达目标后仍然继续收集数据，最后得到的策略效果并不好，同时这么做能加快运行速度
        if isequal(new_state, target_state)
            trajectory = trajectory(1:l_real, :); % 截断
            break; 
        end

        current_state = new_state;
        current_state_idx = (current_state(1)-1) * y_length + current_state(2); % 当前状态在policy数组中的索引
        
        probalities = policy(current_state_idx, :); % 获取当前状态的动作概率分布π(a|s)
        current_action_idx = randsample(number_of_action, 1, true, probalities); % 1-5中抽1个数，true表示放回/独立，权重为probalities
        current_action = actions{current_action_idx};    
    end
            
    g = 0;
    % 对回合的每一步处理
    for t = l_real:-1:1
        s_t = trajectory(t, 1);
        a_t = trajectory(t, 2);
        r_next = trajectory(t, 3);
        g = gamma * g + r_next; % g ← γg + r
        returns(s_t, a_t) = returns(s_t, a_t) + g; % Return(s,a) ← Return(s,a) + g
        numbers(s_t, a_t) = numbers(s_t, a_t) + 1; % Number(s,a) ← Number(s,a) + 1
        q_table(s_t, a_t) = returns(s_t, a_t) / numbers(s_t, a_t); % 策略评价
        
        % 策略改进
        [maxValue, index] = max(q_table(s_t, :)); % 最大价值动作
        policy(s_t,:) = 1 / number_of_action * epsilon;
        policy(s_t,index) = 1 - ((number_of_action - 1) / number_of_action * epsilon);
        
        % 画图
        x_t = ceil(s_t / y_length);
        y_t = mod(s_t - 1, y_length) + 1; % 计算状态在网格中对应的行和列
        v(x_t, y_t) = max(q_table(s_t,:)); 
    end

    if mod(k, 500) == 0 
        clf(figure_handle);
        figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, target_state, actions, k); 
        drawnow;
    end

    if epsilon > 0.1
        epsilon = epsilon - (0.3 - 0.1) / 50000; % ε线性减小
    end
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