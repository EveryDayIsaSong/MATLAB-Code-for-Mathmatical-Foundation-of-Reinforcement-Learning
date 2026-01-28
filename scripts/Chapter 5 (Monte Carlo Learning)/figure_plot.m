function figure_plot(figure_handle, v, policy, x_length, y_length, obstacle_state, final_state, actions, iteration)
    set(0, 'CurrentFigure', figure_handle); % 设置当前图形窗口
    
    % 绘制策略子图
    subplot(1,2,1); % 图形窗口被划分为1行2L列，选择第一个子图
    hold on;
    % 绘制所有格子和策略字符
    for x_s = 1:x_length
        for y_s = 1:y_length
            i = y_s;  % 水平方向（列）
            j = y_length - x_s + 1;  % 垂直方向（行，翻转使 (1,1) 在左上角）
            rectangle('Position', [i, j, 1, 1]);
            s = (x_s - 1) * y_length + y_s;
            [~, a] = max(policy(s,:));
            switch a
                case 1 % 动作 [-1, 0] 代表向上（列不变，行减一，这里对应关系很容易搞错！）
                    symbol = '↑';
                case 2 % 动作 [0, 1] 代表向右
                    symbol = '→';
                case 3 % 动作 [1, 0] 代表向下
                    symbol = '↓';
                case 4 % 动作 [0, -1] 代表向左
                    symbol = '←';
                case 5 % 动作 [0, 0] 代表不动
                    symbol = 'o';
            end
            text(i + 0.5, j + 0.5, symbol, 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'k'); % k代表黑色
        end
    end
    % 绘制障碍物和目标状态的颜色
    for k = 1:size(obstacle_state,1)
        x_o = obstacle_state(k,1);  % 行
        y_o = obstacle_state(k,2);  % 列
        i = y_o;  % 水平
        j = y_length - x_o + 1;  % 垂直
        r = rectangle('Position', [i, j, 1, 1], 'FaceColor', [0.9290 0.6940 0.1250]); % FaceColor指定填充颜色
        uistack(r, 'bottom'); % 将颜色层移到最底层
    end
    x_f = final_state(1);  % 行
    y_f = final_state(2);  % 列
    i = y_f;  % 水平
    j = y_length - x_f + 1;  % 垂直
    r = rectangle('Position', [i, j, 1, 1], 'FaceColor', [0.3010 0.7450 0.9330]);
    uistack(r, 'bottom'); % 将颜色层移到最底层
    % 在网格上方添加列标签
    for col = 1:y_length
        text(col + 0.5, y_length + 1.5, num2str(col), 'HorizontalAlignment', 'center');
    end
    % 在网格左侧添加行标签
    for row = 1:x_length
        text(0.5, y_length - row + 1 + 0.5, num2str(row), 'HorizontalAlignment', 'center');
    end
    axis equal;
    axis off;
    title('Policy');
    
    % 绘制状态值子图
    subplot(1,2,2);
    hold on;
    % 先绘制所有格子和状态值
    for x_s = 1:x_length
        for y_s = 1:y_length
            i = y_s;  % 水平方向（列）
            j = y_length - x_s + 1;  % 垂直方向（行，翻转）
            rectangle('Position', [i, j, 1, 1]);
            text(i + 0.5, j + 0.5, num2str(round(v(x_s, y_s), 1)), 'HorizontalAlignment', 'center', 'FontSize', 10); % State Value
        end
    end
    % 再绘制障碍物和目标状态的颜色
    for k = 1:size(obstacle_state,1)
        x_o = obstacle_state(k,1);  % 行
        y_o = obstacle_state(k,2);  % 列
        i = y_o;  % 水平
        j = y_length - x_o + 1;  % 垂直
        r = rectangle('Position', [i, j, 1, 1], 'FaceColor', [0.9290 0.6940 0.1250]);
        uistack(r, 'bottom'); % 将颜色层移到最底层
    end
    x_f = final_state(1);  % 行
    y_f = final_state(2);  % 列
    i = y_f;  % 水平
    j = y_length - x_f + 1;  % 垂直
    r = rectangle('Position', [i, j, 1, 1], 'FaceColor', [0.3010 0.7450 0.9330]);
    uistack(r, 'bottom'); % 将颜色层移到最底层
    % 在网格上方添加列标签
    for col = 1:y_length
        text(col + 0.5, y_length + 1.5, num2str(col), 'HorizontalAlignment', 'center');
    end
    % 在网格左侧添加行标签
    for row = 1:x_length
        text(0.5, y_length - row + 1 + 0.5, num2str(row), 'HorizontalAlignment', 'center');
    end
    axis equal;
    axis off;
    title('State Value');
    
    % 在图形窗口底部添加当前迭代次数
    annotation('textbox', [0.45, 0.1, 0.1, 0.1], 'String', ['Iteration: ', num2str(iteration)], 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    % [x, y, width, height]定义文本框textbox的位置和大小，值都在0到1之间，例：x为文本框左下角的水平位置，0和1分别表示窗口最左、右边；width为文本框的宽度，相对于窗口宽度的比例

    drawnow; % 立即更新图形
end