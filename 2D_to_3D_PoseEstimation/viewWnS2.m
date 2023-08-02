function viewWnS2(W, S, Shat, edges)

    F = size(W,1)/2;
    x1 = min(min(W(1:2:end, :)));
    y1 = min(min(W(2:2:end, :)));    
    x2 = max(max(W(1:2:end, :)));
    y2 = max(max(W(2:2:end, :)));
    
    X1 = min(min(S(1:3:end, :)));
    Y1 = min(min(S(2:3:end, :)));
    Z1 = min(min(S(3:3:end, :)));
    X2 = max(max(S(1:3:end, :)));
    Y2 = max(max(S(2:3:end, :)));
    Z2 = max(max(S(3:3:end, :)));
    
    for i=1:F 
        subplot(1,2,1) 
        plot(W(2*i-1, :), W(2*i, :), 'k.'); 
        hold on;        
        drawLines2(W(2*i-1:2*i, :), 'k-',edges);        
        hold off;
        grid on
        axis equal
        axis([x1 x2 y1 y2]);  
        
        subplot(1,2,2)
        plot3(S(3*i-2, :), S(3*i, :), S(3*i-1, :), 'k.'); 
        hold on;
        drawLines(S(3*i-2:3*i, :), 'k-',edges);
        if exist('Shat')
            plot3(Shat(3*i-2, :), Shat(3*i, :), Shat(3*i-1, :), 'b.'); 
            drawLines(Shat(3*i-2:3*i, :), 'b-',edges);
        end
        hold off;
        title(sprintf('Frame: %03d',i))
        view(180, 24);  
        grid on    
        axis equal
%         axis([X1 X2 Z1 Z2 Y1 Y2]);
             
%         pause;
    end    
end

function drawLines2(W,lineStyle,edges)
    
    for b = 1:size(edges,1)
        i = edges(b,1); j = edges(b,2);
        plot([W(1,i), W(1,j)], [W(2,i), W(2, j)], lineStyle);            
    end        
end

function drawLines(S,lineStyle,edges)
    
    for b = 1:size(edges,1)
        i = edges(b,1); j = edges(b,2);
        plot3([S(1,i), S(1,j)], [S(3,i), S(3, j)], [S(2,i), S(2, j)], lineStyle, 'linewidth', 1.5);            
    end        
end

