function drawLines(S,lineStyle,edges)
    
    for b = 1:size(edges,1)
        i = edges(b,1); j = edges(b,2);
        plot3([S(1,i), S(1,j)], [S(3,i), S(3, j)], [S(2,i), S(2, j)], lineStyle);            
    end        
end