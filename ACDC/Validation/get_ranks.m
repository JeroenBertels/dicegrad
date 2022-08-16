function [rank,p_values,ranks_tot] = get_ranks(array,dim,order,remove_nans)
    if ~exist('remove_nans','var')
        remove_nans = 0;
    end
    if iscell(array)
        array_tot = [];
        for r = 1:length(array)
            if dim ==3
                array_tot = cat(3,array_tot,array{r});
            else
                array_tot = [array_tot array{r}];
            end
        end
    else
        array_tot = array;
    end

    if remove_nans
        idx_remove = find(prod(isnan(array_tot),dim));
        array_tot(idx_remove,:)=[];
    end
    
    [~,ranks1] = sort(array_tot,dim,order);
    [~,ranks1] = sort(ranks1,dim);
    
    [~,ranks2] = sort(flip(array_tot,dim),dim,order);
    [~,ranks2] = sort(ranks2,dim);
    
    ranks_tot = (ranks1 + flip(ranks2,dim))/2;
    if dim == 3
        ranks_tot= squeeze(mean(ranks_tot,2));
    end
    
    rank = mean(ranks_tot);
    p_values = stat_sign_boot(ranks_tot',rank');
end

