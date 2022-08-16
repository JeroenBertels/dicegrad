function [p_values] = stat_sign_boot(cum_ranks,rank_cum)
    p_values = zeros(size(cum_ranks,1));
    no_it = 100000;
    for r = 1:size(cum_ranks,1)
        for s = 1:size(cum_ranks,1)
            ranks1 = cum_ranks(r,:);
            ranks2 = cum_ranks(s,:);
            FRS_diff = abs(rank_cum(r)-rank_cum(s));
            ranks1_shuffle = nan(size(ranks1));
            ranks2_shuffle = nan(size(ranks2));
            for k = 1:no_it
                random_shuffle = logical(randi([0 1],length(ranks1),1));
                ranks1_shuffle(random_shuffle) = ranks1(random_shuffle);
                ranks1_shuffle(~random_shuffle) = ranks2(~random_shuffle);
                ranks2_shuffle(random_shuffle) = ranks2(random_shuffle);
                ranks2_shuffle(~random_shuffle) = ranks1(~random_shuffle);
                FRS_diff_permuted = abs(mean(ranks1_shuffle)-mean(ranks2_shuffle));
                if FRS_diff_permuted>FRS_diff
                    p_values(r,s) = p_values(r,s)+1;
                end
            end
        end
    end
    p_values = p_values/no_it;
end

